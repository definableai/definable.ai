"""Telegram interface — long-poll Bot API.

Minimal port: bot_token + getUpdates poll + sendMessage reply. Allowlist
filtering on user_id and chat_id. Markdown reply format.

Removed (vs original): per-chat typing circuit breaker, sticker cache,
sliding-window rate limiter, formatting helpers, agent-controlled
inline keyboards. Each can be re-added later by overriding `handle` or
subscribing to `agent.events`.

Usage::

    iface = TelegramInterface(agent, bot_token="...")
    async with iface:
      await iface.serve()
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import httpx

from definable.agent.interface.base import Interface
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class TelegramInterface(Interface):
  """Long-poll Telegram bot bound to one Agent."""

  API_BASE = "https://api.telegram.org/bot{token}"

  def __init__(
    self,
    agent: Agent,
    *,
    bot_token: str,
    allowed_user_ids: list[int] | None = None,
    allowed_chat_ids: list[int] | None = None,
    poll_timeout: int = 30,
    parse_mode: str | None = "Markdown",
  ) -> None:
    super().__init__(agent)
    self.bot_token = bot_token
    self.allowed_user_ids = allowed_user_ids
    self.allowed_chat_ids = allowed_chat_ids
    self.poll_timeout = poll_timeout
    self.parse_mode = parse_mode

    self._client: httpx.AsyncClient | None = None
    self._poll_task: asyncio.Task[Any] | None = None
    self._update_offset: int = 0

  # ---- Interface contract -------------------------------------------------

  async def aopen(self) -> None:
    if not self.bot_token:
      raise ValueError("bot_token is required")
    self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.poll_timeout + 5))
    me = await self._call("getMe")
    log_info(f"[telegram] connected as @{me.get('username')}")
    self._poll_task = asyncio.create_task(self._poll_loop())

  async def aclose(self) -> None:
    if self._poll_task is not None:
      self._poll_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._poll_task
      self._poll_task = None
    if self._client is not None:
      await self._client.aclose()
      self._client = None

  async def _convert(self, raw_message: Any) -> str:
    """raw_message is the inner `message` dict from a Telegram Update."""
    msg = raw_message.get("message") or raw_message.get("edited_message") or {}
    user_id = msg.get("from", {}).get("id")
    chat_id = msg.get("chat", {}).get("id")
    if self.allowed_user_ids is not None and user_id not in self.allowed_user_ids:
      return ""
    if self.allowed_chat_ids is not None and chat_id not in self.allowed_chat_ids:
      return ""
    return msg.get("text") or msg.get("caption") or ""

  async def _send(self, raw_message: Any, reply: str) -> None:
    msg = raw_message.get("message") or raw_message.get("edited_message") or {}
    chat_id = msg.get("chat", {}).get("id")
    if chat_id is None:
      return
    payload = {"chat_id": chat_id, "text": reply}
    if self.parse_mode:
      payload["parse_mode"] = self.parse_mode
    try:
      await self._call("sendMessage", payload)
    except Exception as e:
      # Markdown parse can fail — retry plain text
      log_error(f"[telegram] send failed ({e}); retrying as plain text")
      payload.pop("parse_mode", None)
      with contextlib.suppress(Exception):
        await self._call("sendMessage", payload)

  # ---- polling ------------------------------------------------------------

  async def _poll_loop(self) -> None:
    try:
      while True:
        try:
          updates = await self._call(
            "getUpdates",
            {"offset": self._update_offset, "timeout": self.poll_timeout},
          )
          for upd in updates:
            self._update_offset = max(self._update_offset, upd["update_id"] + 1)
            await self.handle(upd)
        except asyncio.CancelledError:
          raise
        except Exception as e:
          log_error(f"[telegram] poll error: {e}")
          await asyncio.sleep(2.0)
    except asyncio.CancelledError:
      pass

  # ---- HTTP helper --------------------------------------------------------

  async def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
    assert self._client is not None
    url = self.API_BASE.format(token=self.bot_token) + "/" + method
    r = await self._client.post(url, json=payload or {})
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
      raise RuntimeError(f"telegram error: {data}")
    return data.get("result")
