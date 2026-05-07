"""WhatsApp interface — Twilio REST + webhook.

Minimal port: Twilio only. Sends via REST POST to Twilio's Messages API,
receives via FastAPI webhook the user mounts on their public URL.

Removed (vs original): Baileys self-hosted Node sidecar provider, the
pluggable provider abstraction, policy/allowlist module, markdown-to-
WhatsApp formatting, E.164 normalization helpers. The two production
agents (E-Garuda + Clinic) wire Baileys directly via repo-root code;
bringing it into the framework added complexity without proving its
weight.

Requires `pip install definable[serve]` (fastapi + uvicorn).

Usage::

    iface = WhatsAppInterface(
      agent,
      account_sid="AC...",
      auth_token="...",
      from_number="whatsapp:+14155238886",
      host="0.0.0.0",
      port=8800,
      webhook_path="/whatsapp",
    )
    async with iface:
      await iface.serve()

Configure Twilio's WhatsApp Sandbox / phone number to POST inbound
messages at `https://<your-host>:<port>/whatsapp`.
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


class WhatsAppInterface(Interface):
  """Twilio WhatsApp adapter — REST send + webhook receive."""

  def __init__(
    self,
    agent: Agent,
    *,
    account_sid: str,
    auth_token: str,
    from_number: str,
    host: str = "127.0.0.1",
    port: int = 8800,
    webhook_path: str = "/whatsapp",
    allowed_numbers: list[str] | None = None,
  ) -> None:
    super().__init__(agent)
    self.account_sid = account_sid
    self.auth_token = auth_token
    self.from_number = from_number
    self.host = host
    self.port = port
    self.webhook_path = webhook_path
    self.allowed_numbers = allowed_numbers

    self._client: httpx.AsyncClient | None = None
    self._server: Any = None
    self._serve_task: asyncio.Task[Any] | None = None

  async def aopen(self) -> None:
    try:
      import uvicorn
      from fastapi import FastAPI, Form
      from fastapi.responses import PlainTextResponse
    except ImportError as e:
      raise ImportError("WhatsAppInterface requires fastapi + uvicorn — `pip install definable[serve]`") from e

    self._client = httpx.AsyncClient(timeout=30.0, auth=(self.account_sid, self.auth_token))
    app = FastAPI()

    @app.post(self.webhook_path, response_class=PlainTextResponse)
    async def _webhook(
      Body: str = Form(""),  # noqa: N803 — Twilio param name
      From: str = Form(""),  # noqa: N803
      To: str = Form(""),  # noqa: N803
    ) -> str:
      raw = {"Body": Body, "From": From, "To": To}
      asyncio.create_task(self.handle(raw))
      return ""

    config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
    self._server = uvicorn.Server(config)
    self._serve_task = asyncio.create_task(self._server.serve())
    await asyncio.sleep(0.05)
    log_info(f"[whatsapp] webhook listening at {self.host}:{self.port}{self.webhook_path}")

  async def aclose(self) -> None:
    if self._server is not None:
      self._server.should_exit = True
    if self._serve_task is not None:
      self._serve_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._serve_task
      self._serve_task = None
    if self._client is not None:
      await self._client.aclose()
      self._client = None
    log_info("[whatsapp] stopped")

  async def _convert(self, raw_message: Any) -> str:
    sender = raw_message.get("From", "")
    if self.allowed_numbers is not None:
      bare = sender.replace("whatsapp:", "")
      if sender not in self.allowed_numbers and bare not in self.allowed_numbers:
        return ""
    return raw_message.get("Body", "") or ""

  async def _send(self, raw_message: Any, reply: str) -> None:
    sender = raw_message.get("From", "")
    if not sender or self._client is None:
      return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
    try:
      r = await self._client.post(url, data={"From": self.from_number, "To": sender, "Body": reply})
      r.raise_for_status()
    except Exception as e:
      log_error(f"[whatsapp] send to {sender} failed: {e}")
