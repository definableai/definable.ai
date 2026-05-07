"""Slack interface — Socket Mode bot via slack-bolt.

Minimal port: connect via Socket Mode, listen for `message` and
`app_mention` events, reply via WebClient.

Removed (vs original): HTTP webhook mode, slash commands, Block Kit
actions, modal submissions, shortcuts, reaction events, file uploads.
Each can be added back by extending this class or subscribing to
`agent.events`.

Requires `pip install slack-bolt slack-sdk`.

Usage::

    iface = SlackInterface(agent, bot_token="xoxb-...", app_token="xapp-...")
    async with iface:
      await iface.serve()
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from definable.agent.interface.base import Interface
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class SlackInterface(Interface):
  """Socket Mode Slack bot bound to one Agent."""

  def __init__(
    self,
    agent: Agent,
    *,
    bot_token: str,
    app_token: str,
    allowed_channel_ids: list[str] | None = None,
    allowed_user_ids: list[str] | None = None,
    respond_to_bots: bool = False,
    require_mention: bool = False,
  ) -> None:
    super().__init__(agent)
    self.bot_token = bot_token
    self.app_token = app_token
    self.allowed_channel_ids = allowed_channel_ids
    self.allowed_user_ids = allowed_user_ids
    self.respond_to_bots = respond_to_bots
    self.require_mention = require_mention

    self._bolt_app: Any = None
    self._client: Any = None
    self._socket_handler: Any = None
    self._bot_user_id: str | None = None

  # ---- Interface contract -------------------------------------------------

  async def aopen(self) -> None:
    try:
      from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
      from slack_bolt.async_app import AsyncApp
    except ImportError as e:
      raise ImportError("SlackInterface requires slack-bolt — `pip install slack-bolt slack-sdk`") from e

    self._bolt_app = AsyncApp(token=self.bot_token)
    self._client = self._bolt_app.client

    # Identify bot for self-message filtering
    auth = await self._client.auth_test()
    self._bot_user_id = auth.get("user_id")
    log_info(f"[slack] connected as @{auth.get('user')} (user_id={self._bot_user_id})")

    @self._bolt_app.event("message")
    async def _on_message(event: dict[str, Any], say: Any) -> None:
      del say
      if self.require_mention:
        return  # only respond to app_mention in this mode
      await self.handle(event)

    @self._bolt_app.event("app_mention")
    async def _on_mention(event: dict[str, Any], say: Any) -> None:
      del say
      await self.handle(event)

    self._socket_handler = AsyncSocketModeHandler(self._bolt_app, self.app_token)
    await self._socket_handler.connect_async()
    log_info("[slack] Socket Mode connected")

  async def aclose(self) -> None:
    if self._socket_handler is not None:
      with contextlib.suppress(Exception):
        await self._socket_handler.close_async()
      self._socket_handler = None
    self._bolt_app = None
    self._client = None

  async def _convert(self, raw_message: Any) -> str:
    event = raw_message
    if event.get("subtype"):
      return ""  # skip edits, deletes, joins etc
    user_id = event.get("user")
    channel_id = event.get("channel")
    if user_id == self._bot_user_id:
      return ""
    bot_id = event.get("bot_id")
    if bot_id and not self.respond_to_bots:
      return ""
    if self.allowed_channel_ids is not None and channel_id not in self.allowed_channel_ids:
      return ""
    if self.allowed_user_ids is not None and user_id not in self.allowed_user_ids:
      return ""
    text = event.get("text", "")
    # Strip bot mention if present
    if self._bot_user_id and f"<@{self._bot_user_id}>" in text:
      text = text.replace(f"<@{self._bot_user_id}>", "").strip()
    return text

  async def _send(self, raw_message: Any, reply: str) -> None:
    event = raw_message
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    if channel is None:
      return
    try:
      await self._client.chat_postMessage(channel=channel, text=reply, thread_ts=thread_ts)
    except Exception as e:
      log_error(f"[slack] send failed: {e}")
