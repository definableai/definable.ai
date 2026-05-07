"""Discord interface using discord.py.

Connects an Agent to a Discord bot. Receives messages via the gateway,
runs each through `agent.arun`, and replies in the same channel.

Requires `pip install definable[discord]` (pulls in discord.py).
The MESSAGE_CONTENT privileged intent must be enabled in the Discord
Developer Portal for the bot to see message content.

Usage::

    iface = DiscordInterface(agent, bot_token="...")
    async with iface:
      await iface.serve()
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from definable.agent.interface.base import Interface
from definable.utils.log import log_debug, log_info

if TYPE_CHECKING:
  import discord

  from definable.agent.agent import Agent


class DiscordInterface(Interface):
  """Bidirectional Discord bot bound to one Agent."""

  def __init__(
    self,
    agent: Agent,
    *,
    bot_token: str,
    intents_message_content: bool = True,
    allowed_guild_ids: list[int] | None = None,
    allowed_channel_ids: list[int] | None = None,
    respond_to_bots: bool = False,
    command_prefix: str | None = None,
    connect_timeout: float = 30.0,
  ) -> None:
    super().__init__(agent)
    self.bot_token = bot_token
    self.intents_message_content = intents_message_content
    self.allowed_guild_ids = allowed_guild_ids
    self.allowed_channel_ids = allowed_channel_ids
    self.respond_to_bots = respond_to_bots
    self.command_prefix = command_prefix
    self.connect_timeout = connect_timeout

    self._client: discord.Client | None = None
    self._bot_task: asyncio.Task[None] | None = None
    self._ready_event: asyncio.Event | None = None

  # ---- Interface contract -------------------------------------------------

  async def aopen(self) -> None:
    try:
      import discord
    except ImportError as e:
      raise ImportError("DiscordInterface requires discord.py — `pip install definable[discord]`") from e

    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    if self.intents_message_content:
      intents.message_content = True

    self._client = discord.Client(intents=intents)
    self._ready_event = asyncio.Event()

    @self._client.event
    async def on_ready() -> None:
      log_info(f"[discord] connected as {self._client.user if self._client else '?'}")
      if self._ready_event is not None:
        self._ready_event.set()

    @self._client.event
    async def on_message(message: discord.Message) -> None:
      await self.handle(message)

    self._bot_task = asyncio.create_task(self._client.start(self.bot_token))
    try:
      await asyncio.wait_for(self._ready_event.wait(), timeout=self.connect_timeout)
    except (asyncio.TimeoutError, Exception):
      await self.aclose()
      raise

  async def aclose(self) -> None:
    if self._client is not None:
      with contextlib.suppress(Exception):
        await self._client.close()
      self._client = None
    if self._bot_task is not None:
      self._bot_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._bot_task
      self._bot_task = None

  async def _convert(self, raw_message: Any) -> str:
    msg = raw_message  # discord.Message
    if self._client is not None and msg.author == self._client.user:
      return ""
    if msg.author.bot and not self.respond_to_bots:
      log_debug(f"[discord] ignoring bot message {msg.author.id}")
      return ""
    if self.allowed_guild_ids is not None:
      guild = getattr(msg, "guild", None)
      if guild is None or guild.id not in self.allowed_guild_ids:
        return ""
    if self.allowed_channel_ids is not None and msg.channel.id not in self.allowed_channel_ids:
      return ""

    text = msg.content
    if self.command_prefix is not None:
      if not text.startswith(self.command_prefix):
        return ""
      text = text[len(self.command_prefix) :].strip()
    return text

  async def _send(self, raw_message: Any, reply: str) -> None:
    await raw_message.channel.send(reply)
