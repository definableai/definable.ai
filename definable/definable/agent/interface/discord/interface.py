"""Discord interface implementation using discord.py."""

import asyncio
import contextlib
from typing import Any, List, Optional

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.errors import (
  InterfaceConnectionError,
  InterfaceMessageError,
)
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.discord.config import DiscordConfig
from definable.media import Audio, File, Image
from definable.utils.log import log_debug, log_info


class DiscordInterface(BaseInterface):
  """Interface connecting an agent to Discord via discord.py.

  Uses the discord.py library to connect to the Discord gateway and
  receive messages in real time. Requires the MESSAGE_CONTENT privileged
  intent to be enabled in the Discord Developer Portal.

  Args:
    agent: The Agent instance.
    config: DiscordConfig with bot token and settings.
    session_manager: Optional session manager.
    hooks: Optional list of hooks.

  Example:
    interface = DiscordInterface(
      agent=agent,
      config=DiscordConfig(bot_token="BOT_TOKEN"),
    )
    async with interface:
      await interface.serve_forever()
  """

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._dc_config: DiscordConfig = self.config  # type: ignore[assignment]
    self._client: Any = None  # discord.Client, typed as Any to avoid import at module level
    self._bot_task: Optional[asyncio.Task[None]] = None
    self._ready_event: Optional[asyncio.Event] = None

  # --- Lifecycle ---

  async def _start_receiver(self) -> None:
    try:
      import discord
    except ImportError:
      raise InterfaceConnectionError(
        "discord.py is required for DiscordInterface. Install it with: pip install 'definable[discord]'",
        platform="discord",
      )

    # Configure intents
    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    if self._dc_config.intents_message_content:
      intents.message_content = True

    self._client = discord.Client(intents=intents)
    self._ready_event = asyncio.Event()

    # Register event handlers
    @self._client.event
    async def on_ready() -> None:
      assert self._client is not None
      log_info(f"[discord] Connected as {self._client.user}")
      assert self._ready_event is not None
      self._ready_event.set()

    @self._client.event
    async def on_message(message: Any) -> None:
      await self.handle_platform_message(message)

    # Start the bot in a background task
    self._bot_task = asyncio.create_task(self._client.start(self._dc_config.bot_token))

    # Wait for the bot to be ready (or fail fast on bad token)
    try:
      await asyncio.wait_for(self._ready_event.wait(), timeout=self._dc_config.connect_timeout)
    except asyncio.TimeoutError:
      await self._cleanup_client()
      raise InterfaceConnectionError(
        "Timed out waiting for Discord gateway connection",
        platform="discord",
      )
    except Exception:
      await self._cleanup_client()
      raise

  async def _stop_receiver(self) -> None:
    await self._cleanup_client()

  async def _cleanup_client(self) -> None:
    """Close the discord client and cancel the bot task."""
    if self._client is not None:
      with contextlib.suppress(Exception):
        await self._client.close()
      self._client = None

    if self._bot_task is not None:
      self._bot_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._bot_task
      self._bot_task = None

  # --- Inbound conversion ---

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """Convert a discord.Message to InterfaceMessage."""
    # Skip messages from the bot itself
    if self._client is not None and raw_message.author == self._client.user:
      return None

    # Skip bot messages unless configured to respond
    if raw_message.author.bot and not self._dc_config.respond_to_bots:
      log_debug(f"[discord] Ignoring message from bot {raw_message.author.id}")
      return None

    # Guild access control
    if self._dc_config.allowed_guild_ids is not None:
      guild = getattr(raw_message, "guild", None)
      if guild is None or guild.id not in self._dc_config.allowed_guild_ids:
        log_debug("[discord] Ignoring message from unauthorized guild")
        return None

    # Channel access control
    if self._dc_config.allowed_channel_ids is not None:
      if raw_message.channel.id not in self._dc_config.allowed_channel_ids:
        log_debug(f"[discord] Ignoring message from unauthorized channel {raw_message.channel.id}")
        return None

    text = raw_message.content

    # Command prefix filtering
    if self._dc_config.command_prefix is not None:
      if not text.startswith(self._dc_config.command_prefix):
        return None
      # Strip the prefix from the text
      text = text[len(self._dc_config.command_prefix) :].strip()

    # Extract media from attachments
    images: Optional[List[Image]] = None
    audio_list: Optional[List[Audio]] = None
    files: Optional[List[File]] = None

    for attachment in raw_message.attachments:
      content_type = attachment.content_type or ""
      if content_type.startswith("image/"):
        if images is None:
          images = []
        images.append(Image(url=attachment.url))
      elif content_type.startswith("audio/"):
        if audio_list is None:
          audio_list = []
        audio_list.append(Audio(url=attachment.url, mime_type=content_type))
      else:
        if files is None:
          files = []
        files.append(
          File(
            url=attachment.url,
            mime_type=content_type or None,
            filename=attachment.filename,
            size=attachment.size,
          )
        )

    # Reply context
    reply_to_message_id: Optional[str] = None
    if raw_message.reference and raw_message.reference.message_id:
      reply_to_message_id = str(raw_message.reference.message_id)

    username = raw_message.author.display_name or raw_message.author.name

    return InterfaceMessage(
      text=text or None,
      platform="discord",
      platform_user_id=str(raw_message.author.id),
      platform_chat_id=str(raw_message.channel.id),
      platform_message_id=str(raw_message.id),
      username=username,
      images=images,
      audio=audio_list,
      files=files,
      reply_to_message_id=reply_to_message_id,
      metadata={"raw": raw_message},
    )

  # --- Response sending ---

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send response back to Discord."""
    channel = raw_message.channel

    # Show typing indicator
    if self.config.typing_indicator:
      with contextlib.suppress(Exception):
        await channel.typing()

    # Send text content (split if needed)
    if response.content:
      max_len = self._dc_config.max_message_length
      chunks = self._split_text(response.content, max_len)

      for i, chunk in enumerate(chunks):
        try:
          if i == 0:
            await raw_message.reply(chunk)
          else:
            await channel.send(chunk)
        except Exception as e:
          raise InterfaceMessageError(
            f"Failed to send message: {e}",
            platform="discord",
          ) from e

    # Send images as file attachments
    if response.images:
      for image in response.images:
        await self._send_image(channel, image)

    # Send files
    if response.files:
      for file in response.files:
        await self._send_file(channel, file)

  async def _send_image(self, channel: Any, image: Image) -> None:
    """Send an image to a Discord channel."""
    import discord as discord_lib

    if image.url:
      # Send URL as embed
      embed = discord_lib.Embed()
      embed.set_image(url=image.url)
      await channel.send(embed=embed)
    elif image.filepath:
      await channel.send(file=discord_lib.File(str(image.filepath)))
    elif image.content:
      import io

      fp = io.BytesIO(image.content)
      filename = f"image.{image.format or 'png'}"
      await channel.send(file=discord_lib.File(fp, filename=filename))

  async def _send_file(self, channel: Any, file: File) -> None:
    """Send a file to a Discord channel."""
    import discord as discord_lib

    if file.filepath:
      await channel.send(file=discord_lib.File(str(file.filepath)))
    elif file.url:
      # Send the URL as a message
      await channel.send(file.url)

  # --- Utilities ---

  @staticmethod
  def _split_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks respecting max_length.

    Tries to split at newlines, then at spaces, falling back to
    hard splits if necessary.
    """
    if len(text) <= max_length:
      return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
      if len(remaining) <= max_length:
        chunks.append(remaining)
        break

      # Try to split at a newline
      split_pos = remaining.rfind("\n", 0, max_length)
      if split_pos == -1:
        # Try to split at a space
        split_pos = remaining.rfind(" ", 0, max_length)
      if split_pos == -1:
        # Hard split
        split_pos = max_length

      chunks.append(remaining[:split_pos])
      remaining = remaining[split_pos:].lstrip("\n")

    return chunks
