"""Tests for the Discord interface (mocked — no real Discord connection)."""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.interfaces.discord.config import DiscordConfig
from definable.interfaces.discord.interface import DiscordInterface
from definable.interfaces.errors import InterfaceError


# --- Config tests ---


class TestDiscordConfig:
  def test_valid_config(self):
    config = DiscordConfig(bot_token="bot_token_here")
    assert config.platform == "discord"
    assert config.bot_token == "bot_token_here"
    assert config.max_message_length == 2000
    assert config.intents_message_content is True
    assert config.respond_to_bots is False
    assert config.command_prefix is None

  def test_missing_token_raises(self):
    with pytest.raises(InterfaceError, match="bot_token is required"):
      DiscordConfig(bot_token="")

  def test_defaults(self):
    config = DiscordConfig(bot_token="tok")
    assert config.allowed_guild_ids is None
    assert config.allowed_channel_ids is None
    assert config.connect_timeout == 30.0

  def test_access_control_fields(self):
    config = DiscordConfig(
      bot_token="tok",
      allowed_guild_ids=[111, 222],
      allowed_channel_ids=[333],
    )
    assert config.allowed_guild_ids == [111, 222]
    assert config.allowed_channel_ids == [333]

  def test_command_prefix(self):
    config = DiscordConfig(bot_token="tok", command_prefix="!ask")
    assert config.command_prefix == "!ask"


# --- Helper to build mock discord.Message ---


def _make_discord_message(
  content: str = "Hello",
  author_id: int = 100,
  author_name: str = "TestUser",
  author_display_name: str = "Test User",
  author_bot: bool = False,
  channel_id: int = 200,
  guild_id: int = 300,
  message_id: int = 400,
  attachments: Optional[list] = None,
  reference: object = None,
) -> MagicMock:
  """Build a mock discord.Message object."""
  msg = MagicMock()
  msg.content = content
  msg.id = message_id

  msg.author = MagicMock()
  msg.author.id = author_id
  msg.author.name = author_name
  msg.author.display_name = author_display_name
  msg.author.bot = author_bot

  msg.channel = MagicMock()
  msg.channel.id = channel_id

  msg.guild = MagicMock()
  msg.guild.id = guild_id

  msg.attachments = attachments or []
  msg.reference = reference

  return msg


# --- Inbound conversion tests ---


class TestDiscordInbound:
  @pytest.fixture
  def interface(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="test_token"),
    )
    # Simulate a connected client with a bot user
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 999  # The bot's own user ID
    return iface

  @pytest.mark.asyncio
  async def test_convert_text_message(self, interface):
    raw = _make_discord_message(content="Hello bot", author_id=100)
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Hello bot"
    assert msg.platform == "discord"
    assert msg.platform_user_id == "100"
    assert msg.platform_chat_id == "200"
    assert msg.platform_message_id == "400"
    assert msg.username == "Test User"

  @pytest.mark.asyncio
  async def test_skip_self_messages(self, interface):
    raw = _make_discord_message()
    # Make the message author be the bot itself
    raw.author = interface._client.user
    msg = await interface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_skip_bot_messages(self, interface):
    raw = _make_discord_message(author_bot=True)
    msg = await interface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_respond_to_bots_when_configured(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="tok", respond_to_bots=True),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 999

    raw = _make_discord_message(author_bot=True)
    msg = await iface._convert_inbound(raw)
    assert msg is not None

  @pytest.mark.asyncio
  async def test_guild_access_control_blocked(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="tok", allowed_guild_ids=[999]),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 0

    raw = _make_discord_message(guild_id=300)
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_guild_access_control_allowed(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="tok", allowed_guild_ids=[300]),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 0

    raw = _make_discord_message(guild_id=300)
    msg = await iface._convert_inbound(raw)
    assert msg is not None

  @pytest.mark.asyncio
  async def test_channel_access_control_blocked(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="tok", allowed_channel_ids=[999]),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 0

    raw = _make_discord_message(channel_id=200)
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_command_prefix_filter(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="tok", command_prefix="!ask"),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    iface._client.user.id = 0

    # Without prefix → skip
    raw = _make_discord_message(content="Hello")
    msg = await iface._convert_inbound(raw)
    assert msg is None

    # With prefix → text is stripped
    raw = _make_discord_message(content="!ask What is AI?")
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "What is AI?"

  @pytest.mark.asyncio
  async def test_image_attachment(self, interface):
    attachment = MagicMock()
    attachment.content_type = "image/png"
    attachment.url = "https://cdn.discord.com/image.png"
    attachment.filename = "image.png"
    attachment.size = 1024

    raw = _make_discord_message(attachments=[attachment])
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.images is not None
    assert len(msg.images) == 1
    assert msg.images[0].url == "https://cdn.discord.com/image.png"

  @pytest.mark.asyncio
  async def test_audio_attachment(self, interface):
    attachment = MagicMock()
    attachment.content_type = "audio/ogg"
    attachment.url = "https://cdn.discord.com/voice.ogg"
    attachment.filename = "voice.ogg"
    attachment.size = 2048

    raw = _make_discord_message(attachments=[attachment])
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.audio is not None
    assert len(msg.audio) == 1
    assert msg.audio[0].mime_type == "audio/ogg"

  @pytest.mark.asyncio
  async def test_file_attachment(self, interface):
    attachment = MagicMock()
    attachment.content_type = "application/pdf"
    attachment.url = "https://cdn.discord.com/doc.pdf"
    attachment.filename = "doc.pdf"
    attachment.size = 5000

    raw = _make_discord_message(attachments=[attachment])
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.files is not None
    assert len(msg.files) == 1
    assert msg.files[0].filename == "doc.pdf"

  @pytest.mark.asyncio
  async def test_reply_context(self, interface):
    ref = MagicMock()
    ref.message_id = 123
    raw = _make_discord_message(reference=ref)
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.reply_to_message_id == "123"


# --- Response sending tests ---


class TestDiscordSending:
  @pytest.fixture
  def interface(self, mock_agent):
    iface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="test_token"),
    )
    iface._client = MagicMock()
    iface._client.user = MagicMock()
    return iface

  @pytest.mark.asyncio
  async def test_send_text_reply(self, interface, sample_interface_message):
    from definable.interfaces.message import InterfaceResponse

    raw = _make_discord_message()
    raw.reply = AsyncMock()
    raw.channel.send = AsyncMock()
    raw.channel.typing = AsyncMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))

    msg = sample_interface_message(platform="discord")
    resp = InterfaceResponse(content="Hello!")
    await interface._send_response(msg, resp, raw)

    raw.reply.assert_called_once_with("Hello!")

  @pytest.mark.asyncio
  async def test_text_splitting_at_2000(self, interface):
    chunks = interface._split_text("a" * 2500, 2000)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2000
    assert len(chunks[1]) == 500

  @pytest.mark.asyncio
  async def test_long_text_sends_multiple(self, interface, sample_interface_message):
    from definable.interfaces.message import InterfaceResponse

    raw = _make_discord_message()
    raw.reply = AsyncMock()
    raw.channel.send = AsyncMock()
    raw.channel.typing = AsyncMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))

    msg = sample_interface_message(platform="discord")
    resp = InterfaceResponse(content="a" * 2500)
    await interface._send_response(msg, resp, raw)

    # First chunk via reply, second via channel.send
    raw.reply.assert_called_once()
    raw.channel.send.assert_called_once()


# --- Full pipeline test ---


class TestDiscordPipeline:
  @pytest.mark.asyncio
  async def test_message_to_response(self, mock_agent):
    interface = DiscordInterface(
      agent=mock_agent,
      config=DiscordConfig(bot_token="test_token"),
    )
    interface._client = MagicMock()
    interface._client.user = MagicMock()
    interface._client.user.id = 999

    # Set up running state
    interface._running = True
    import asyncio

    interface._request_semaphore = asyncio.Semaphore(10)

    raw = _make_discord_message(content="What is 2+2?")
    raw.reply = AsyncMock()
    raw.channel.send = AsyncMock()
    raw.channel.typing = AsyncMock(return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))

    await interface.handle_platform_message(raw)

    # Verify reply was called with the agent's response
    raw.reply.assert_called_once_with("Test response")
