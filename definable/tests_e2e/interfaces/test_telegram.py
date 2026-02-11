"""Tests for the Telegram interface (mocked — no real API calls)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.interfaces.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.interfaces.telegram.config import TelegramConfig
from definable.interfaces.telegram.interface import TelegramInterface


# --- Config tests ---


class TestTelegramConfig:
  def test_valid_config(self):
    config = TelegramConfig(bot_token="123:ABC")
    assert config.platform == "telegram"
    assert config.bot_token == "123:ABC"
    assert config.mode == "polling"
    assert config.max_message_length == 4096

  def test_missing_token_raises(self):
    with pytest.raises(InterfaceError, match="bot_token is required"):
      TelegramConfig(bot_token="")

  def test_webhook_requires_url(self):
    with pytest.raises(InterfaceError, match="webhook_url is required"):
      TelegramConfig(bot_token="123:ABC", mode="webhook")

  def test_webhook_with_url(self):
    config = TelegramConfig(
      bot_token="123:ABC",
      mode="webhook",
      webhook_url="https://example.com",
    )
    assert config.mode == "webhook"
    assert config.webhook_url == "https://example.com"

  def test_defaults(self):
    config = TelegramConfig(bot_token="123:ABC")
    assert config.parse_mode == "HTML"
    assert config.polling_interval == 0.5
    assert config.polling_timeout == 30
    assert config.connect_timeout == 10.0
    assert config.request_timeout == 60.0
    assert config.allowed_user_ids is None
    assert config.allowed_chat_ids is None

  def test_access_control_fields(self):
    config = TelegramConfig(
      bot_token="123:ABC",
      allowed_user_ids=[111, 222],
      allowed_chat_ids=[333],
    )
    assert config.allowed_user_ids == [111, 222]
    assert config.allowed_chat_ids == [333]


# --- Inbound conversion tests ---


class TestTelegramInbound:
  @pytest.fixture
  def interface(self, mock_agent):
    return TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token"),
    )

  @pytest.mark.asyncio
  async def test_convert_text_message(self, interface):
    raw = {
      "message_id": 42,
      "from": {"id": 12345, "username": "testuser"},
      "chat": {"id": 67890},
      "text": "Hello bot",
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Hello bot"
    assert msg.platform == "telegram"
    assert msg.platform_user_id == "12345"
    assert msg.platform_chat_id == "67890"
    assert msg.platform_message_id == "42"
    assert msg.username == "testuser"

  @pytest.mark.asyncio
  async def test_convert_caption_message(self, interface):
    raw = {
      "message_id": 43,
      "from": {"id": 12345, "first_name": "Test"},
      "chat": {"id": 67890},
      "caption": "Photo caption",
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Photo caption"
    assert msg.username == "Test"

  @pytest.mark.asyncio
  async def test_access_control_user_blocked(self, mock_agent):
    iface = TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token", allowed_user_ids=[999]),
    )
    raw = {
      "message_id": 1,
      "from": {"id": 12345},
      "chat": {"id": 67890},
      "text": "blocked",
    }
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_access_control_user_allowed(self, mock_agent):
    iface = TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token", allowed_user_ids=[12345]),
    )
    raw = {
      "message_id": 1,
      "from": {"id": 12345},
      "chat": {"id": 67890},
      "text": "allowed",
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None

  @pytest.mark.asyncio
  async def test_access_control_chat_blocked(self, mock_agent):
    iface = TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token", allowed_chat_ids=[999]),
    )
    raw = {
      "message_id": 1,
      "from": {"id": 12345},
      "chat": {"id": 67890},
      "text": "blocked",
    }
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_photo_extraction(self, interface):
    """Verify largest photo is picked and file URL resolved."""
    interface._get_file_url = AsyncMock(return_value="https://tg.example/photo.jpg")
    raw = {
      "message_id": 1,
      "from": {"id": 1},
      "chat": {"id": 1},
      "photo": [
        {"file_id": "small", "file_size": 100},
        {"file_id": "large", "file_size": 1000},
      ],
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.images is not None
    assert len(msg.images) == 1
    assert msg.images[0].url == "https://tg.example/photo.jpg"
    interface._get_file_url.assert_called_once_with("large")

  @pytest.mark.asyncio
  async def test_voice_extraction(self, interface):
    interface._get_file_url = AsyncMock(return_value="https://tg.example/voice.ogg")
    raw = {
      "message_id": 1,
      "from": {"id": 1},
      "chat": {"id": 1},
      "voice": {"file_id": "v1", "mime_type": "audio/ogg", "duration": 5},
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.audio is not None
    assert len(msg.audio) == 1
    assert msg.audio[0].mime_type == "audio/ogg"

  @pytest.mark.asyncio
  async def test_document_extraction(self, interface):
    interface._get_file_url = AsyncMock(return_value="https://tg.example/doc.pdf")
    raw = {
      "message_id": 1,
      "from": {"id": 1},
      "chat": {"id": 1},
      "document": {
        "file_id": "d1",
        "mime_type": "application/pdf",
        "file_name": "report.pdf",
        "file_size": 5000,
      },
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.files is not None
    assert len(msg.files) == 1
    assert msg.files[0].filename == "report.pdf"

  @pytest.mark.asyncio
  async def test_reply_context(self, interface):
    raw = {
      "message_id": 10,
      "from": {"id": 1},
      "chat": {"id": 1},
      "text": "reply",
      "reply_to_message": {"message_id": 5},
    }
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.reply_to_message_id == "5"


# --- Response sending tests ---


class TestTelegramSending:
  @pytest.fixture
  def interface(self, mock_agent):
    return TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token"),
    )

  @pytest.mark.asyncio
  async def test_send_text(self, interface, sample_interface_message):
    interface._api_call = AsyncMock(return_value={})
    from definable.interfaces.message import InterfaceResponse

    msg = sample_interface_message(platform="telegram", chat_id="123")
    resp = InterfaceResponse(content="Hello!")
    raw = {"chat": {"id": 123}}

    await interface._send_response(msg, resp, raw)
    # sendChatAction + sendMessage
    assert interface._api_call.call_count == 2
    send_call = interface._api_call.call_args_list[1]
    assert send_call[0][0] == "sendMessage"
    assert send_call[0][1]["text"] == "Hello!"
    assert send_call[0][1]["chat_id"] == "123"

  @pytest.mark.asyncio
  async def test_text_splitting(self, interface):
    chunks = interface._split_text("a" * 5000, 4096)
    assert len(chunks) == 2
    assert len(chunks[0]) == 4096
    assert len(chunks[1]) == 904

  @pytest.mark.asyncio
  async def test_text_splitting_at_newline(self, interface):
    text = "a" * 4000 + "\n" + "b" * 200
    chunks = interface._split_text(text, 4096)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 4000

  @pytest.mark.asyncio
  async def test_send_with_parse_mode_fallback(self, interface, sample_interface_message):
    """If sendMessage with parse_mode fails, retry without it."""
    call_count = {"n": 0}

    async def mock_api_call(method, data=None):
      if method == "sendChatAction":
        return {}
      call_count["n"] += 1
      if call_count["n"] == 1:
        raise InterfaceMessageError("Bad request: can't parse HTML", platform="telegram")
      return {}

    interface._api_call = mock_api_call
    from definable.interfaces.message import InterfaceResponse

    msg = sample_interface_message(platform="telegram", chat_id="123")
    resp = InterfaceResponse(content="<bad>html")
    raw = {"chat": {"id": 123}}

    await interface._send_response(msg, resp, raw)
    assert call_count["n"] == 2  # First fails, second succeeds without parse_mode


# --- API error handling tests ---


class TestTelegramApiErrors:
  @pytest.fixture
  def interface(self, mock_agent):
    iface = TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token"),
    )
    return iface

  @pytest.mark.asyncio
  async def test_401_raises_auth_error(self, interface):
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": False, "error_code": 401, "description": "Unauthorized"}
    mock_response.status_code = 401

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceAuthenticationError, match="Invalid bot token"):
      await interface._api_call("getMe")

  @pytest.mark.asyncio
  async def test_429_raises_rate_limit_error(self, interface):
    mock_response = MagicMock()
    mock_response.json.return_value = {
      "ok": False,
      "error_code": 429,
      "description": "Too Many Requests",
      "parameters": {"retry_after": 10},
    }
    mock_response.status_code = 429

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceRateLimitError) as exc_info:
      await interface._api_call("sendMessage", {"chat_id": "1", "text": "hi"})
    assert exc_info.value.retry_after == 10.0

  @pytest.mark.asyncio
  async def test_400_raises_message_error(self, interface):
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": False, "error_code": 400, "description": "Bad Request"}
    mock_response.status_code = 400

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceMessageError, match="Bad request"):
      await interface._api_call("sendMessage", {"chat_id": "1", "text": ""})

  @pytest.mark.asyncio
  async def test_connection_error(self, interface):
    import httpx

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    interface._client = mock_client

    with pytest.raises(InterfaceConnectionError, match="Failed to connect"):
      await interface._api_call("getMe")

  @pytest.mark.asyncio
  async def test_timeout_error(self, interface):
    import httpx

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ReadTimeout("Read timed out")
    interface._client = mock_client

    with pytest.raises(InterfaceConnectionError, match="timed out"):
      await interface._api_call("getMe")


# --- Full pipeline test ---


class TestTelegramPipeline:
  @pytest.mark.asyncio
  async def test_message_to_response(self, mock_agent, sample_interface_message):
    """Verify full pipeline: raw message → agent → response sent."""
    interface = TelegramInterface(
      agent=mock_agent,
      config=TelegramConfig(bot_token="test:token"),
    )
    interface._api_call = AsyncMock(return_value={})

    # Simulate starting (sets up semaphore)
    interface._running = True
    import asyncio

    interface._request_semaphore = asyncio.Semaphore(10)

    raw = {
      "message_id": 1,
      "from": {"id": 100, "username": "user"},
      "chat": {"id": 200},
      "text": "What is 2+2?",
    }

    await interface.handle_platform_message(raw)

    # Verify sendMessage was called with agent's response
    send_calls = [c for c in interface._api_call.call_args_list if c[0][0] == "sendMessage"]
    assert len(send_calls) >= 1
    assert "Test response" in send_calls[0][0][1]["text"]
