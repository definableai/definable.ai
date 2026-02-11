"""Tests for the Signal interface (mocked — no real signal-cli-rest-api)."""

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.interfaces.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.interfaces.signal.config import SignalConfig
from definable.interfaces.signal.interface import SignalInterface


# --- Config tests ---


class TestSignalConfig:
  def test_valid_config(self):
    config = SignalConfig(phone_number="+1234567890")
    assert config.platform == "signal"
    assert config.phone_number == "+1234567890"
    assert config.api_base_url == "http://localhost:8080"
    assert config.max_message_length == 65536
    assert config.trust_all_keys is True

  def test_missing_phone_raises(self):
    with pytest.raises(InterfaceError, match="phone_number is required"):
      SignalConfig(phone_number="")

  def test_defaults(self):
    config = SignalConfig(phone_number="+1")
    assert config.polling_interval == 1.0
    assert config.connect_timeout == 10.0
    assert config.request_timeout == 60.0
    assert config.allowed_phone_numbers is None
    assert config.allowed_group_ids is None

  def test_access_control_fields(self):
    config = SignalConfig(
      phone_number="+1",
      allowed_phone_numbers=["+111", "+222"],
      allowed_group_ids=["g1", "g2"],
    )
    assert config.allowed_phone_numbers == ["+111", "+222"]
    assert config.allowed_group_ids == ["g1", "g2"]

  def test_custom_api_url(self):
    config = SignalConfig(
      phone_number="+1",
      api_base_url="http://signal-api:9090",
    )
    assert config.api_base_url == "http://signal-api:9090"


# --- Helper to build Signal envelope ---


def _make_signal_envelope(
  source: str = "+15551234567",
  source_name: str = "Alice",
  timestamp: int = 1700000000,
  message: str = "Hello",
  group_id: Optional[str] = None,
  attachments: Optional[list] = None,
  quote: Optional[dict] = None,
) -> dict:
  """Build a mock signal-cli-rest-api envelope."""
  data_message: dict[str, Any] = {"message": message}
  if attachments:
    data_message["attachments"] = attachments
  if group_id:
    data_message["groupInfo"] = {"groupId": group_id}
  if quote:
    data_message["quote"] = quote

  return {
    "envelope": {
      "source": source,
      "sourceNumber": source,
      "sourceName": source_name,
      "timestamp": timestamp,
      "dataMessage": data_message,
    }
  }


# --- Inbound conversion tests ---


class TestSignalInbound:
  @pytest.fixture
  def interface(self, mock_agent):
    return SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )

  @pytest.mark.asyncio
  async def test_convert_text_message(self, interface):
    raw = _make_signal_envelope(
      source="+15551234567",
      source_name="Alice",
      message="Hello bot",
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Hello bot"
    assert msg.platform == "signal"
    assert msg.platform_user_id == "+15551234567"
    assert msg.platform_chat_id == "+15551234567"  # DM: chat_id == source
    assert msg.username == "Alice"

  @pytest.mark.asyncio
  async def test_convert_group_message(self, interface):
    raw = _make_signal_envelope(
      source="+15551234567",
      message="Group msg",
      group_id="abc123",
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.platform_chat_id == "abc123"

  @pytest.mark.asyncio
  async def test_access_control_phone_blocked(self, mock_agent):
    iface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(
        phone_number="+10000000000",
        allowed_phone_numbers=["+19999999999"],
      ),
    )
    raw = _make_signal_envelope(source="+15551234567")
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_access_control_phone_allowed(self, mock_agent):
    iface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(
        phone_number="+10000000000",
        allowed_phone_numbers=["+15551234567"],
      ),
    )
    raw = _make_signal_envelope(source="+15551234567")
    msg = await iface._convert_inbound(raw)
    assert msg is not None

  @pytest.mark.asyncio
  async def test_access_control_group_blocked(self, mock_agent):
    iface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(
        phone_number="+10000000000",
        allowed_group_ids=["allowed_group"],
      ),
    )
    raw = _make_signal_envelope(group_id="blocked_group")
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_access_control_group_allowed(self, mock_agent):
    iface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(
        phone_number="+10000000000",
        allowed_group_ids=["my_group"],
      ),
    )
    raw = _make_signal_envelope(group_id="my_group")
    msg = await iface._convert_inbound(raw)
    assert msg is not None

  @pytest.mark.asyncio
  async def test_image_attachment(self, interface):
    raw = _make_signal_envelope(
      attachments=[
        {"id": "att1", "contentType": "image/jpeg"},
      ],
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.images is not None
    assert len(msg.images) == 1
    assert "att1" in msg.images[0].url

  @pytest.mark.asyncio
  async def test_audio_attachment(self, interface):
    raw = _make_signal_envelope(
      attachments=[
        {"id": "att2", "contentType": "audio/ogg"},
      ],
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.audio is not None
    assert len(msg.audio) == 1
    assert msg.audio[0].mime_type == "audio/ogg"

  @pytest.mark.asyncio
  async def test_file_attachment(self, interface):
    raw = _make_signal_envelope(
      attachments=[
        {"id": "att3", "contentType": "application/pdf", "filename": "doc.pdf", "size": 5000},
      ],
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.files is not None
    assert len(msg.files) == 1
    assert msg.files[0].filename == "doc.pdf"

  @pytest.mark.asyncio
  async def test_reply_context(self, interface):
    raw = _make_signal_envelope(
      quote={"id": 1699999999, "author": "+15551234567", "text": "original"},
    )
    msg = await interface._convert_inbound(raw)
    assert msg is not None
    assert msg.reply_to_message_id == "1699999999"


# --- Response sending tests ---


class TestSignalSending:
  @pytest.fixture
  def interface(self, mock_agent):
    return SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )

  @pytest.mark.asyncio
  async def test_send_text_dm(self, interface, sample_interface_message):
    interface._api_call = AsyncMock(return_value={})

    from definable.interfaces.message import InterfaceResponse

    msg = sample_interface_message(platform="signal")
    resp = InterfaceResponse(content="Hello!")
    raw = _make_signal_envelope(source="+15551234567")

    await interface._send_response(msg, resp, raw)

    interface._api_call.assert_called_once_with(
      "POST",
      "/v2/send",
      json={
        "message": "Hello!",
        "number": "+10000000000",
        "text_mode": "normal",
        "recipients": ["+15551234567"],
      },
    )

  @pytest.mark.asyncio
  async def test_send_text_group(self, interface, sample_interface_message):
    interface._api_call = AsyncMock(return_value={})

    from definable.interfaces.message import InterfaceResponse

    msg = sample_interface_message(platform="signal")
    resp = InterfaceResponse(content="Group reply")
    raw = _make_signal_envelope(source="+15551234567", group_id="grp1")

    await interface._send_response(msg, resp, raw)

    call_args = interface._api_call.call_args
    assert call_args[1]["json"]["recipients"] == ["grp1"]


# --- API error handling tests ---


class TestSignalApiErrors:
  @pytest.fixture
  def interface(self, mock_agent):
    iface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )
    return iface

  @pytest.mark.asyncio
  async def test_401_raises_auth_error(self, interface):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceAuthenticationError, match="Authentication failed"):
      await interface._api_call("GET", "/v1/about")

  @pytest.mark.asyncio
  async def test_429_raises_rate_limit(self, interface):
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Too Many Requests"

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceRateLimitError):
      await interface._api_call("POST", "/v2/send", json={"message": "hi"})

  @pytest.mark.asyncio
  async def test_400_raises_message_error(self, interface):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request: missing number"

    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    interface._client = mock_client

    with pytest.raises(InterfaceMessageError, match="Bad request"):
      await interface._api_call("POST", "/v2/send", json={})

  @pytest.mark.asyncio
  async def test_connection_error(self, interface):
    import httpx

    mock_client = AsyncMock()
    mock_client.request.side_effect = httpx.ConnectError("Connection refused")
    interface._client = mock_client

    with pytest.raises(InterfaceConnectionError, match="Failed to connect"):
      await interface._api_call("GET", "/v1/about")

  @pytest.mark.asyncio
  async def test_timeout_error(self, interface):
    import httpx

    mock_client = AsyncMock()
    mock_client.request.side_effect = httpx.ReadTimeout("Read timed out")
    interface._client = mock_client

    with pytest.raises(InterfaceConnectionError, match="timed out"):
      await interface._api_call("GET", "/v1/receive/+1")


# --- Poll loop test ---


class TestSignalPolling:
  @pytest.mark.asyncio
  async def test_receive_messages(self, mock_agent):
    interface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )
    envelopes = [
      _make_signal_envelope(source="+1111", message="msg1"),
      _make_signal_envelope(source="+2222", message="msg2"),
    ]
    interface._api_call = AsyncMock(return_value=envelopes)

    result = await interface._receive_messages()
    assert len(result) == 2

  @pytest.mark.asyncio
  async def test_process_envelope_skips_non_data(self, mock_agent):
    interface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )
    interface.handle_platform_message = AsyncMock()

    # Envelope without dataMessage (e.g. a receipt)
    envelope = {"envelope": {"source": "+1111", "timestamp": 123}}
    await interface._process_envelope(envelope)
    interface.handle_platform_message.assert_not_called()


# --- Full pipeline test ---


class TestSignalPipeline:
  @pytest.mark.asyncio
  async def test_message_to_response(self, mock_agent):
    interface = SignalInterface(
      agent=mock_agent,
      config=SignalConfig(phone_number="+10000000000"),
    )
    interface._api_call = AsyncMock(return_value={})

    # Set up running state
    interface._running = True
    import asyncio

    interface._request_semaphore = asyncio.Semaphore(10)

    raw = _make_signal_envelope(
      source="+15551234567",
      message="What is 2+2?",
    )

    await interface.handle_platform_message(raw)

    # Verify _api_call was called with POST /v2/send
    send_calls = [c for c in interface._api_call.call_args_list if c[0] == ("POST", "/v2/send")]
    assert len(send_calls) >= 1
    payload = send_calls[0][1]["json"]
    assert "Test response" in payload["message"]
    assert payload["recipients"] == ["+15551234567"]
