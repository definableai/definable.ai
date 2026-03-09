"""Tests for TwilioProvider — webhook handling, send logic, retry."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.interface.whatsapp.provider import ConnectionStatus
from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider


@pytest.fixture
def provider():
  return TwilioProvider(
    account_sid="AC123",
    auth_token="secret",
    from_number="whatsapp:+14155238886",
  )


# --------------------------------------------------------------------------- #
# Lifecycle                                                                    #
# --------------------------------------------------------------------------- #


class TestTwilioLifecycle:
  @pytest.mark.asyncio
  async def test_connect_disconnect(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)
    assert provider._connected is True
    assert provider._http_client is not None

    await provider.disconnect()
    assert provider._connected is False
    assert provider._http_client is None

  @pytest.mark.asyncio
  async def test_disconnect_idempotent(self, provider):
    await provider.disconnect()  # not connected — should not raise
    assert provider._connected is False

  @pytest.mark.asyncio
  async def test_status(self, provider):
    s = await provider.status()
    assert isinstance(s, ConnectionStatus)
    assert s.connected is False
    assert s.linked is True  # has credentials


# --------------------------------------------------------------------------- #
# Webhook handling                                                             #
# --------------------------------------------------------------------------- #


class TestTwilioWebhook:
  @pytest.mark.asyncio
  async def test_handle_webhook_valid(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)

    form = {
      "Body": "Hello agent",
      "From": "whatsapp:+15551234567",
      "To": "whatsapp:+14155238886",
      "MessageSid": "SM123",
      "NumMedia": "0",
    }
    msg = await provider.handle_webhook(form)
    assert msg is not None
    assert msg.body == "Hello agent"
    assert msg.from_phone == "15551234567"
    assert msg.is_group is False

  @pytest.mark.asyncio
  async def test_handle_webhook_empty_body(self, provider):
    msg = await provider.handle_webhook({"Body": "", "From": "whatsapp:+1555"})
    assert msg is None

  @pytest.mark.asyncio
  async def test_handle_webhook_no_from(self, provider):
    msg = await provider.handle_webhook({"Body": "Hello", "From": ""})
    assert msg is None

  @pytest.mark.asyncio
  async def test_handle_webhook_fires_callback(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)

    form = {"Body": "Test", "From": "whatsapp:+15551234567", "MessageSid": "SM1"}
    await provider.handle_webhook(form)

    # Give the fire-and-forget task a moment
    import asyncio

    await asyncio.sleep(0.05)
    callback.assert_called_once()


# --------------------------------------------------------------------------- #
# Capabilities                                                                 #
# --------------------------------------------------------------------------- #


class TestTwilioCapabilities:
  def test_supports_media(self, provider):
    assert provider.supports_media is True

  def test_no_polls(self, provider):
    assert provider.supports_polls is False

  def test_no_reactions(self, provider):
    assert provider.supports_reactions is False

  def test_no_groups(self, provider):
    assert provider.supports_groups is False

  def test_no_qr_login(self, provider):
    assert provider.supports_qr_login is False

  def test_provider_name(self, provider):
    assert provider.provider_name == "twilio"

  @pytest.mark.asyncio
  async def test_send_poll_returns_error(self, provider):
    from definable.agent.interface.whatsapp.provider import PollMessage

    result = await provider.send_poll(PollMessage(to="x", question="q", options=["a"]))
    assert result.success is False
    assert "not support" in (result.error or "").lower()

  @pytest.mark.asyncio
  async def test_send_reaction_returns_error(self, provider):
    from definable.agent.interface.whatsapp.provider import ReactionMessage

    result = await provider.send_reaction(ReactionMessage(chat_jid="x", message_id="1", emoji="👍"))
    assert result.success is False


# --------------------------------------------------------------------------- #
# Send with mock HTTP                                                          #
# --------------------------------------------------------------------------- #


class TestTwilioSend:
  @pytest.mark.asyncio
  async def test_send_text_success(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"sid": "SM999"}
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    result = await provider.send_text("+15551234567", "Hello!")
    assert result.success is True
    assert result.message_id == "SM999"
    assert provider._send_count == 1

    await provider.disconnect()

  @pytest.mark.asyncio
  async def test_send_text_4xx_no_retry(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    result = await provider.send_text("+15551234567", "Hello!")
    assert result.success is False
    assert provider._error_count == 1
    # Only called once (no retry for 4xx)
    assert provider._http_client.post.call_count == 1

    await provider.disconnect()

  @pytest.mark.asyncio
  async def test_send_text_not_connected(self, provider):
    result = await provider.send_text("+15551234567", "Hello!")
    assert result.success is False
    assert "not initialized" in (result.error or "").lower()

  @pytest.mark.asyncio
  async def test_send_media_with_url(self, provider):
    from definable.agent.interface.whatsapp.provider import OutboundMessage
    from definable.media import Image

    callback = AsyncMock()
    await provider.connect(callback)

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"sid": "SM888"}
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    msg = OutboundMessage(to="+1555", body="Look", image=Image(url="https://example.com/img.png"))
    result = await provider.send_media(msg)
    assert result.success is True

    # Verify MediaUrl was in the POST data
    call_kwargs = provider._http_client.post.call_args
    assert "MediaUrl" in call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))

    await provider.disconnect()

  @pytest.mark.asyncio
  async def test_send_media_no_url(self, provider):
    from definable.agent.interface.whatsapp.provider import OutboundMessage
    from definable.media import Image

    callback = AsyncMock()
    await provider.connect(callback)

    msg = OutboundMessage(to="+1555", body="Look", image=Image(content=b"raw bytes"))
    result = await provider.send_media(msg)
    assert result.success is False
    assert "url" in (result.error or "").lower()

    await provider.disconnect()

  @pytest.mark.asyncio
  async def test_send_adds_whatsapp_prefix(self, provider):
    callback = AsyncMock()
    await provider.connect(callback)

    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"sid": "SM777"}
    provider._http_client.post = AsyncMock(return_value=mock_resp)

    await provider.send_text("+15551234567", "Hello!")

    call_kwargs = provider._http_client.post.call_args
    data = call_kwargs.kwargs.get("data", call_kwargs[1].get("data", {}))
    assert data["To"] == "whatsapp:+15551234567"

    await provider.disconnect()
