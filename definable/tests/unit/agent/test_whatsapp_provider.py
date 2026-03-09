"""Tests for WhatsApp provider protocol and data types."""

import pytest

from definable.agent.interface.whatsapp.provider import (
  ConnectionStatus,
  InboundMessage,
  OutboundMessage,
  PollMessage,
  QRLoginResult,
  ReactionMessage,
  SendResult,
  WhatsAppProvider,
)


class TestDataTypes:
  def test_inbound_message_defaults(self):
    msg = InboundMessage(id="1", from_phone="+1555", from_jid="1555@s.whatsapp.net", chat_jid="1555@s.whatsapp.net")
    assert msg.body == ""
    assert msg.is_group is False
    assert msg.is_from_me is False
    assert msg.images is None
    assert msg.raw == {}

  def test_outbound_message_text_only(self):
    msg = OutboundMessage(to="1555@s.whatsapp.net", body="Hello")
    assert msg.image is None
    assert msg.audio is None

  def test_poll_message(self):
    poll = PollMessage(to="1555@s.whatsapp.net", question="Pick one", options=["A", "B"])
    assert len(poll.options) == 2
    assert poll.allows_multiple is False

  def test_reaction_message(self):
    r = ReactionMessage(chat_jid="1555@s.whatsapp.net", message_id="abc", emoji="👍")
    assert r.from_me is False
    assert r.participant is None

  def test_send_result_success(self):
    r = SendResult(success=True, message_id="msg123")
    assert r.error is None

  def test_send_result_failure(self):
    r = SendResult(success=False, error="timeout")
    assert r.message_id is None

  def test_connection_status_defaults(self):
    s = ConnectionStatus()
    assert s.connected is False
    assert s.linked is False
    assert s.self_phone is None

  def test_qr_login_result(self):
    r = QRLoginResult(qr_data="data:image/png;base64,abc", message="Scan QR")
    assert r.connected is False


class TestProviderABC:
  def test_cannot_instantiate(self):
    with pytest.raises(TypeError):
      WhatsAppProvider()  # type: ignore[abstract]

  def test_default_capabilities(self):
    """Verify that a minimal concrete provider has correct defaults."""

    class MinimalProvider(WhatsAppProvider):
      async def connect(self, on_message):
        pass

      async def disconnect(self):
        pass

      async def send_text(self, to, body):
        return SendResult(success=True)

      async def send_media(self, msg):
        return SendResult(success=False)

      async def send_poll(self, poll):
        return SendResult(success=False)

      async def send_reaction(self, reaction):
        return SendResult(success=False)

      async def send_composing(self, to):
        pass

      async def status(self):
        return ConnectionStatus()

    p = MinimalProvider()
    assert p.supports_polls is False
    assert p.supports_reactions is False
    assert p.supports_groups is False
    assert p.supports_media is False
    assert p.supports_qr_login is False
    assert p.provider_name == "unknown"

  @pytest.mark.asyncio
  async def test_default_qr_login(self):
    """Default QR login returns 'not supported'."""

    class MinimalProvider(WhatsAppProvider):
      async def connect(self, on_message):
        pass

      async def disconnect(self):
        pass

      async def send_text(self, to, body):
        return SendResult(success=True)

      async def send_media(self, msg):
        return SendResult(success=False)

      async def send_poll(self, poll):
        return SendResult(success=False)

      async def send_reaction(self, reaction):
        return SendResult(success=False)

      async def send_composing(self, to):
        pass

      async def status(self):
        return ConnectionStatus()

    p = MinimalProvider()
    result = await p.login_qr_start()
    assert result.connected is False
    assert "not supported" in result.message.lower()

    result = await p.login_qr_wait()
    assert result.connected is False

    assert await p.logout() is False
