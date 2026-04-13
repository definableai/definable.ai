"""Tests for WhatsAppInterface — config, lifecycle, message handling."""

import pytest

from definable.agent.interface.whatsapp.config import WhatsAppConfig
from definable.agent.interface.whatsapp.interface import WhatsAppInterface


class TestWhatsAppConfig:
  def test_defaults(self):
    config = WhatsAppConfig()
    assert config.platform == "whatsapp"
    assert config.account_sid == ""
    assert config.auth_token == ""
    assert config.from_number == ""
    assert config.webhook_path == "/whatsapp/webhook"
    assert config.status_callback_path == "/whatsapp/status"
    assert config.validate_signatures is True
    assert config.max_message_length == 1600

  def test_custom(self):
    config = WhatsAppConfig(
      account_sid="AC123",
      auth_token="token",
      from_number="whatsapp:+14155238886",
    )
    assert config.account_sid == "AC123"
    assert config.auth_token == "token"
    assert config.from_number == "whatsapp:+14155238886"


class TestWhatsAppInterface:
  def test_init_defaults(self):
    iface = WhatsAppInterface()
    assert iface.config.platform == "whatsapp"
    assert iface._wa_config.webhook_path == "/whatsapp/webhook"
    assert iface.needs_server() is True

  def test_init_custom(self):
    iface = WhatsAppInterface(
      account_sid="AC123",
      auth_token="secret",
      from_number="whatsapp:+1234567890",
      webhook_path="/wa/hook",
    )
    assert iface._wa_config.account_sid == "AC123"
    assert iface._wa_config.from_number == "whatsapp:+1234567890"
    assert iface._wa_config.webhook_path == "/wa/hook"

  def test_create_router(self):
    iface = WhatsAppInterface()
    router = iface.create_router()
    assert hasattr(router, "routes")
    # Should have 2 routes: webhook + status callback
    route_paths = [r.path for r in router.routes]
    assert "/whatsapp/webhook" in route_paths
    assert "/whatsapp/status" in route_paths

  @pytest.mark.asyncio
  async def test_convert_inbound_valid(self):
    iface = WhatsAppInterface()
    raw = {
      "Body": "Hello agent",
      "From": "whatsapp:+15551234567",
      "To": "whatsapp:+14155238886",
      "MessageSid": "SM123",
      "NumMedia": "0",
    }
    msg = await iface._convert_inbound(raw)
    assert msg is not None
    assert msg.text == "Hello agent"
    assert msg.platform == "whatsapp"
    assert msg.platform_user_id == "15551234567"  # normalized E.164 (bare digits)

  @pytest.mark.asyncio
  async def test_convert_inbound_empty_body(self):
    iface = WhatsAppInterface()
    raw = {"Body": "", "From": "whatsapp:+1234"}
    msg = await iface._convert_inbound(raw)
    assert msg is None

  @pytest.mark.asyncio
  async def test_convert_inbound_no_from(self):
    iface = WhatsAppInterface()
    raw = {"Body": "Hello", "From": ""}
    msg = await iface._convert_inbound(raw)
    assert msg is None


class TestWhatsAppMessageSplitting:
  def test_short_message(self):
    chunks = WhatsAppInterface._split_message("Hello", 1600)
    assert chunks == ["Hello"]

  def test_exact_length(self):
    text = "x" * 1600
    chunks = WhatsAppInterface._split_message(text, 1600)
    assert len(chunks) == 1

  def test_long_message_split(self):
    text = " ".join(["word"] * 500)  # ~2500 chars
    chunks = WhatsAppInterface._split_message(text, 1600)
    assert len(chunks) > 1
    assert all(len(c) <= 1600 for c in chunks)
    # Reconstruct and verify no data loss
    reconstructed = " ".join(chunks)
    assert reconstructed.replace("  ", " ") == text.strip()

  def test_no_space_split(self):
    text = "a" * 3200
    chunks = WhatsAppInterface._split_message(text, 1600)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 1600
    assert chunks[1] == "a" * 1600


class TestWhatsAppExports:
  def test_import_from_whatsapp_package(self):
    from definable.agent.interface.whatsapp import WhatsAppConfig, WhatsAppInterface

    assert WhatsAppConfig is not None
    assert WhatsAppInterface is not None

  def test_import_from_interface_package(self):
    from definable.agent.interface import WhatsAppConfig, WhatsAppInterface

    assert WhatsAppConfig is not None
    assert WhatsAppInterface is not None
