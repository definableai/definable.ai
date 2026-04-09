"""Tests for EmailInterface — config, lifecycle, message handling."""

import email.mime.text
import pytest

from definable.agent.interface.email.config import EmailConfig
from definable.agent.interface.email.interface import EmailInterface


class TestEmailConfig:
  def test_defaults(self):
    config = EmailConfig()
    assert config.platform == "email"
    assert config.imap_host == ""
    assert config.imap_port == 993
    assert config.smtp_host == ""
    assert config.smtp_port == 587
    assert config.email_address == ""
    assert config.imap_folder == "INBOX"
    assert config.poll_interval == 30.0
    assert config.mark_as_read is True
    assert config.subject_prefix == "Re: "
    assert config.reply_quote_original is True

  def test_custom(self):
    config = EmailConfig(
      imap_host="imap.gmail.com",
      smtp_host="smtp.gmail.com",
      email_address="agent@example.com",
      email_password="secret",
      poll_interval=60.0,
    )
    assert config.imap_host == "imap.gmail.com"
    assert config.smtp_host == "smtp.gmail.com"
    assert config.email_address == "agent@example.com"
    assert config.poll_interval == 60.0


class TestEmailInterface:
  def test_init_defaults(self):
    iface = EmailInterface()
    assert iface.config.platform == "email"
    assert iface._email_config.imap_host == ""
    assert iface._poll_task is None

  def test_init_custom(self):
    iface = EmailInterface(
      imap_host="imap.test.com",
      smtp_host="smtp.test.com",
      email_address="bot@test.com",
      email_password="pass",
    )
    assert iface._email_config.imap_host == "imap.test.com"
    assert iface._email_config.smtp_host == "smtp.test.com"
    assert iface._email_config.email_address == "bot@test.com"

  def test_deprecated_config_param(self):
    config = EmailConfig(imap_host="imap.test.com")
    with pytest.warns(DeprecationWarning):
      iface = EmailInterface(config=config)
    assert iface._email_config.imap_host == "imap.test.com"

  @pytest.mark.asyncio
  async def test_start_without_imap_host_raises(self):
    iface = EmailInterface()
    from definable.agent.testing import create_test_agent

    iface.bind(create_test_agent())
    with pytest.raises(ValueError, match="imap_host"):
      await iface.start()


class TestEmailBodyExtraction:
  def test_extract_plain_text(self):
    msg = email.mime.text.MIMEText("Hello world", "plain")
    body = EmailInterface._extract_body(msg)
    assert body == "Hello world"

  def test_extract_multipart(self):
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart("alternative")
    msg.attach(email.mime.text.MIMEText("Plain version", "plain"))
    msg.attach(email.mime.text.MIMEText("<b>HTML version</b>", "html"))
    body = EmailInterface._extract_body(msg)
    assert body == "Plain version"


class TestEmailConvertInbound:
  @pytest.mark.asyncio
  async def test_convert_valid(self):
    iface = EmailInterface()
    msg = email.mime.text.MIMEText("Hello agent", "plain")
    msg["From"] = "user@example.com"
    msg["Subject"] = "Test subject"
    msg["Message-ID"] = "<msg123@example.com>"

    raw = {"uid": "1", "email_message": msg}
    result = await iface._convert_inbound(raw)
    assert result is not None
    assert result.text == "Hello agent"
    assert result.platform == "email"
    assert result.platform_user_id == "user@example.com"
    assert result.metadata["subject"] == "Test subject"
    assert result.metadata["message_id"] == "<msg123@example.com>"

  @pytest.mark.asyncio
  async def test_convert_no_sender(self):
    iface = EmailInterface()
    msg = email.mime.text.MIMEText("Hello", "plain")
    # No From header
    raw = {"uid": "1", "email_message": msg}
    result = await iface._convert_inbound(raw)
    assert result is None

  @pytest.mark.asyncio
  async def test_convert_empty_body(self):
    iface = EmailInterface()
    msg = email.mime.text.MIMEText("", "plain")
    msg["From"] = "user@example.com"
    raw = {"uid": "1", "email_message": msg}
    result = await iface._convert_inbound(raw)
    assert result is None


class TestEmailExports:
  def test_import_from_email_package(self):
    from definable.agent.interface.email import EmailConfig, EmailInterface

    assert EmailConfig is not None
    assert EmailInterface is not None

  def test_import_from_interface_package(self):
    from definable.agent.interface import EmailConfig, EmailInterface

    assert EmailConfig is not None
    assert EmailInterface is not None


class TestServerInterfaceDetection:
  def test_websocket_needs_server(self):
    from definable.agent.interface.websocket import WebSocketInterface

    iface = WebSocketInterface()
    assert iface.needs_server() is True

  def test_whatsapp_needs_server(self):
    from definable.agent.interface.whatsapp import WhatsAppInterface

    iface = WhatsAppInterface()
    assert iface.needs_server() is True
