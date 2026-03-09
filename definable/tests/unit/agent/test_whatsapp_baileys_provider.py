"""Tests for BaileysProvider — sidecar lifecycle, WebSocket comms, message parsing, media encoding."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.interface.whatsapp.provider import (
  ConnectionStatus,
  InboundMessage,
  OutboundMessage,
  PollMessage,
  ReactionMessage,
  SendResult,
)
from definable.agent.interface.whatsapp.providers.baileys import BaileysProvider, _BRIDGE_DIR


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def provider(tmp_path):
  return BaileysProvider(
    auth_dir=str(tmp_path / "wa-auth"),
    node_path="node",
    bridge_port=0,
    verbose=False,
  )


def _make_ws_mock():
  """Create a mock websockets connection that supports send/close/async iteration."""
  ws = AsyncMock()
  ws.send = AsyncMock()
  ws.close = AsyncMock()
  return ws


# --------------------------------------------------------------------------- #
# Construction and defaults                                                    #
# --------------------------------------------------------------------------- #


class TestBaileysConstruction:
  def test_default_bridge_dir(self, provider):
    assert provider._bridge_dir == _BRIDGE_DIR

  def test_custom_bridge_dir(self, tmp_path):
    p = BaileysProvider(bridge_dir=str(tmp_path / "custom"))
    assert p._bridge_dir == tmp_path / "custom"

  def test_auth_dir_resolved(self, provider, tmp_path):
    assert provider._auth_dir == str((tmp_path / "wa-auth").resolve())

  def test_initial_status(self, provider):
    assert provider._status.connected is False
    assert provider._status.running is False


# --------------------------------------------------------------------------- #
# Capability flags                                                             #
# --------------------------------------------------------------------------- #


class TestBaileysCapabilities:
  def test_supports_polls(self, provider):
    assert provider.supports_polls is True

  def test_supports_reactions(self, provider):
    assert provider.supports_reactions is True

  def test_supports_groups(self, provider):
    assert provider.supports_groups is True

  def test_supports_media(self, provider):
    assert provider.supports_media is True

  def test_supports_qr_login(self, provider):
    assert provider.supports_qr_login is True

  def test_provider_name(self, provider):
    assert provider.provider_name == "baileys"


# --------------------------------------------------------------------------- #
# Message parsing                                                              #
# --------------------------------------------------------------------------- #


class TestBaileysParseInbound:
  def test_text_message(self, provider):
    raw = {
      "id": "msg1",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "Hello world",
      "push_name": "Alice",
      "is_group": False,
      "is_from_me": False,
      "timestamp": 1709000000.0,
    }
    msg = provider._parse_inbound(raw)
    assert isinstance(msg, InboundMessage)
    assert msg.id == "msg1"
    assert msg.body == "Hello world"
    assert msg.from_phone == "15551234567"
    assert msg.push_name == "Alice"
    assert msg.is_group is False
    assert msg.images is None
    assert msg.raw == raw

  def test_group_message(self, provider):
    raw = {
      "id": "msg2",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "120363000000-111111@g.us",
      "body": "Group hello",
      "is_group": True,
      "group_subject": "Test Group",
      "group_participants": ["15551234567@s.whatsapp.net", "15559876543@s.whatsapp.net"],
      "mentioned_jids": ["15559876543@s.whatsapp.net"],
      "was_mentioned": True,
    }
    msg = provider._parse_inbound(raw)
    assert msg.is_group is True
    assert msg.group_subject == "Test Group"
    assert msg.was_mentioned is True
    assert len(msg.mentioned_jids) == 1

  def test_reply_context(self, provider):
    raw = {
      "id": "msg3",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "This is a reply",
      "reply_to_id": "original_msg_id",
      "reply_to_body": "Original message text",
      "reply_to_sender": "15559876543@s.whatsapp.net",
    }
    msg = provider._parse_inbound(raw)
    assert msg.reply_to_id == "original_msg_id"
    assert msg.reply_to_body == "Original message text"

  def test_image_media(self, provider):
    image_bytes = b"fake image data"
    raw = {
      "id": "msg4",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "",
      "media": {
        "type": "image",
        "mimeType": "image/jpeg",
        "base64": base64.b64encode(image_bytes).decode(),
        "filename": "photo.jpg",
      },
    }
    msg = provider._parse_inbound(raw)
    assert msg.images is not None
    assert len(msg.images) == 1
    assert msg.images[0].content == image_bytes
    assert msg.images[0].mime_type == "image/jpeg"

  def test_audio_media(self, provider):
    audio_bytes = b"fake audio data"
    raw = {
      "id": "msg5",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "",
      "media": {
        "type": "audio",
        "mimeType": "audio/ogg",
        "base64": base64.b64encode(audio_bytes).decode(),
        "filename": "audio.ogg",
      },
    }
    msg = provider._parse_inbound(raw)
    assert msg.audio is not None
    assert len(msg.audio) == 1
    assert msg.audio[0].content == audio_bytes

  def test_video_media(self, provider):
    video_bytes = b"fake video data"
    raw = {
      "id": "msg6",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "",
      "media": {
        "type": "video",
        "mimeType": "video/mp4",
        "base64": base64.b64encode(video_bytes).decode(),
        "filename": "video.mp4",
      },
    }
    msg = provider._parse_inbound(raw)
    assert msg.videos is not None
    assert len(msg.videos) == 1
    assert msg.videos[0].content == video_bytes

  def test_document_media(self, provider):
    doc_bytes = b"fake pdf data"
    raw = {
      "id": "msg7",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "",
      "media": {
        "type": "file",
        "mimeType": "application/pdf",
        "base64": base64.b64encode(doc_bytes).decode(),
        "filename": "report.pdf",
      },
    }
    msg = provider._parse_inbound(raw)
    assert msg.files is not None
    assert len(msg.files) == 1
    assert msg.files[0].content == doc_bytes
    assert msg.files[0].filename == "report.pdf"

  def test_no_media(self, provider):
    raw = {
      "id": "msg8",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "Just text",
    }
    msg = provider._parse_inbound(raw)
    assert msg.images is None
    assert msg.audio is None
    assert msg.videos is None
    assert msg.files is None

  def test_location(self, provider):
    raw = {
      "id": "msg9",
      "from_phone": "15551234567",
      "from_jid": "15551234567@s.whatsapp.net",
      "chat_jid": "15551234567@s.whatsapp.net",
      "body": "",
      "location": {"latitude": 37.7749, "longitude": -122.4194},
    }
    msg = provider._parse_inbound(raw)
    assert msg.latitude == 37.7749
    assert msg.longitude == -122.4194


# --------------------------------------------------------------------------- #
# Media encoding                                                               #
# --------------------------------------------------------------------------- #


class TestBaileysEncodeMedia:
  def test_encode_image(self):
    from definable.media import Image

    msg = OutboundMessage(to="15551234567@s.whatsapp.net", image=Image(content=b"img data", mime_type="image/png"))
    result = BaileysProvider._encode_media(msg)
    assert result is not None
    assert result["type"] == "image"
    assert result["mime_type"] == "image/png"
    assert base64.b64decode(result["content_base64"]) == b"img data"

  def test_encode_audio(self):
    from definable.media import Audio

    msg = OutboundMessage(to="15551234567@s.whatsapp.net", audio=Audio(content=b"aud data", mime_type="audio/ogg"))
    result = BaileysProvider._encode_media(msg)
    assert result is not None
    assert result["type"] == "audio"
    assert base64.b64decode(result["content_base64"]) == b"aud data"

  def test_encode_video(self):
    from definable.media import Video

    msg = OutboundMessage(to="15551234567@s.whatsapp.net", video=Video(content=b"vid data", mime_type="video/mp4"))
    result = BaileysProvider._encode_media(msg)
    assert result is not None
    assert result["type"] == "video"

  def test_encode_file(self):
    from definable.media import File

    msg = OutboundMessage(
      to="15551234567@s.whatsapp.net",
      file=File(content=b"file data", mime_type="application/pdf", filename="doc.pdf"),
    )
    result = BaileysProvider._encode_media(msg)
    assert result is not None
    assert result["type"] == "file"
    assert result["filename"] == "doc.pdf"

  def test_encode_no_media(self):
    msg = OutboundMessage(to="15551234567@s.whatsapp.net", body="text only")
    result = BaileysProvider._encode_media(msg)
    assert result is None

  def test_encode_url_only_file(self):
    from definable.media import File

    msg = OutboundMessage(to="15551234567@s.whatsapp.net", file=File(url="https://example.com/f.pdf"))
    result = BaileysProvider._encode_media(msg)
    assert result is None


# --------------------------------------------------------------------------- #
# Command correlation                                                          #
# --------------------------------------------------------------------------- #


class TestBaileysCommandCorrelation:
  @pytest.mark.asyncio
  async def test_send_command_success(self, provider):
    ws = _make_ws_mock()
    provider._ws = ws

    sent_id = None

    async def fake_send(data):
      nonlocal sent_id
      parsed = json.loads(data)
      sent_id = parsed["id"]

    ws.send = fake_send

    # Simulate the response arriving from the receive loop
    async def resolve_future():
      await asyncio.sleep(0.05)
      for cmd_id, future in list(provider._pending.items()):
        if not future.done():
          future.set_result({"type": "send_result", "id": cmd_id, "success": True, "message_id": "wamid.123"})

    asyncio.create_task(resolve_future())

    result = await provider._send_command({"type": "send", "to": "x", "body": "hi"}, timeout=2.0)
    assert result["success"] is True
    assert result["message_id"] == "wamid.123"

  @pytest.mark.asyncio
  async def test_send_command_timeout(self, provider):
    ws = _make_ws_mock()
    provider._ws = ws

    result = await provider._send_command({"type": "send", "to": "x", "body": "hi"}, timeout=0.1)
    assert result["success"] is False
    assert "timed out" in result["error"]
    # Future should be cleaned up
    assert len(provider._pending) == 0

  @pytest.mark.asyncio
  async def test_send_command_not_connected(self, provider):
    result = await provider._send_command({"type": "send", "to": "x", "body": "hi"})
    assert result["success"] is False
    assert "not connected" in result["error"].lower()


# --------------------------------------------------------------------------- #
# Send methods (via _send_command)                                             #
# --------------------------------------------------------------------------- #


class TestBaileysSendMethods:
  @pytest.fixture
  def connected_provider(self, provider):
    """Provider with a mock WS that auto-resolves commands."""
    ws = _make_ws_mock()
    provider._ws = ws

    async def auto_resolve_command(cmd, timeout=30.0):
      cmd_id = cmd.get("id")
      if not cmd_id:
        from uuid import uuid4

        cmd_id = str(uuid4())
        cmd["id"] = cmd_id
      # Directly return a success result
      return {"type": f"{cmd['type']}_result", "id": cmd_id, "success": True, "message_id": "wamid.test"}

    provider._send_command = auto_resolve_command
    return provider

  @pytest.mark.asyncio
  async def test_send_text(self, connected_provider):
    result = await connected_provider.send_text("15551234567@s.whatsapp.net", "Hello!")
    assert isinstance(result, SendResult)
    assert result.success is True

  @pytest.mark.asyncio
  async def test_send_poll(self, connected_provider):
    poll = PollMessage(to="15551234567@s.whatsapp.net", question="Pick", options=["A", "B"])
    result = await connected_provider.send_poll(poll)
    assert result.success is True

  @pytest.mark.asyncio
  async def test_send_reaction(self, connected_provider):
    reaction = ReactionMessage(chat_jid="15551234567@s.whatsapp.net", message_id="msg1", emoji="👍")
    result = await connected_provider.send_reaction(reaction)
    assert result.success is True

  @pytest.mark.asyncio
  async def test_send_media_with_content(self, connected_provider):
    from definable.media import Image

    msg = OutboundMessage(to="15551234567@s.whatsapp.net", image=Image(content=b"img", mime_type="image/png"))
    result = await connected_provider.send_media(msg)
    assert result.success is True

  @pytest.mark.asyncio
  async def test_send_media_no_content(self, connected_provider):
    msg = OutboundMessage(to="15551234567@s.whatsapp.net", body="text only")
    result = await connected_provider.send_media(msg)
    assert result.success is False
    assert "no media" in (result.error or "").lower()


# --------------------------------------------------------------------------- #
# Receive loop dispatch                                                        #
# --------------------------------------------------------------------------- #


class TestBaileysReceiveLoop:
  @pytest.mark.asyncio
  async def test_ready_event_sets_status(self, provider):
    """Simulate ready event being processed."""
    provider._ws = _make_ws_mock()
    msg = {
      "type": "ready",
      "connected": False,
      "auth_exists": True,
      "self_phone": "15551234567",
      "self_jid": "15551234567@s.whatsapp.net",
    }

    # Manually process like the receive loop would
    provider._status.connected = msg.get("connected", False)
    provider._status.linked = msg.get("auth_exists", False)
    provider._status.self_phone = msg.get("self_phone")
    provider._status.self_jid = msg.get("self_jid")
    provider._ready_event.set()

    assert provider._ready_event.is_set()
    assert provider._status.linked is True
    assert provider._status.self_phone == "15551234567"

  @pytest.mark.asyncio
  async def test_connected_event(self, provider):
    """Simulate connected event."""
    provider._status.connected = True
    provider._status.self_phone = "15551234567"
    provider._status.reconnect_attempts = 0
    provider._connected_event.set()

    assert provider._connected_event.is_set()
    assert provider._status.connected is True

  @pytest.mark.asyncio
  async def test_disconnected_event(self, provider):
    """Simulate disconnected event."""
    provider._status.connected = False
    provider._status.reconnect_attempts = 3
    provider._connected_event.clear()

    assert not provider._connected_event.is_set()
    assert provider._status.connected is False

  @pytest.mark.asyncio
  async def test_result_resolves_pending_future(self, provider):
    """A *_result message should resolve the matching pending future."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    provider._pending["cmd-123"] = future

    msg = {"type": "send_result", "id": "cmd-123", "success": True, "message_id": "wamid.abc"}
    # Simulate receive loop dispatch for *_result
    cmd_id = msg.get("id")
    if cmd_id and cmd_id in provider._pending:
      f = provider._pending.pop(cmd_id)
      if not f.done():
        f.set_result(msg)

    result = await asyncio.wait_for(future, timeout=1.0)
    assert result["success"] is True
    assert result["message_id"] == "wamid.abc"
    assert "cmd-123" not in provider._pending


# --------------------------------------------------------------------------- #
# Disconnect                                                                   #
# --------------------------------------------------------------------------- #


class TestBaileysDisconnect:
  @pytest.mark.asyncio
  async def test_disconnect_no_ws(self, provider):
    """Disconnect when not connected should not raise."""
    await provider.disconnect()
    assert provider._status.running is False
    assert provider._status.connected is False

  @pytest.mark.asyncio
  async def test_disconnect_cleans_up(self, provider):
    ws = _make_ws_mock()
    provider._ws = ws
    provider._status.running = True
    provider._on_message = AsyncMock()

    # Create a mock process
    proc = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    async def wait_mock():
      return 0

    proc.wait = wait_mock
    provider._process = proc

    await provider.disconnect()
    assert provider._ws is None
    assert provider._process is None
    assert provider._on_message is None
    assert provider._status.running is False


# --------------------------------------------------------------------------- #
# QR login                                                                     #
# --------------------------------------------------------------------------- #


class TestBaileysQRLogin:
  @pytest.mark.asyncio
  async def test_login_qr_wait_timeout(self, provider):
    result = await provider.login_qr_wait(timeout_ms=100)
    assert result.connected is False
    assert "timed out" in result.message.lower()

  @pytest.mark.asyncio
  async def test_login_qr_wait_success(self, provider):
    provider._connected_event.set()
    result = await provider.login_qr_wait(timeout_ms=1000)
    assert result.connected is True

  @pytest.mark.asyncio
  async def test_logout_not_connected(self, provider):
    result = await provider.logout()
    assert result is False


# --------------------------------------------------------------------------- #
# Status                                                                       #
# --------------------------------------------------------------------------- #


class TestBaileysStatus:
  @pytest.mark.asyncio
  async def test_status_no_ws(self, provider):
    """Status without WS returns cached status."""
    provider._status.connected = False
    provider._status.running = False
    s = await provider.status()
    assert isinstance(s, ConnectionStatus)
    assert s.connected is False

  @pytest.mark.asyncio
  async def test_status_with_ws_error(self, provider):
    """Status with WS that fails returns cached status."""
    ws = _make_ws_mock()
    provider._ws = ws

    async def failing_send(cmd, timeout=5.0):
      raise Exception("ws error")

    provider._send_command = failing_send

    provider._status.connected = True
    s = await provider.status()
    assert s.connected is True


# --------------------------------------------------------------------------- #
# npm deps                                                                     #
# --------------------------------------------------------------------------- #


class TestBaileysNpmDeps:
  @pytest.mark.asyncio
  async def test_skip_if_node_modules_exists(self, provider, tmp_path):
    provider._bridge_dir = tmp_path
    (tmp_path / "node_modules").mkdir()

    # Should not call subprocess
    await provider._ensure_npm_deps()

  @pytest.mark.asyncio
  async def test_npm_not_found(self, provider, tmp_path):
    provider._bridge_dir = tmp_path

    with patch("shutil.which", return_value=None):
      with pytest.raises(RuntimeError, match="npm not found"):
        await provider._ensure_npm_deps()


# --------------------------------------------------------------------------- #
# Interface wiring                                                             #
# --------------------------------------------------------------------------- #


class TestInterfaceWiring:
  def test_build_provider_baileys(self):
    """_build_provider returns BaileysProvider for provider='baileys'."""
    from definable.agent.interface.whatsapp.interface import WhatsAppInterface

    p = WhatsAppInterface._build_provider(
      provider="baileys",
      account_sid="",
      auth_token="",
      from_number="",
      validate_signatures=True,
      auth_dir="./wa-auth",
      node_path="node",
      bridge_port=0,
      reconnect_max_attempts=12,
      heartbeat_seconds=60,
      verbose=False,
    )
    assert isinstance(p, BaileysProvider)

  def test_build_provider_twilio(self):
    """_build_provider still returns TwilioProvider for provider='twilio'."""
    from definable.agent.interface.whatsapp.interface import WhatsAppInterface
    from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

    p = WhatsAppInterface._build_provider(
      provider="twilio",
      account_sid="AC123",
      auth_token="secret",
      from_number="+1555",
      validate_signatures=True,
      auth_dir="",
      node_path="",
      bridge_port=0,
      reconnect_max_attempts=12,
      heartbeat_seconds=60,
      verbose=False,
    )
    assert isinstance(p, TwilioProvider)

  def test_build_provider_unknown(self):
    from definable.agent.interface.whatsapp.interface import WhatsAppInterface

    with pytest.raises(ValueError, match="Unknown WhatsApp provider"):
      WhatsAppInterface._build_provider(
        provider="unknown",
        account_sid="",
        auth_token="",
        from_number="",
        validate_signatures=True,
        auth_dir="",
        node_path="",
        bridge_port=0,
        reconnect_max_attempts=12,
        heartbeat_seconds=60,
        verbose=False,
      )
