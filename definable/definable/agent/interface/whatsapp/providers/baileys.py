"""Baileys WhatsApp Web provider — Node.js sidecar bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from definable.agent.interface.whatsapp.normalize import redact_phone
from definable.agent.interface.whatsapp.provider import (
  ConnectionStatus,
  InboundMessage,
  MessageCallback,
  OutboundMessage,
  PollMessage,
  QRLoginResult,
  ReactionMessage,
  SendResult,
  WhatsAppProvider,
)
from definable.media import Audio, File, Image, Video
from definable.utils.log import log_debug, log_error, log_info, log_warning

# Bridge location: sibling _bridge/ dir inside the whatsapp interface package
# baileys.py → providers/ → whatsapp/_bridge/
_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "_bridge"

_WS_CONNECT_TIMEOUT = 15.0
_WS_CONNECT_RETRY_DELAY = 0.3
_WS_CONNECT_MAX_RETRIES = 50
_COMMAND_TIMEOUT = 30.0
_NPM_INSTALL_TIMEOUT = 120


class BaileysProvider(WhatsAppProvider):
  """Baileys (WhatsApp Web) provider via Node.js sidecar.

  Self-hosted. Free. Full protocol access.
  Supports: text, media, polls, reactions, groups, QR login.
  Requires: Node.js >= 18 on the host.

  Args:
    auth_dir: Directory for WhatsApp credential storage.
    node_path: Path to the ``node`` binary.
    bridge_dir: Path to the bridge JS directory. Defaults to the
      bundled bridge in ``definable/bridge/whatsapp/``.
    bridge_port: WebSocket port. ``0`` = auto-assign (recommended).
    verbose: Enable verbose logging in both Python and Node.js.
    reconnect_max_attempts: Max reconnect attempts before giving up.
    heartbeat_seconds: Heartbeat interval in seconds.
  """

  def __init__(
    self,
    *,
    auth_dir: str = "./whatsapp-auth",
    node_path: str = "node",
    bridge_dir: Optional[str] = None,
    bridge_port: int = 0,
    verbose: bool = False,
    reconnect_max_attempts: int = 12,
    heartbeat_seconds: int = 60,
  ) -> None:
    self._auth_dir = str(Path(auth_dir).resolve())
    self._node_path = node_path
    self._bridge_dir = Path(bridge_dir) if bridge_dir else _BRIDGE_DIR
    self._bridge_port = bridge_port
    self._verbose = verbose
    self._reconnect_max_attempts = reconnect_max_attempts
    self._heartbeat_seconds = heartbeat_seconds

    self._on_message: Optional[MessageCallback] = None
    self._process: Optional[asyncio.subprocess.Process] = None
    self._ws: Optional[Any] = None  # websockets connection
    self._receive_task: Optional[asyncio.Task[None]] = None
    self._pending: Dict[str, asyncio.Future[dict]] = {}
    self._status = ConnectionStatus()
    self._ready_event = asyncio.Event()
    self._last_qr: Optional[dict] = None
    self._connected_event = asyncio.Event()

  # --- Provider protocol ---

  async def connect(self, on_message: MessageCallback) -> None:
    self._on_message = on_message

    # Ensure node is available
    node_path = shutil.which(self._node_path) or self._node_path
    if not shutil.which(node_path):
      raise RuntimeError(f"Node.js not found at '{self._node_path}'. Install Node.js >= 18 or set node_path to the correct binary.")

    # Ensure npm dependencies are installed
    await self._ensure_npm_deps()

    # Ensure auth dir exists
    Path(self._auth_dir).mkdir(parents=True, exist_ok=True)

    # Spawn sidecar process
    cmd = [
      node_path,
      str(self._bridge_dir / "index.js"),
      f"--port={self._bridge_port}",
      f"--auth-dir={self._auth_dir}",
      f"--heartbeat={self._heartbeat_seconds}",
    ]
    if self._verbose:
      cmd.append("--verbose")

    log_info(f"[whatsapp:baileys] Starting sidecar: {' '.join(cmd[:3])}...")
    self._process = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=None,  # inherit so QR codes and errors reach the terminal
    )

    # Read port from stdout
    actual_port = await self._read_port()

    # Connect WebSocket
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
      raise ImportError("websockets is required for BaileysProvider. Install: pip install websockets")

    self._ws = await self._connect_ws(actual_port)

    # Start receive loop
    self._receive_task = asyncio.create_task(self._receive_loop())

    # Wait for ready event
    try:
      await asyncio.wait_for(self._ready_event.wait(), timeout=_WS_CONNECT_TIMEOUT)
    except asyncio.TimeoutError:
      raise RuntimeError("Sidecar did not become ready within timeout") from None

    self._status.running = True
    log_info(f"[whatsapp:baileys] Connected (port={actual_port}, auth={self._auth_dir})")

  async def disconnect(self) -> None:
    self._status.running = False

    if self._ws:
      with contextlib.suppress(Exception):
        await self._ws.send(json.dumps({"type": "shutdown"}))
        await asyncio.sleep(0.5)
      with contextlib.suppress(Exception):
        await self._ws.close()
      self._ws = None

    if self._receive_task and not self._receive_task.done():
      self._receive_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._receive_task
      self._receive_task = None

    if self._process:
      try:
        self._process.terminate()
        await asyncio.wait_for(self._process.wait(), timeout=5.0)
      except (asyncio.TimeoutError, ProcessLookupError):
        with contextlib.suppress(ProcessLookupError):
          self._process.kill()
      self._process = None

    self._on_message = None
    self._status.connected = False
    log_info("[whatsapp:baileys] Disconnected")

  async def send_text(self, to: str, body: str) -> SendResult:
    result = await self._send_command({
      "type": "send",
      "to": to,
      "body": body,
    })
    return SendResult(
      success=result.get("success", False),
      message_id=result.get("message_id"),
      error=result.get("error"),
    )

  async def send_media(self, msg: OutboundMessage) -> SendResult:
    media_payload = self._encode_media(msg)
    if media_payload is None:
      return SendResult(success=False, error="No media content available")

    result = await self._send_command({
      "type": "send",
      "to": msg.to,
      "body": msg.body,
      "media": media_payload,
      "reply_to_id": msg.reply_to_id,
    })
    return SendResult(
      success=result.get("success", False),
      message_id=result.get("message_id"),
      error=result.get("error"),
    )

  async def send_poll(self, poll: PollMessage) -> SendResult:
    result = await self._send_command({
      "type": "send_poll",
      "to": poll.to,
      "question": poll.question,
      "options": poll.options,
      "allows_multiple": poll.allows_multiple,
    })
    return SendResult(
      success=result.get("success", False),
      message_id=result.get("message_id"),
      error=result.get("error"),
    )

  async def send_reaction(self, reaction: ReactionMessage) -> SendResult:
    result = await self._send_command({
      "type": "send_reaction",
      "chat_jid": reaction.chat_jid,
      "message_id": reaction.message_id,
      "emoji": reaction.emoji,
      "from_me": reaction.from_me,
      "participant": reaction.participant,
    })
    return SendResult(
      success=result.get("success", False),
      error=result.get("error"),
    )

  async def send_composing(self, to: str) -> None:
    if self._ws:
      with contextlib.suppress(Exception):
        await self._ws.send(json.dumps({"type": "send_composing", "to": to}))

  async def status(self) -> ConnectionStatus:
    # Try to get fresh status from sidecar
    if self._ws:
      try:
        result = await self._send_command({"type": "get_status"}, timeout=5.0)
        return ConnectionStatus(
          connected=result.get("connected", False),
          running=result.get("running", False),
          reconnect_attempts=result.get("reconnect_attempts", 0),
          last_connected_at=result.get("last_connected_at"),
          last_message_at=result.get("last_message_at"),
          last_error=result.get("last_error"),
          linked=result.get("linked", False),
          self_phone=result.get("self_phone"),
          self_jid=result.get("self_jid"),
        )
      except Exception:
        pass
    return self._status

  async def login_qr_start(self, force: bool = False) -> QRLoginResult:
    result = await self._send_command({
      "type": "login_qr_start",
      "force": force,
    })
    if self._last_qr:
      return QRLoginResult(
        qr_data=self._last_qr.get("data"),
        message=result.get("message", "QR generated"),
      )
    return QRLoginResult(message=result.get("message", "QR login initiated"))

  async def login_qr_wait(self, timeout_ms: int = 60_000) -> QRLoginResult:
    try:
      await asyncio.wait_for(self._connected_event.wait(), timeout=timeout_ms / 1000)
      return QRLoginResult(connected=True, message="WhatsApp connected successfully")
    except asyncio.TimeoutError:
      return QRLoginResult(connected=False, message="QR scan timed out")

  async def logout(self) -> bool:
    try:
      result = await self._send_command({"type": "logout"})
      return result.get("success", False)
    except Exception:
      return False

  # --- Capability flags ---

  @property
  def supports_polls(self) -> bool:
    return True

  @property
  def supports_reactions(self) -> bool:
    return True

  @property
  def supports_groups(self) -> bool:
    return True

  @property
  def supports_media(self) -> bool:
    return True

  @property
  def supports_qr_login(self) -> bool:
    return True

  @property
  def provider_name(self) -> str:
    return "baileys"

  # --- Internal: WebSocket communication ---

  async def _read_port(self) -> int:
    """Read the actual port from the sidecar's stdout."""
    assert self._process and self._process.stdout
    deadline = time.monotonic() + _WS_CONNECT_TIMEOUT
    while time.monotonic() < deadline:
      line_bytes = await asyncio.wait_for(self._process.stdout.readline(), timeout=_WS_CONNECT_TIMEOUT)
      line = line_bytes.decode().strip()
      if line.startswith("PORT:"):
        return int(line.split(":")[1])
    raise RuntimeError("Sidecar did not report its port")

  async def _connect_ws(self, port: int) -> Any:
    """Connect to the sidecar WebSocket with retries."""
    import websockets

    for attempt in range(_WS_CONNECT_MAX_RETRIES):
      try:
        ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        return ws
      except (ConnectionRefusedError, OSError):
        if attempt < _WS_CONNECT_MAX_RETRIES - 1:
          await asyncio.sleep(_WS_CONNECT_RETRY_DELAY)
        else:
          raise RuntimeError(f"Could not connect to sidecar WebSocket on port {port}") from None

  async def _receive_loop(self) -> None:
    """Read WebSocket messages and dispatch to handlers."""
    assert self._ws is not None
    try:
      async for raw in self._ws:
        try:
          msg = json.loads(raw)
        except json.JSONDecodeError:
          continue

        msg_type = msg.get("type", "")

        if msg_type == "message":
          inbound = self._parse_inbound(msg)
          if self._on_message:
            asyncio.create_task(self._on_message(inbound))

        elif msg_type == "ready":
          self._status.connected = msg.get("connected", False)
          self._status.linked = msg.get("auth_exists", False)
          self._status.self_phone = msg.get("self_phone")
          self._status.self_jid = msg.get("self_jid")
          self._ready_event.set()

        elif msg_type == "connected":
          self._status.connected = True
          self._status.self_phone = msg.get("self_phone")
          self._status.self_jid = msg.get("self_jid")
          self._status.last_connected_at = time.time()
          self._status.reconnect_attempts = 0
          self._connected_event.set()
          log_info(f"[whatsapp:baileys] Connected as {redact_phone(msg.get('self_phone', ''))}")

        elif msg_type == "disconnected":
          self._status.connected = False
          self._status.last_disconnect_at = time.time()
          self._status.reconnect_attempts = msg.get("attempt", 0)
          self._connected_event.clear()
          if msg.get("reconnecting"):
            log_debug(f"[whatsapp:baileys] Disconnected, reconnecting (attempt {msg.get('attempt', 0)})")
          else:
            log_warning(f"[whatsapp:baileys] Disconnected: {msg.get('reason', 'unknown')}")

        elif msg_type == "qr":
          self._last_qr = msg
          log_info("[whatsapp:baileys] QR code generated — scan with WhatsApp on your phone")

        elif msg_type == "logged_out":
          self._status.connected = False
          self._status.linked = False
          log_warning(f"[whatsapp:baileys] {msg.get('message', 'Logged out')}")

        elif msg_type == "status":
          self._status.last_message_at = msg.get("last_message_at")
          self._status.last_error = msg.get("last_error")

        elif msg_type == "error":
          self._status.last_error = msg.get("message")
          log_error(f"[whatsapp:baileys] Bridge error: {msg.get('message', 'Unknown')}")

        elif msg_type.endswith("_result"):
          cmd_id = msg.get("id")
          if cmd_id and cmd_id in self._pending:
            future = self._pending.pop(cmd_id)
            if not future.done():
              future.set_result(msg)

    except asyncio.CancelledError:
      pass
    except Exception as e:
      if not self._status.running:
        return
      log_error(f"[whatsapp:baileys] Receive loop error: {e}")

  async def _send_command(self, cmd: dict, timeout: float = _COMMAND_TIMEOUT) -> dict:
    """Send a command and wait for the correlated response."""
    if not self._ws:
      return {"success": False, "error": "Not connected to sidecar"}

    cmd_id = str(uuid4())
    cmd["id"] = cmd_id

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict] = loop.create_future()
    self._pending[cmd_id] = future

    try:
      await self._ws.send(json.dumps(cmd))
      return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
      self._pending.pop(cmd_id, None)
      return {"success": False, "error": f"Command {cmd['type']} timed out after {timeout}s"}
    except Exception as e:
      self._pending.pop(cmd_id, None)
      return {"success": False, "error": str(e)}

  # --- Internal: setup ---

  async def _ensure_npm_deps(self) -> None:
    """Run npm install in the bridge directory if node_modules is missing."""
    node_modules = self._bridge_dir / "node_modules"
    if node_modules.exists():
      return

    log_info("[whatsapp:baileys] Installing bridge dependencies (first time only)...")
    npm_path = shutil.which("npm")
    if not npm_path:
      raise RuntimeError("npm not found. Install Node.js >= 18 (includes npm).")

    proc = await asyncio.create_subprocess_exec(
      npm_path,
      "install",
      "--production",
      cwd=str(self._bridge_dir),
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_NPM_INSTALL_TIMEOUT)
    if proc.returncode != 0:
      raise RuntimeError(f"npm install failed (exit {proc.returncode}): {stderr.decode()[:500]}")
    log_info("[whatsapp:baileys] Bridge dependencies installed")

  # --- Internal: message parsing ---

  def _parse_inbound(self, msg: dict) -> InboundMessage:
    """Parse a bridge 'message' event into an InboundMessage."""
    media = msg.get("media")
    images = None
    audio_list = None
    videos = None
    files = None

    if media:
      import base64

      content_bytes = base64.b64decode(media.get("base64", "")) if media.get("base64") else None
      media_type = media.get("type", "file")

      if media_type == "image" and content_bytes:
        images = [Image(content=content_bytes, mime_type=media.get("mimeType"), format=media.get("filename", "").split(".")[-1] or None)]
      elif media_type == "audio" and content_bytes:
        audio_list = [Audio(content=content_bytes, mime_type=media.get("mimeType"))]
      elif media_type == "video" and content_bytes:
        videos = [Video(content=content_bytes, mime_type=media.get("mimeType"))]
      elif content_bytes:
        files = [File(content=content_bytes, mime_type=media.get("mimeType"), filename=media.get("filename"))]

    location = msg.get("location")

    return InboundMessage(
      id=msg.get("id", ""),
      from_phone=msg.get("from_phone", ""),
      from_jid=msg.get("from_jid", ""),
      chat_jid=msg.get("chat_jid", ""),
      body=msg.get("body", ""),
      push_name=msg.get("push_name", ""),
      is_group=msg.get("is_group", False),
      is_from_me=msg.get("is_from_me", False),
      timestamp=msg.get("timestamp", 0.0),
      reply_to_id=msg.get("reply_to_id"),
      reply_to_body=msg.get("reply_to_body"),
      reply_to_sender=msg.get("reply_to_sender"),
      group_subject=msg.get("group_subject"),
      group_participants=msg.get("group_participants"),
      mentioned_jids=msg.get("mentioned_jids"),
      was_mentioned=msg.get("was_mentioned", False),
      images=images,
      audio=audio_list,
      videos=videos,
      files=files,
      latitude=location.get("latitude") if location else None,
      longitude=location.get("longitude") if location else None,
      raw=msg,
    )

  @staticmethod
  def _encode_media(msg: OutboundMessage) -> Optional[dict]:
    """Encode an OutboundMessage's media for the bridge wire format."""
    import base64

    if msg.image:
      content = msg.image.get_content_bytes()
      if content:
        return {
          "type": "image",
          "mime_type": msg.image.mime_type or "image/jpeg",
          "content_base64": base64.b64encode(content).decode(),
          "filename": str(msg.image.filepath or "image.jpg").split("/")[-1],
        }
    if msg.audio:
      content = msg.audio.get_content_bytes()
      if content:
        return {
          "type": "audio",
          "mime_type": msg.audio.mime_type or "audio/ogg",
          "content_base64": base64.b64encode(content).decode(),
        }
    if msg.video:
      content = msg.video.get_content_bytes()
      if content:
        return {
          "type": "video",
          "mime_type": msg.video.mime_type or "video/mp4",
          "content_base64": base64.b64encode(content).decode(),
          "filename": str(msg.video.filepath or "video.mp4").split("/")[-1],
        }
    if msg.file:
      content_bytes: Optional[bytes] = None
      if msg.file.content and isinstance(msg.file.content, bytes):
        content_bytes = msg.file.content
      elif msg.file.url:
        return None  # Can't encode URL-only files
      elif msg.file.filepath:
        with open(msg.file.filepath, "rb") as f:
          content_bytes = f.read()
      if content_bytes:
        return {
          "type": "file",
          "mime_type": msg.file.mime_type or "application/octet-stream",
          "content_base64": base64.b64encode(content_bytes).decode(),
          "filename": msg.file.filename or "file",
        }
    return None
