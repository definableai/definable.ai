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

# WebSocket frame limit — Baileys auth state can grow to 10-15MB+
_WS_MAX_SIZE = 50 * 1024 * 1024  # 50 MB

# Heartbeat monitoring — if we don't hear from the bridge in this many
# multiples of the heartbeat interval, consider the connection dead.
_HEARTBEAT_MISS_THRESHOLD = 3

# Send retry defaults
_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE_DELAY = 1.0  # seconds


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
    send_max_retries: Max retries for transient send failures.
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
    send_max_retries: int = _SEND_MAX_RETRIES,
  ) -> None:
    self._auth_dir = str(Path(auth_dir).resolve())
    self._node_path = node_path
    self._bridge_dir = Path(bridge_dir) if bridge_dir else _BRIDGE_DIR
    self._bridge_port = bridge_port
    self._verbose = verbose
    self._reconnect_max_attempts = reconnect_max_attempts
    self._heartbeat_seconds = heartbeat_seconds
    self._send_max_retries = send_max_retries

    self._on_message: Optional[MessageCallback] = None
    self._process: Optional[asyncio.subprocess.Process] = None
    self._ws: Optional[Any] = None  # websockets connection
    self._receive_task: Optional[asyncio.Task[None]] = None
    self._heartbeat_task: Optional[asyncio.Task[None]] = None
    self._pending: Dict[str, asyncio.Future[dict]] = {}
    self._status = ConnectionStatus()
    self._ready_event = asyncio.Event()
    self._last_qr: Optional[dict] = None
    self._connected_event = asyncio.Event()
    self._shutting_down = False
    self._actual_port: Optional[int] = None

    # Heartbeat monitoring
    self._last_heartbeat_at: float = 0.0
    self._ws_reconnect_attempts = 0
    self._ws_max_reconnect_attempts = 5

  # --- Provider protocol ---

  async def connect(self, on_message: MessageCallback) -> None:
    self._on_message = on_message
    self._shutting_down = False

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
    self._actual_port = await self._read_port()

    # Connect WebSocket
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
      raise ImportError("websockets is required for BaileysProvider. Install: pip install websockets")

    self._ws = await self._connect_ws(self._actual_port)

    # Start receive loop + heartbeat monitor
    self._receive_task = asyncio.create_task(self._receive_loop())
    self._receive_task.add_done_callback(self._on_receive_loop_done)
    self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    # Wait for ready event
    try:
      await asyncio.wait_for(self._ready_event.wait(), timeout=_WS_CONNECT_TIMEOUT)
    except asyncio.TimeoutError:
      raise RuntimeError("Sidecar did not become ready within timeout") from None

    self._status.running = True
    self._last_heartbeat_at = time.monotonic()
    log_info(f"[whatsapp:baileys] Connected (port={self._actual_port}, auth={self._auth_dir})")

  async def disconnect(self) -> None:
    self._shutting_down = True
    self._status.running = False

    # Cancel heartbeat monitor
    if self._heartbeat_task and not self._heartbeat_task.done():
      self._heartbeat_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._heartbeat_task
      self._heartbeat_task = None

    # Send graceful shutdown command to sidecar
    if self._ws:
      with contextlib.suppress(Exception):
        await self._ws.send(json.dumps({"type": "shutdown"}))
        await asyncio.sleep(0.5)
      with contextlib.suppress(Exception):
        await self._ws.close()
      self._ws = None

    # Cancel receive loop
    if self._receive_task and not self._receive_task.done():
      self._receive_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._receive_task
      self._receive_task = None

    # Reject all pending futures
    self._reject_all_pending("Provider disconnecting")

    # Terminate sidecar process with proper reaping
    await self._kill_process()

    self._on_message = None
    self._status.connected = False
    self._actual_port = None
    log_info("[whatsapp:baileys] Disconnected")

  async def send_text(self, to: str, body: str) -> SendResult:
    return await self._send_with_retry({
      "type": "send",
      "to": to,
      "body": body,
    })

  async def send_media(self, msg: OutboundMessage) -> SendResult:
    media_payload = self._encode_media(msg)
    if media_payload is None:
      return SendResult(success=False, error="No media content available")

    return await self._send_with_retry({
      "type": "send",
      "to": msg.to,
      "body": msg.body,
      "media": media_payload,
      "reply_to_id": msg.reply_to_id,
    })

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

  # --- Contact validation & management ---

  async def check_on_whatsapp(self, phones: list[str]) -> list[dict]:
    """Check if phone numbers are registered on WhatsApp.

    Args:
      phones: List of E.164 phone numbers (e.g. ["+919810464995"]).

    Returns:
      List of dicts with ``jid`` and ``exists`` for each number.
      Empty list on failure.
    """
    result = await self._send_command({
      "type": "check_on_whatsapp",
      "phones": phones,
    })
    if result.get("success"):
      return result.get("results", [])
    log_warning(f"[whatsapp:baileys] check_on_whatsapp failed: {result.get('error')}")
    return []

  async def save_contact(self, jid: str, name: str) -> bool:
    """Save a contact to the WhatsApp address book.

    Args:
      jid: WhatsApp JID (e.g. ``"919810464995@s.whatsapp.net"``).
      name: Display name for the contact.

    Returns:
      True if saved successfully.
    """
    result = await self._send_command({
      "type": "save_contact",
      "jid": jid,
      "name": name,
    })
    if result.get("success"):
      log_debug(f"[whatsapp:baileys] Saved contact: {redact_phone(jid)} as {name!r}")
      return True
    log_warning(f"[whatsapp:baileys] save_contact failed: {result.get('error')}")
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

  # --- Internal: send with retry ---

  async def _send_with_retry(self, cmd: dict) -> SendResult:
    """Send a command with retry on transient failures."""
    last_error = ""
    for attempt in range(self._send_max_retries):
      result = await self._send_command(cmd)
      success = result.get("success", False)
      error = result.get("error", "")

      if success:
        return SendResult(
          success=True,
          message_id=result.get("message_id"),
        )

      last_error = error

      # Don't retry on permanent failures
      if self._is_permanent_error(error):
        break

      # Retry with backoff
      if attempt < self._send_max_retries - 1:
        delay = _SEND_RETRY_BASE_DELAY * (2**attempt)
        log_debug(f"[whatsapp:baileys] Send failed ({error}), retrying in {delay:.1f}s (attempt {attempt + 1}/{self._send_max_retries})")
        await asyncio.sleep(delay)

    return SendResult(success=False, error=last_error)

  @staticmethod
  def _is_permanent_error(error: str) -> bool:
    """Check if an error is permanent (no point retrying)."""
    permanent_patterns = [
      "not connected",
      "not on whatsapp",
      "invalid jid",
      "blocked",
      "not found",
      "invalid",
    ]
    error_lower = error.lower()
    return any(p in error_lower for p in permanent_patterns)

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
        ws = await websockets.connect(
          f"ws://127.0.0.1:{port}",
          max_size=_WS_MAX_SIZE,
          ping_interval=20,
          ping_timeout=20,
          close_timeout=5,
        )
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
          log_warning("[whatsapp:baileys] Received non-JSON WebSocket frame, skipping")
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
          # Heartbeat from the bridge — update monitoring timestamp
          self._last_heartbeat_at = time.monotonic()
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
      if self._shutting_down:
        return
      log_error(f"[whatsapp:baileys] Receive loop error: {e}")

  def _on_receive_loop_done(self, task: asyncio.Task[None]) -> None:
    """Callback when the receive loop exits — clean up pending futures and trigger reconnect."""
    # Reject all pending command futures so callers don't hang
    self._reject_all_pending("WebSocket connection lost")

    if self._shutting_down:
      return

    # Mark as disconnected
    self._status.connected = False
    self._connected_event.clear()

    # Schedule a reconnect attempt
    log_warning("[whatsapp:baileys] Receive loop exited, scheduling WebSocket reconnect...")
    try:
      loop = asyncio.get_running_loop()
      loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._reconnect_ws()))
    except RuntimeError:
      pass

  async def _reconnect_ws(self) -> None:
    """Attempt to reconnect the WebSocket to the still-running sidecar."""
    if self._shutting_down or self._actual_port is None:
      return

    # Check if sidecar process is still alive
    if self._process is None or self._process.returncode is not None:
      log_warning("[whatsapp:baileys] Sidecar process is dead, cannot reconnect WebSocket")
      self._status.running = False
      return

    for attempt in range(self._ws_max_reconnect_attempts):
      # Re-check after awaits — state may have changed
      shutdown: bool = self._shutting_down
      if shutdown:
        return

      delay = min(2.0 * (1.5**attempt), 30.0)
      log_info(f"[whatsapp:baileys] WebSocket reconnect attempt {attempt + 1}/{self._ws_max_reconnect_attempts} in {delay:.1f}s")
      await asyncio.sleep(delay)

      try:
        # Close stale WebSocket if still lingering
        if self._ws:
          with contextlib.suppress(Exception):
            await self._ws.close()
          self._ws = None

        self._ws = await self._connect_ws(self._actual_port)
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._receive_task.add_done_callback(self._on_receive_loop_done)
        self._last_heartbeat_at = time.monotonic()
        self._ws_reconnect_attempts = 0
        log_info("[whatsapp:baileys] WebSocket reconnected successfully")

        # Wait for fresh ready/connected status
        try:
          await asyncio.wait_for(self._ready_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
          log_warning("[whatsapp:baileys] Sidecar did not re-send ready after reconnect")
        return
      except Exception as e:
        log_warning(f"[whatsapp:baileys] WebSocket reconnect failed: {e}")

    log_error(f"[whatsapp:baileys] WebSocket reconnect exhausted ({self._ws_max_reconnect_attempts} attempts)")
    self._status.running = False

  async def _heartbeat_monitor(self) -> None:
    """Monitor heartbeat messages from the bridge — detect stale connections."""
    interval = self._heartbeat_seconds * _HEARTBEAT_MISS_THRESHOLD
    try:
      while not self._shutting_down:
        await asyncio.sleep(float(self._heartbeat_seconds))

        # Re-check after await — state may have changed
        shutdown: bool = self._shutting_down
        if shutdown:
          break

        # Skip monitoring if we're not supposed to be connected
        if not self._status.running:
          continue

        elapsed = time.monotonic() - self._last_heartbeat_at
        if elapsed > interval:
          log_warning(f"[whatsapp:baileys] No heartbeat for {elapsed:.0f}s (expected every {self._heartbeat_seconds}s) — connection may be dead")
          self._status.connected = False

          # Force-close the WebSocket to trigger reconnect via _on_receive_loop_done
          if self._ws:
            with contextlib.suppress(Exception):
              await self._ws.close()

    except asyncio.CancelledError:
      pass

  def _reject_all_pending(self, reason: str) -> None:
    """Reject all pending command futures with an error."""
    if not self._pending:
      return
    count = len(self._pending)
    for cmd_id, future in list(self._pending.items()):
      if not future.done():
        future.set_result({"success": False, "error": reason})
    self._pending.clear()
    if count > 0:
      log_debug(f"[whatsapp:baileys] Rejected {count} pending command(s): {reason}")

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

  # --- Internal: process lifecycle ---

  async def _kill_process(self) -> None:
    """Terminate and fully reap the sidecar process."""
    if self._process is None:
      return

    proc = self._process
    self._process = None

    # Already exited?
    if proc.returncode is not None:
      return

    # Graceful terminate
    try:
      proc.terminate()
    except ProcessLookupError:
      return

    # Wait for exit with timeout
    try:
      await asyncio.wait_for(proc.wait(), timeout=5.0)
      return
    except asyncio.TimeoutError:
      pass

    # Force kill
    try:
      proc.kill()
    except ProcessLookupError:
      return

    # Always reap to prevent zombies
    with contextlib.suppress(Exception):
      await asyncio.wait_for(proc.wait(), timeout=3.0)

  # --- Internal: setup ---

  async def _ensure_npm_deps(self) -> None:
    """Run npm install in the bridge directory if node_modules is missing."""
    node_modules = self._bridge_dir / "node_modules"
    if node_modules.exists():
      # If the core dependency is missing but node_modules exists, clean up partial install
      if not (node_modules / "@whiskeysockets" / "baileys").exists() and (self._bridge_dir / "package.json").exists():
        log_warning("[whatsapp:baileys] Incomplete node_modules detected, reinstalling...")
        shutil.rmtree(node_modules, ignore_errors=True)
      else:
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
    media_error = msg.get("media_error")
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
    elif media_error:
      log_warning(f"[whatsapp:baileys] Media download failed for message {msg.get('id', '?')}: {media_error}")

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
