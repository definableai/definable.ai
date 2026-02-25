"""WebSocket interface — real-time bidirectional agent communication via FastAPI."""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.agent.interface.websocket.config import WebSocketConfig
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver


class WebSocketInterface(BaseInterface):
  """Real-time bidirectional interface via WebSocket.

  Mounts a WebSocket endpoint on the AgentServer's FastAPI app.
  Clients connect, send JSON messages, and receive streamed responses.

  Wire protocol (JSON)::

    Client → Server:
      {"type": "message", "text": "Hello", "session_id": "...", "user_id": "..."}

    Server → Client:
      {"type": "response", "content": "Hi there!", "session_id": "...", "run_id": "..."}
      {"type": "error", "message": "..."}
      {"type": "heartbeat"}

  Example::

    interface = WebSocketInterface(
      agent=agent,
      path="/ws",
      heartbeat_interval=30,
    )
    runtime = AgentRuntime(agent, interfaces=[interface], enable_server=True)
    await runtime.start()
  """

  def __init__(
    self,
    *,
    # WebSocket-specific
    path: str = "/ws",
    heartbeat_interval: float = 30.0,
    max_connections: int = 100,
    auth_on_connect: bool = True,
    message_format: str = "json",
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 65536,
    rate_limit_messages_per_minute: int = 60,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    # Deprecated
    config: Optional[WebSocketConfig] = None,
  ) -> None:
    if config is not None:
      warnings.warn(
        "Passing config= to WebSocketInterface is deprecated. Pass params directly as keyword arguments.",
        DeprecationWarning,
        stacklevel=2,
      )
      resolved_config = config
    else:
      resolved_config = WebSocketConfig(
        path=path,
        heartbeat_interval=heartbeat_interval,
        max_connections=max_connections,
        auth_on_connect=auth_on_connect,
        message_format=message_format,
        max_session_history=max_session_history,
        session_ttl_seconds=session_ttl_seconds,
        max_concurrent_requests=max_concurrent_requests,
        error_message=error_message,
        typing_indicator=typing_indicator,
        max_message_length=max_message_length,
        rate_limit_messages_per_minute=rate_limit_messages_per_minute,
      )
    super().__init__(
      agent=agent,
      config=resolved_config,
      session_manager=session_manager,
      hooks=hooks,
      identity_resolver=identity_resolver,
      auth=auth,
    )
    self._ws_config: WebSocketConfig = self.config  # type: ignore[assignment]
    self._connections: Dict[str, Any] = {}  # conn_id → WebSocket
    self._connection_semaphore: Optional[asyncio.Semaphore] = None
    self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

  # --- Router for AgentServer ---

  def create_router(self) -> Any:
    """Create a FastAPI APIRouter with the WebSocket endpoint.

    Returns:
      FastAPI APIRouter instance.
    """
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect

    # Make WebSocket resolvable for string annotations (from __future__ import annotations)
    globals()["WebSocket"] = WebSocket

    router = APIRouter()

    @router.websocket(self._ws_config.path)
    async def websocket_endpoint(ws: WebSocket) -> None:
      # Connection limit check
      if self._connection_semaphore is not None:
        if self._connection_semaphore.locked():
          await ws.close(code=1013, reason="Max connections reached")
          return

      await ws.accept()
      conn_id = str(uuid4())
      self._connections[conn_id] = ws
      log_info(f"[websocket] Client connected: {conn_id} (total={len(self._connections)})")

      # Start heartbeat
      if self._ws_config.heartbeat_interval > 0:
        self._heartbeat_tasks[conn_id] = asyncio.create_task(self._heartbeat_loop(conn_id, ws))

      try:
        while True:
          raw = await ws.receive_text()
          try:
            data = json.loads(raw)
          except json.JSONDecodeError:
            await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
            continue

          msg_type = data.get("type", "message")
          if msg_type == "ping":
            await ws.send_text(json.dumps({"type": "pong"}))
            continue

          if msg_type == "message":
            # Build a raw_message dict for the pipeline
            raw_message = {
              "conn_id": conn_id,
              "websocket": ws,
              "data": data,
            }
            await self.handle_platform_message(raw_message)

      except WebSocketDisconnect:
        log_info(f"[websocket] Client disconnected: {conn_id}")
      except Exception as e:
        log_error(f"[websocket] Error on connection {conn_id}: {e}")
      finally:
        self._connections.pop(conn_id, None)
        task = self._heartbeat_tasks.pop(conn_id, None)
        if task is not None:
          task.cancel()

    return router

  # --- BaseInterface implementation ---

  async def _start_receiver(self) -> None:
    self._connection_semaphore = asyncio.Semaphore(self._ws_config.max_connections)
    log_info(f"[websocket] Receiver started (path={self._ws_config.path})")

  async def _stop_receiver(self) -> None:
    # Cancel all heartbeat tasks
    for task in self._heartbeat_tasks.values():
      task.cancel()
    self._heartbeat_tasks.clear()

    # Close all connections
    import contextlib

    for conn_id, ws in list(self._connections.items()):
      with contextlib.suppress(Exception):
        await ws.close(code=1001, reason="Server shutting down")
    self._connections.clear()
    log_info("[websocket] Receiver stopped")

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    data = raw_message["data"]
    conn_id = raw_message["conn_id"]

    text = data.get("text", "")
    session_id = data.get("session_id", conn_id)
    user_id = data.get("user_id", conn_id)

    if not text:
      return None

    return InterfaceMessage(
      text=text,
      platform="websocket",
      platform_user_id=user_id,
      platform_chat_id=session_id,
      platform_message_id="",
      metadata={"conn_id": conn_id},
    )

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    ws = raw_message["websocket"]
    payload = {
      "type": "response",
      "content": response.content or "",
    }
    if hasattr(original_msg, "metadata") and "conn_id" in original_msg.metadata:
      payload["conn_id"] = original_msg.metadata["conn_id"]

    try:
      await ws.send_text(json.dumps(payload))
    except Exception as e:
      log_error(f"[websocket] Failed to send response: {e}")

  # --- Heartbeat ---

  async def _heartbeat_loop(self, conn_id: str, ws: Any) -> None:
    """Send periodic heartbeat pings."""
    try:
      while True:
        await asyncio.sleep(self._ws_config.heartbeat_interval)
        try:
          await ws.send_text(json.dumps({"type": "heartbeat"}))
        except Exception:
          break
    except asyncio.CancelledError:
      pass

  # --- Introspection ---

  @property
  def active_connections(self) -> int:
    """Number of currently connected clients."""
    return len(self._connections)

  def needs_server(self) -> bool:
    """WebSocket interface always requires the HTTP server."""
    return True
