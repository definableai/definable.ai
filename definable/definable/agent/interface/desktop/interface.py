"""Desktop interface — localhost WebSocket chat for development.

Minimal port: websocket server bound to 127.0.0.1. JSON in / JSON out.
Each connection is its own conversation.

Removed (vs original): the bridge_client.py macOS Vapor 4 sidecar HTTP
client (camera / screen / OCR / shell). That belongs as a Toolkit, not
embedded in the interface — keep concerns separate.

Requires `pip install websockets`.

Usage::

    iface = DesktopInterface(agent, port=8765)
    async with iface:
      await iface.serve()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

from definable.agent.interface.base import Interface
from definable.utils.log import log_error, log_info, log_warning

if TYPE_CHECKING:
  from websockets.legacy.server import WebSocketServerProtocol
  from websockets.server import Server as WebSocketServer  # type: ignore[attr-defined]

  from definable.agent.agent import Agent


class DesktopInterface(Interface):
  """Localhost WebSocket adapter for development / desktop UIs."""

  def __init__(
    self,
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
  ) -> None:
    super().__init__(agent)
    self.host = host
    self.port = port
    self._server: WebSocketServer | None = None

  async def aopen(self) -> None:
    try:
      import websockets
    except ImportError as e:
      raise ImportError("DesktopInterface requires 'websockets' — `pip install websockets`") from e

    self._server = await websockets.serve(self._handle_connection, self.host, self.port)  # type: ignore[arg-type]
    log_info(f"[desktop] ws://{self.host}:{self.port}")

  async def aclose(self) -> None:
    if self._server is not None:
      self._server.close()
      with contextlib.suppress(Exception):
        await self._server.wait_closed()
      self._server = None
    log_info("[desktop] stopped")

  async def _convert(self, raw_message: Any) -> str:
    payload = raw_message.get("payload") if isinstance(raw_message, dict) else None
    if not payload:
      return ""
    return str(payload.get("text", "")).strip()

  async def _send(self, raw_message: Any, reply: str) -> None:
    ws = raw_message.get("ws") if isinstance(raw_message, dict) else None
    if ws is None:
      return
    try:
      await ws.send(json.dumps({"content": reply}))
    except Exception as e:
      log_warning(f"[desktop] send failed: {e}")

  async def _handle_connection(self, ws: WebSocketServerProtocol) -> None:
    log_info("[desktop] client connected")
    try:
      async for raw in ws:
        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          data = {"text": str(raw)}
        await self.handle({"ws": ws, "payload": data})
    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_error(f"[desktop] connection error: {e}")
    finally:
      log_info("[desktop] client disconnected")
