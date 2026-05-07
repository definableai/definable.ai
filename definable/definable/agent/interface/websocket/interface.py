"""WebSocket interface — bidirectional JSON agent endpoint via FastAPI.

Wire protocol::

    Client -> Server: {"text": "hello"}
    Server -> Client: {"content": "hi back"}
    Server -> Client: {"error": "..."}             (on failure)

Usage::

    from definable import Agent
    from definable.agent.interface.websocket import WebSocketInterface

    agent = Agent(name="ws", model="openai/gpt-5.4-mini")
    iface = WebSocketInterface(agent, host="0.0.0.0", port=8765)
    async with iface:
      await iface.serve()    # blocks until cancelled

Each new WS connection is its own conversation. The base class wires
`_convert` (raw JSON -> prompt text) and `_send` (text -> JSON reply);
the FastAPI/uvicorn glue lives entirely inside aopen / aclose.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from definable.agent.interface.base import Interface

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class WebSocketInterface(Interface):
  """Self-contained FastAPI WebSocket server bound to one Agent."""

  def __init__(
    self,
    agent: Agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/ws",
  ) -> None:
    super().__init__(agent)
    self.host = host
    self.port = port
    self.path = path
    self._server: Any = None
    self._app: Any = None
    self._serve_task: asyncio.Task[Any] | None = None

  async def aopen(self) -> None:
    # Lazy imports — fastapi/uvicorn are optional extras.
    try:
      import uvicorn
      from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as e:
      raise ImportError("WebSocketInterface requires fastapi + uvicorn — `pip install definable[serve]`") from e

    app = FastAPI()

    @app.websocket(self.path)
    async def _ws_endpoint(ws: WebSocket) -> None:
      await ws.accept()
      try:
        while True:
          raw = await ws.receive_json()
          await self.handle((ws, raw))
      except WebSocketDisconnect:
        return

    config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
    self._app = app
    self._server = uvicorn.Server(config)
    self._serve_task = asyncio.create_task(self._server.serve())
    # Brief settle so clients can connect immediately after aopen returns.
    await asyncio.sleep(0.05)

  async def aclose(self) -> None:
    if self._server is not None:
      self._server.should_exit = True
    if self._serve_task is not None:
      self._serve_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._serve_task

  async def _convert(self, raw_message: Any) -> str:
    _ws, payload = raw_message
    text = payload.get("text") if isinstance(payload, dict) else None
    return str(text) if text else ""

  async def _send(self, raw_message: Any, reply: str) -> None:
    ws, _payload = raw_message
    await ws.send_json({"content": reply})
