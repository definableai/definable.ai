"""DesktopInterface — local WebSocket chat interface for macOS desktop agents.

Starts an optional WebSocket server on localhost so you can chat with an agent
from any WebSocket client. Primary use-case is local development and testing.
For production remote access, use TelegramInterface or DiscordInterface.

Example::

    from definable.agent.interface.desktop import DesktopInterface, DesktopConfig

    interface = DesktopInterface(
        agent=agent,
        config=DesktopConfig(websocket_port=8765),
    )
    async with interface:
        await interface.serve_forever()
"""

import contextlib
import json
from typing import Any, Optional, Set

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.desktop.config import DesktopConfig
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.utils.log import log_error, log_info, log_warning


class DesktopInterface(BaseInterface):
  """Local WebSocket interface for desktop-side agent interaction.

  Exposes the agent over a WebSocket server on localhost. Each connected
  client gets its own session. Messages are plain JSON:

  Inbound::

      {"text": "What apps are open?", "user_id": "local"}

  Outbound::

      {"content": "Safari, Terminal, VS Code are running.", "images": []}

  If ``websocket_port`` is ``0`` (default), no server is started — useful when
  the agent is driven entirely through the MacOS skill from another interface.

  Args:
    agent: Agent instance to connect.
    config: :class:`DesktopConfig` instance.
    **kwargs: Forwarded to :class:`BaseInterface`.
  """

  def __init__(self, *, agent: Any = None, config: DesktopConfig, **kwargs: Any) -> None:
    super().__init__(agent=agent, config=config, **kwargs)
    self._config: DesktopConfig = config
    self._server: Any = None  # websockets.WebSocketServer
    self._connected: Set[Any] = set()

  async def _start_receiver(self) -> None:
    port = self._config.websocket_port
    if port == 0:
      log_info("[desktop] WebSocket server disabled (websocket_port=0)")
      return

    try:
      import websockets  # type: ignore[import-not-found]
    except ImportError as exc:
      raise ImportError("DesktopInterface requires 'websockets'. Install with: pip install 'definable[desktop]'") from exc

    self._server = await websockets.serve(
      self._handle_ws_connection,
      self._config.bridge_host,
      port,
    )
    log_info(f"[desktop] WebSocket server listening on ws://{self._config.bridge_host}:{port}")

  async def _stop_receiver(self) -> None:
    if self._server is not None:
      self._server.close()
      with contextlib.suppress(Exception):
        await self._server.wait_closed()
      self._server = None
    log_info("[desktop] WebSocket server stopped")

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """Parse a raw WebSocket message dict into an InterfaceMessage."""
    try:
      text = raw_message.get("text", "").strip()
      user_id = str(raw_message.get("user_id", "local"))
      if not text:
        return None
      return InterfaceMessage(
        platform="desktop",
        platform_user_id=user_id,
        platform_chat_id=user_id,
        platform_message_id="",
        text=text,
      )
    except Exception as e:
      log_warning(f"[desktop] Failed to parse inbound message: {e}")
      return None

  async def _send_response(self, original_msg: InterfaceMessage, response: InterfaceResponse, raw_message: Any) -> None:
    """Send a JSON response back through the raw WebSocket connection."""
    websocket = raw_message.get("_ws")
    if websocket is None:
      return
    payload: dict[str, Any] = {"content": response.content or ""}
    if response.images:
      payload["images"] = response.images
    try:
      await websocket.send(json.dumps(payload))
    except Exception as e:
      log_warning(f"[desktop] Failed to send WebSocket response: {e}")

  async def _handle_ws_connection(self, websocket: Any) -> None:
    """Handle a single WebSocket client connection."""
    self._connected.add(websocket)
    log_info(f"[desktop] Client connected (total: {len(self._connected)})")
    try:
      async for raw in websocket:
        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          data = {"text": str(raw)}
        # Inject the live websocket so _send_response can write back
        data["_ws"] = websocket
        await self.handle_platform_message(data)
    except Exception as e:
      log_error(f"[desktop] WebSocket error: {e}")
    finally:
      self._connected.discard(websocket)
      log_info(f"[desktop] Client disconnected (total: {len(self._connected)})")
