"""DesktopInterface — local WebSocket chat interface for macOS desktop agents.

Starts an optional WebSocket server on localhost so you can chat with an agent
from any WebSocket client. Primary use-case is local development and testing.
For production remote access, use TelegramInterface or DiscordInterface.

Example::

    from definable.agent.interface.desktop import DesktopInterface

    interface = DesktopInterface(
        agent=agent,
        websocket_port=8765,
    )
    async with interface:
        await interface.serve_forever()
"""

import contextlib
import json
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.desktop.config import DesktopConfig
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.utils.log import log_error, log_info, log_warning

if TYPE_CHECKING:
  from websockets.legacy.server import WebSocketServerProtocol
  from websockets.server import Server as WebSocketServer  # type: ignore[attr-defined]

  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver


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

  Example::

      interface = DesktopInterface(agent=agent, websocket_port=8765)
  """

  def __init__(
    self,
    *,
    # Desktop-specific
    bridge_host: str = "127.0.0.1",
    bridge_port: int = 7777,
    bridge_token: Optional[str] = None,
    auto_screenshot: bool = False,
    screenshot_on_error: bool = True,
    websocket_port: int = 0,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 100_000,
    rate_limit_messages_per_minute: int = 30,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    # Deprecated
    config: Optional[DesktopConfig] = None,
  ) -> None:
    if config is not None:
      warnings.warn(
        "Passing config= to DesktopInterface is deprecated. Pass bridge_host, websocket_port, and other params directly as keyword arguments.",
        DeprecationWarning,
        stacklevel=2,
      )
      resolved_config = config
    else:
      resolved_config = DesktopConfig(
        bridge_host=bridge_host,
        bridge_port=bridge_port,
        bridge_token=bridge_token,
        auto_screenshot=auto_screenshot,
        screenshot_on_error=screenshot_on_error,
        websocket_port=websocket_port,
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
    self._config: DesktopConfig = resolved_config
    self._server: Optional["WebSocketServer"] = None
    self._connected: Set["WebSocketServerProtocol"] = set()

  async def _start_receiver(self) -> None:
    port = self._config.websocket_port
    if port == 0:
      log_info("[desktop] WebSocket server disabled (websocket_port=0)")
      return

    try:
      import websockets
    except ImportError as exc:
      raise ImportError("DesktopInterface requires 'websockets'. Install with: pip install 'definable[desktop]'") from exc

    self._server = await websockets.serve(
      self._handle_ws_connection,  # type: ignore[arg-type]
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
    payload: Dict[str, Any] = {"content": response.content or ""}
    if response.images:
      payload["images"] = response.images
    try:
      await websocket.send(json.dumps(payload))
    except Exception as e:
      log_warning(f"[desktop] Failed to send WebSocket response: {e}")

  async def _handle_ws_connection(self, websocket: "WebSocketServerProtocol") -> None:
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
