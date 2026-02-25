"""Configuration for WebSocket interface."""

from dataclasses import dataclass

from definable.agent.interface.config import InterfaceConfig


@dataclass(frozen=True)
class WebSocketConfig(InterfaceConfig):
  """Configuration for the WebSocket interface.

  Attributes:
    path: WebSocket endpoint path (mounted on AgentServer).
    heartbeat_interval: Seconds between heartbeat pings (0 to disable).
    max_connections: Maximum concurrent WebSocket connections.
    auth_on_connect: Require auth on WebSocket upgrade (vs per-message).
    message_format: Wire protocol format ("json" only for now).
  """

  platform: str = "websocket"
  path: str = "/ws"
  heartbeat_interval: float = 30.0
  max_connections: int = 100
  auth_on_connect: bool = True
  message_format: str = "json"
  max_message_length: int = 65536
