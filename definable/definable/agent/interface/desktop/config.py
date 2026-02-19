"""Configuration for the Desktop interface and Bridge client."""

from dataclasses import dataclass
from typing import Optional

from definable.agent.interface.config import InterfaceConfig


@dataclass(frozen=True)
class DesktopConfig(InterfaceConfig):
  """Configuration for the DesktopInterface and BridgeClient.

  Extends :class:`InterfaceConfig` with desktop-bridge–specific settings.

  Attributes:
    bridge_host: Hostname where the bridge listens (default: ``"127.0.0.1"``).
    bridge_port: Port the bridge listens on (default: ``7777``).
    bridge_token: Bearer token for the bridge. If ``None``, reads from
      ``~/.definable/bridge-token`` automatically.
    auto_screenshot: Attach a screenshot to every agent response (default: ``False``).
    screenshot_on_error: Capture a screenshot when a tool fails (default: ``True``).
    websocket_port: Port for the local WebSocket chat server. ``0`` disables it.
    max_message_length: Maximum message length in characters.
  """

  platform: str = "desktop"
  bridge_host: str = "127.0.0.1"
  bridge_port: int = 7777
  bridge_token: Optional[str] = None
  auto_screenshot: bool = False
  screenshot_on_error: bool = True
  websocket_port: int = 0
  max_message_length: int = 100_000
