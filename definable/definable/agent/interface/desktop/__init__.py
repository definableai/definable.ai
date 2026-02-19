"""Definable Desktop — bridge client and optional local WebSocket interface.

Provides:
- :class:`BridgeClient`: Low-level HTTP client for the Swift bridge app.
- :class:`DesktopConfig`: Configuration for the desktop interface and bridge.
- :class:`DesktopInterface`: Optional local WebSocket chat interface.

Quick Start::

    from definable.agent.interface.desktop import BridgeClient, DesktopConfig, DesktopInterface
"""

from definable.agent.interface.desktop.bridge_client import BridgeClient
from definable.agent.interface.desktop.config import DesktopConfig
from definable.agent.interface.desktop.interface import DesktopInterface

__all__ = ["BridgeClient", "DesktopConfig", "DesktopInterface"]
