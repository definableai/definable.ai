"""Base transport interface for MCP protocol."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from definable.mcp.types import JSONRPCResponse


class BaseTransport(ABC):
  """Abstract base class for MCP transports.

  Transports handle the low-level communication with MCP servers,
  including sending JSON-RPC messages and receiving responses.

  Supported transports:
  - StdioTransport: Communicate with subprocess servers via stdin/stdout
  - SSETransport: Communicate with HTTP servers using Server-Sent Events
  """

  def __init__(self, server_name: str) -> None:
    """Initialize the transport.

    Args:
        server_name: Name of the server for logging/error messages.
    """
    self.server_name: str = server_name
    self._connected: bool = False

  @property
  def connected(self) -> bool:
    """Check if transport is connected."""
    return self._connected

  @abstractmethod
  async def connect(self) -> None:
    """Establish connection to the MCP server.

    Raises:
        MCPConnectionError: If connection fails.
        MCPTimeoutError: If connection times out.
    """
    pass

  @abstractmethod
  async def disconnect(self) -> None:
    """Close connection to the MCP server.

    Should be idempotent - calling on already disconnected transport is safe.
    """
    pass

  @abstractmethod
  async def send_request(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
  ) -> JSONRPCResponse:
    """Send a JSON-RPC request and wait for response.

    Args:
        method: JSON-RPC method name.
        params: Method parameters (optional).
        timeout: Request timeout in seconds (optional).

    Returns:
        JSONRPCResponse containing result or error.

    Raises:
        MCPConnectionError: If not connected or connection lost.
        MCPTimeoutError: If request times out.
        MCPProtocolError: If response is invalid.
    """
    pass

  @abstractmethod
  async def send_notification(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Send a JSON-RPC notification (no response expected).

    Args:
        method: JSON-RPC method name.
        params: Method parameters (optional).

    Raises:
        MCPConnectionError: If not connected or connection lost.
    """
    pass

  async def __aenter__(self) -> "BaseTransport":
    """Async context manager entry."""
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit."""
    await self.disconnect()
