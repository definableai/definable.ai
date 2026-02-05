"""HTTP transport for MCP servers using the Streamable HTTP protocol.

This implements the MCP Streamable HTTP transport where all communication
happens via POST requests and responses can be either JSON or SSE streams.
"""

import asyncio
import contextlib
import json
from typing import Any, Dict, Optional

from definable.mcp.errors import MCPConnectionError, MCPProtocolError, MCPTimeoutError
from definable.mcp.transports.base import BaseTransport
from definable.mcp.types import JSONRPCNotification, JSONRPCRequest, JSONRPCResponse
from definable.utils.log import log_debug, log_warning

try:
  import httpx
except ImportError:
  httpx = None  # type: ignore


class HTTPTransport(BaseTransport):
  """Transport for MCP servers using HTTP POST with SSE responses.

  This implements the MCP "Streamable HTTP" transport specification where:
  - All messages are sent via HTTP POST to a single endpoint
  - The Accept header must include both application/json and text/event-stream
  - Responses can be direct JSON or SSE streams with event: message

  Example:
      transport = HTTPTransport(
          server_name="gmail",
          url="https://example.com/mcp",
      )
      async with transport:
          response = await transport.send_request("tools/list")
  """

  def __init__(
    self,
    server_name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    connect_timeout: float = 30.0,
    request_timeout: float = 60.0,
  ) -> None:
    """Initialize the HTTP transport.

    Args:
        server_name: Name of the server for logging/error messages.
        url: MCP endpoint URL.
        headers: Additional HTTP headers to send.
        connect_timeout: Timeout for initial connection.
        request_timeout: Default timeout for requests.
    """
    super().__init__(server_name)

    if httpx is None:
      raise ImportError("httpx is required for HTTP transport. Install with: pip install httpx")

    self._url: str = url
    self._headers: Dict[str, str] = headers or {}
    self._connect_timeout: float = connect_timeout
    self._request_timeout: float = request_timeout

    # Explicitly re-declare for type checking
    self._connected: bool = False

    # HTTP client
    self._client: Optional[httpx.AsyncClient] = None

    # Request tracking
    self._request_id: int = 0

  async def connect(self) -> None:
    """Create HTTP client and verify connection.

    For HTTP transport, we just create the client - actual connection
    verification happens during the first request (initialize).

    Raises:
        MCPConnectionError: If client creation fails.
    """
    if self._connected:
      return

    log_debug(f"MCP [{self.server_name}] Creating HTTP client for: {self._url}")

    try:
      self._client = httpx.AsyncClient(
        timeout=httpx.Timeout(
          connect=self._connect_timeout,
          read=self._request_timeout,
          write=self._request_timeout,
          pool=self._connect_timeout,
        ),
        headers={
          **self._headers,
          "Content-Type": "application/json",
          "Accept": "application/json, text/event-stream",
        },
      )
      self._connected = True
      log_debug(f"MCP [{self.server_name}] HTTP client ready")

    except Exception as e:
      raise MCPConnectionError(
        f"Failed to create HTTP client for '{self.server_name}': {e}",
        server_name=self.server_name,
        original_error=e,
      )

  async def disconnect(self) -> None:
    """Close the HTTP client.

    Idempotent - safe to call multiple times.
    """
    if not self._connected:
      return

    self._connected = False

    if self._client:
      with contextlib.suppress(Exception):
        await self._client.aclose()
      self._client = None

    log_debug(f"MCP [{self.server_name}] HTTP client closed")

  async def send_request(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
  ) -> JSONRPCResponse:
    """Send a JSON-RPC request via HTTP POST.

    Args:
        method: JSON-RPC method name.
        params: Method parameters.
        timeout: Request timeout.

    Returns:
        JSONRPCResponse with result or error.

    Raises:
        MCPConnectionError: If not connected or request fails.
        MCPTimeoutError: If request times out.
        MCPProtocolError: If response is invalid.
    """
    if not self._connected or not self._client:
      raise MCPConnectionError(
        f"MCP server '{self.server_name}' not connected",
        server_name=self.server_name,
      )

    # Generate request ID
    self._request_id += 1
    request_id = self._request_id

    # Create request
    request = JSONRPCRequest(
      method=method,
      params=params,
      id=request_id,
    )

    log_debug(f"MCP [{self.server_name}] -> {method} (id={request_id})")

    try:
      response = await self._client.post(
        self._url,
        json=request.model_dump(exclude_none=True),
        timeout=timeout or self._request_timeout,
      )

      # Check for HTTP errors
      if response.status_code == 406:
        # Not Acceptable - wrong Accept header
        raise MCPProtocolError(
          "Server requires Accept: application/json, text/event-stream",
          server_name=self.server_name,
        )

      if response.status_code >= 400:
        raise MCPConnectionError(
          f"HTTP {response.status_code}: {response.text[:200]}",
          server_name=self.server_name,
        )

      # Parse response - can be either direct JSON or SSE
      content_type = response.headers.get("content-type", "")

      if "text/event-stream" in content_type:
        # Parse SSE response
        return self._parse_sse_response(response.text, request_id)
      else:
        # Direct JSON response
        try:
          data = response.json()
          return JSONRPCResponse.model_validate(data)
        except Exception as e:
          raise MCPProtocolError(
            f"Invalid JSON response: {e}",
            server_name=self.server_name,
          )

    except asyncio.TimeoutError:
      raise MCPTimeoutError(
        f"Request '{method}' to '{self.server_name}' timed out",
        server_name=self.server_name,
        timeout_seconds=timeout or self._request_timeout,
      )
    except httpx.ConnectError as e:
      raise MCPConnectionError(
        f"Connection to '{self.server_name}' failed: {e}",
        server_name=self.server_name,
        original_error=e,
      )
    except httpx.TimeoutException:
      raise MCPTimeoutError(
        f"Request '{method}' to '{self.server_name}' timed out",
        server_name=self.server_name,
        timeout_seconds=timeout or self._request_timeout,
      )
    except Exception as e:
      if isinstance(e, (MCPConnectionError, MCPTimeoutError, MCPProtocolError)):
        raise
      raise MCPConnectionError(
        f"Request to '{self.server_name}' failed: {e}",
        server_name=self.server_name,
        original_error=e,
      )

  def _parse_sse_response(self, text: str, expected_id: int) -> JSONRPCResponse:
    """Parse an SSE response containing JSON-RPC message.

    Args:
        text: SSE response text.
        expected_id: Expected request ID.

    Returns:
        Parsed JSONRPCResponse.

    Raises:
        MCPProtocolError: If response is invalid.
    """
    # SSE format:
    # event: message
    # data: {"jsonrpc":"2.0","result":...,"id":1}

    for line in text.split("\n"):
      line = line.strip()
      if line.startswith("data:"):
        data_str = line[5:].strip()
        try:
          data = json.loads(data_str)
          response = JSONRPCResponse.model_validate(data)
          log_debug(f"MCP [{self.server_name}] <- response (id={response.id})")
          return response
        except json.JSONDecodeError as e:
          raise MCPProtocolError(
            f"Invalid JSON in SSE data: {e}",
            server_name=self.server_name,
          )
        except Exception as e:
          raise MCPProtocolError(
            f"Invalid JSON-RPC response: {e}",
            server_name=self.server_name,
          )

    raise MCPProtocolError(
      "No data found in SSE response",
      server_name=self.server_name,
    )

  async def send_notification(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Send a JSON-RPC notification via HTTP POST.

    Args:
        method: JSON-RPC method name.
        params: Method parameters.

    Raises:
        MCPConnectionError: If not connected or request fails.
    """
    if not self._connected or not self._client:
      raise MCPConnectionError(
        f"MCP server '{self.server_name}' not connected",
        server_name=self.server_name,
      )

    notification = JSONRPCNotification(
      method=method,
      params=params,
    )

    log_debug(f"MCP [{self.server_name}] -> {method} (notification)")

    try:
      response = await self._client.post(
        self._url,
        json=notification.model_dump(exclude_none=True),
      )

      # For notifications, we don't expect a meaningful response
      # but we should check for errors
      if response.status_code >= 400:
        log_warning(f"MCP [{self.server_name}] Notification got HTTP {response.status_code}")

    except Exception as e:
      raise MCPConnectionError(
        f"Notification to '{self.server_name}' failed: {e}",
        server_name=self.server_name,
        original_error=e,
      )
