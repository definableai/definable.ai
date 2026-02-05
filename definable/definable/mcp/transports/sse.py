"""SSE transport for MCP servers over HTTP with Server-Sent Events."""

import asyncio
import contextlib
import json
import uuid
from typing import Any, Dict, Optional

from definable.mcp.errors import MCPConnectionError, MCPProtocolError, MCPTimeoutError
from definable.mcp.transports.base import BaseTransport
from definable.mcp.types import JSONRPCNotification, JSONRPCRequest, JSONRPCResponse
from definable.utils.log import log_debug, log_error, log_warning

try:
  import httpx
except ImportError:
  httpx = None  # type: ignore


class SSETransport(BaseTransport):
  """Transport for MCP servers using HTTP with Server-Sent Events.

  Uses HTTP POST for client-to-server messages and SSE for
  server-to-client messages.

  Example:
      transport = SSETransport(
          server_name="web",
          url="http://localhost:3000/sse",
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
    """Initialize the SSE transport.

    Args:
        server_name: Name of the server for logging/error messages.
        url: SSE endpoint URL.
        headers: Additional HTTP headers to send.
        connect_timeout: Timeout for initial SSE connection.
        request_timeout: Default timeout for requests.
    """
    super().__init__(server_name)

    if httpx is None:
      raise ImportError("httpx is required for SSE transport. Install with: pip install httpx")

    self._url: str = url
    self._headers: Dict[str, str] = headers or {}
    self._connect_timeout: float = connect_timeout
    self._request_timeout: float = request_timeout

    # Explicitly re-declare for type checking
    self._connected: bool = False

    # HTTP client
    self._client: Optional[httpx.AsyncClient] = None

    # SSE connection
    self._sse_response: Optional[httpx.Response] = None
    self._session_id: Optional[str] = None
    self._messages_endpoint: Optional[str] = None

    # Request tracking
    self._request_id = 0
    self._pending_requests: Dict[int, asyncio.Future[JSONRPCResponse]] = {}

    # Background reader task
    self._reader_task: Optional[asyncio.Task[None]] = None
    self._shutdown_event = asyncio.Event()

  async def connect(self) -> None:
    """Establish SSE connection to the MCP server.

    Raises:
        MCPConnectionError: If connection fails.
        MCPTimeoutError: If connection times out.
    """
    if self._connected:
      return

    log_debug(f"MCP [{self.server_name}] Connecting to SSE: {self._url}")

    # Create HTTP client
    self._client = httpx.AsyncClient(
      timeout=httpx.Timeout(
        connect=self._connect_timeout,
        read=self._request_timeout,
        write=self._request_timeout,
        pool=self._connect_timeout,
      ),
      headers=self._headers,
    )

    try:
      # Establish SSE connection
      self._sse_response = await self._client.stream(
        "GET",
        self._url,
        headers={"Accept": "text/event-stream"},
      ).__aenter__()

      if self._sse_response.status_code != 200:
        raise MCPConnectionError(
          f"SSE connection failed with status {self._sse_response.status_code}",
          server_name=self.server_name,
        )

      # Parse initial endpoint event
      # The server should send an 'endpoint' event with the messages URL
      async for line in self._sse_response.aiter_lines():
        line = line.strip()
        if not line:
          continue

        if line.startswith("event:"):
          event_type = line[6:].strip()
        elif line.startswith("data:"):
          data = line[5:].strip()
          if event_type == "endpoint":
            try:
              endpoint_data = json.loads(data)
              self._messages_endpoint = endpoint_data.get("url") or endpoint_data.get("endpoint")
              self._session_id = endpoint_data.get("sessionId") or str(uuid.uuid4())
              break
            except json.JSONDecodeError:
              # Data might just be a URL string
              self._messages_endpoint = data
              self._session_id = str(uuid.uuid4())
              break

      if not self._messages_endpoint:
        # Fallback: assume /message endpoint relative to base URL
        base_url = self._url.rsplit("/", 1)[0]
        self._messages_endpoint = f"{base_url}/message"
        self._session_id = str(uuid.uuid4())

      # Start background reader for SSE events
      self._shutdown_event.clear()
      self._reader_task = asyncio.create_task(self._read_sse_events())

      self._connected = True
      log_debug(f"MCP [{self.server_name}] SSE connected (session={self._session_id})")

    except asyncio.TimeoutError:
      await self._cleanup_client()
      raise MCPTimeoutError(
        f"Timeout connecting to MCP server '{self.server_name}'",
        server_name=self.server_name,
        timeout_seconds=self._connect_timeout,
      )
    except httpx.ConnectError as e:
      await self._cleanup_client()
      raise MCPConnectionError(
        f"Failed to connect to '{self._url}': {e}",
        server_name=self.server_name,
        original_error=e,
      )
    except Exception as e:
      await self._cleanup_client()
      if isinstance(e, (MCPConnectionError, MCPTimeoutError)):
        raise
      raise MCPConnectionError(
        f"Failed to connect to MCP server '{self.server_name}': {e}",
        server_name=self.server_name,
        original_error=e,
      )

  async def _cleanup_client(self) -> None:
    """Clean up HTTP client resources."""
    if self._sse_response:
      with contextlib.suppress(Exception):
        await self._sse_response.aclose()
      self._sse_response = None

    if self._client:
      with contextlib.suppress(Exception):
        await self._client.aclose()
      self._client = None

  async def disconnect(self) -> None:
    """Close the SSE connection.

    Idempotent - safe to call multiple times.
    """
    if not self._connected:
      return

    self._connected = False
    self._shutdown_event.set()

    # Cancel pending requests
    for future in self._pending_requests.values():
      if not future.done():
        future.cancel()
    self._pending_requests.clear()

    # Stop reader task
    if self._reader_task and not self._reader_task.done():
      self._reader_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._reader_task
      self._reader_task = None

    # Close HTTP resources
    await self._cleanup_client()

    self._session_id = None
    self._messages_endpoint = None

    log_debug(f"MCP [{self.server_name}] SSE disconnected")

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
    if not self._connected or not self._client or not self._messages_endpoint:
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

    # Create future for response (if using SSE for responses)
    future: asyncio.Future[JSONRPCResponse] = asyncio.get_event_loop().create_future()
    self._pending_requests[request_id] = future

    log_debug(f"MCP [{self.server_name}] -> {method} (id={request_id})")

    try:
      # Send request via HTTP POST
      response = await self._client.post(
        self._messages_endpoint,
        json=request.model_dump(exclude_none=True),
        headers={
          "Content-Type": "application/json",
          **({"X-Session-Id": self._session_id} if self._session_id else {}),
        },
        timeout=timeout or self._request_timeout,
      )

      # Some servers return response directly in POST response
      if response.status_code == 200:
        try:
          data = response.json()
          if "result" in data or "error" in data:
            self._pending_requests.pop(request_id, None)
            return JSONRPCResponse.model_validate(data)
        except (json.JSONDecodeError, ValueError):
          pass

      # Otherwise wait for response via SSE
      try:
        rpc_response = await asyncio.wait_for(
          future,
          timeout=timeout or self._request_timeout,
        )
        return rpc_response
      except asyncio.TimeoutError:
        self._pending_requests.pop(request_id, None)
        raise MCPTimeoutError(
          f"Request '{method}' to '{self.server_name}' timed out",
          server_name=self.server_name,
          timeout_seconds=timeout or self._request_timeout,
        )

    except asyncio.TimeoutError:
      self._pending_requests.pop(request_id, None)
      raise MCPTimeoutError(
        f"Request '{method}' to '{self.server_name}' timed out",
        server_name=self.server_name,
        timeout_seconds=timeout or self._request_timeout,
      )
    except httpx.HTTPStatusError as e:
      self._pending_requests.pop(request_id, None)
      raise MCPConnectionError(
        f"HTTP error from '{self.server_name}': {e.response.status_code}",
        server_name=self.server_name,
        original_error=e,
      )
    except Exception as e:
      self._pending_requests.pop(request_id, None)
      if isinstance(e, (MCPConnectionError, MCPTimeoutError)):
        raise
      raise MCPConnectionError(
        f"Request to '{self.server_name}' failed: {e}",
        server_name=self.server_name,
        original_error=e,
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
    if not self._connected or not self._client or not self._messages_endpoint:
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
      await self._client.post(
        self._messages_endpoint,
        json=notification.model_dump(exclude_none=True),
        headers={
          "Content-Type": "application/json",
          **({"X-Session-Id": self._session_id} if self._session_id else {}),
        },
      )
    except Exception as e:
      raise MCPConnectionError(
        f"Notification to '{self.server_name}' failed: {e}",
        server_name=self.server_name,
        original_error=e,
      )

  async def _read_sse_events(self) -> None:
    """Background task that reads SSE events and dispatches responses."""
    if not self._sse_response:
      return

    event_type = ""
    try:
      async for line in self._sse_response.aiter_lines():
        if self._shutdown_event.is_set():
          break

        line = line.strip()
        if not line:
          event_type = ""
          continue

        if line.startswith("event:"):
          event_type = line[6:].strip()
        elif line.startswith("data:"):
          data_str = line[5:].strip()
          if event_type == "message":
            try:
              data = json.loads(data_str)
              self._handle_message(data)
            except json.JSONDecodeError as e:
              log_warning(f"MCP [{self.server_name}] Invalid JSON in SSE: {e}")

    except asyncio.CancelledError:
      pass
    except Exception as e:
      if not self._shutdown_event.is_set():
        log_error(f"MCP [{self.server_name}] SSE reader error: {e}")

    # Mark connection as closed
    if self._connected and not self._shutdown_event.is_set():
      self._connected = False
      for future in self._pending_requests.values():
        if not future.done():
          future.set_exception(
            MCPConnectionError(
              f"Connection to '{self.server_name}' lost",
              server_name=self.server_name,
            )
          )
      self._pending_requests.clear()

  def _handle_message(self, data: Dict[str, Any]) -> None:
    """Handle an incoming JSON-RPC message from SSE.

    Args:
        data: Parsed JSON message.
    """
    # Handle response (has id)
    if "id" in data and data["id"] is not None:
      request_id = data["id"]
      if request_id in self._pending_requests:
        future = self._pending_requests.pop(request_id)
        if not future.done():
          try:
            response = JSONRPCResponse.model_validate(data)
            log_debug(f"MCP [{self.server_name}] <- response (id={request_id})")
            future.set_result(response)
          except Exception as e:
            future.set_exception(
              MCPProtocolError(
                f"Invalid response: {e}",
                server_name=self.server_name,
              )
            )
      else:
        log_warning(f"MCP [{self.server_name}] Unexpected response id={request_id}")

    # Handle notification (no id)
    elif "method" in data:
      method = data.get("method", "unknown")
      log_debug(f"MCP [{self.server_name}] <- {method} (notification)")
      # Notifications currently logged but not dispatched
