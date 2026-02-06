"""Stdio transport for MCP servers running as subprocesses."""

import asyncio
import contextlib
import json
import os
from typing import Any, Dict, Optional

from definable.mcp.errors import MCPConnectionError, MCPProtocolError, MCPTimeoutError
from definable.mcp.transports.base import BaseTransport
from definable.mcp.types import JSONRPCNotification, JSONRPCRequest, JSONRPCResponse
from definable.utils.log import log_debug, log_error, log_warning


class StdioTransport(BaseTransport):
  """Transport for MCP servers communicating via stdin/stdout.

  Launches the MCP server as a subprocess and communicates using
  newline-delimited JSON-RPC messages over stdio.

  Example:
      transport = StdioTransport(
          server_name="filesystem",
          command="npx",
          args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      )
      async with transport:
          response = await transport.send_request("tools/list")
  """

  def __init__(
    self,
    server_name: str,
    command: str,
    args: Optional[list] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    connect_timeout: float = 30.0,
    request_timeout: float = 60.0,
  ) -> None:
    """Initialize the stdio transport.

    Args:
        server_name: Name of the server for logging/error messages.
        command: Command to execute (e.g., "npx", "python").
        args: Command arguments.
        env: Environment variables (merged with current env).
        cwd: Working directory for the subprocess.
        connect_timeout: Timeout for starting subprocess.
        request_timeout: Default timeout for requests.
    """
    super().__init__(server_name)
    self._command: str = command
    self._args: list = args or []
    self._env: Optional[Dict[str, str]] = env
    self._cwd: Optional[str] = cwd
    self._connect_timeout: float = connect_timeout
    self._request_timeout: float = request_timeout

    # Explicitly re-declare for type checking
    self._connected: bool = False

    # Process handle
    self._process: Optional[asyncio.subprocess.Process] = None

    # Request tracking
    self._request_id = 0
    self._pending_requests: Dict[int, asyncio.Future[JSONRPCResponse]] = {}

    # Background reader task
    self._reader_task: Optional[asyncio.Task[None]] = None
    self._shutdown_event: Optional[asyncio.Event] = None

  async def connect(self) -> None:
    """Start the subprocess and establish communication.

    Raises:
        MCPConnectionError: If subprocess fails to start.
        MCPTimeoutError: If connection times out.
    """
    if self._connected:
      return

    # Build environment
    process_env = os.environ.copy()
    if self._env:
      process_env.update(self._env)

    log_debug(f"MCP [{self.server_name}] Starting subprocess: {self._command} {' '.join(self._args)}")

    try:
      self._process = await asyncio.wait_for(
        asyncio.create_subprocess_exec(
          self._command,
          *self._args,
          stdin=asyncio.subprocess.PIPE,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
          env=process_env,
          cwd=self._cwd,
        ),
        timeout=self._connect_timeout,
      )
    except asyncio.TimeoutError:
      raise MCPTimeoutError(
        f"Timeout starting MCP server '{self.server_name}'",
        server_name=self.server_name,
        timeout_seconds=self._connect_timeout,
      )
    except FileNotFoundError:
      raise MCPConnectionError(
        f"Command not found: {self._command}",
        server_name=self.server_name,
      )
    except PermissionError:
      raise MCPConnectionError(
        f"Permission denied executing: {self._command}",
        server_name=self.server_name,
      )
    except Exception as e:
      raise MCPConnectionError(
        f"Failed to start MCP server '{self.server_name}': {e}",
        server_name=self.server_name,
        original_error=e,
      )

    # Start background reader (create Event in async context to bind to current loop)
    self._shutdown_event = asyncio.Event()
    self._reader_task = asyncio.create_task(self._read_responses())

    self._connected = True
    log_debug(f"MCP [{self.server_name}] Subprocess started (pid={self._process.pid})")

  async def disconnect(self) -> None:
    """Stop the subprocess and clean up.

    Idempotent - safe to call multiple times.
    """
    if not self._connected:
      return

    self._connected = False
    if self._shutdown_event:
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

    # Terminate process
    if self._process:
      try:
        self._process.terminate()
        try:
          await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
          log_warning(f"MCP [{self.server_name}] Process did not terminate, killing")
          self._process.kill()
          await self._process.wait()
      except ProcessLookupError:
        pass  # Already terminated
      except Exception as e:
        log_error(f"MCP [{self.server_name}] Error terminating process: {e}")
      finally:
        self._process = None

    log_debug(f"MCP [{self.server_name}] Disconnected")

  async def send_request(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
  ) -> JSONRPCResponse:
    """Send a JSON-RPC request and wait for response.

    Args:
        method: JSON-RPC method name.
        params: Method parameters.
        timeout: Request timeout (uses default if not specified).

    Returns:
        JSONRPCResponse with result or error.

    Raises:
        MCPConnectionError: If not connected or write fails.
        MCPTimeoutError: If request times out.
        MCPProtocolError: If response is invalid.
    """
    if not self._connected or not self._process or not self._process.stdin:
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

    # Create future for response
    future: asyncio.Future[JSONRPCResponse] = asyncio.get_event_loop().create_future()
    self._pending_requests[request_id] = future

    # Send request
    try:
      message = request.model_dump_json(exclude_none=True) + "\n"
      log_debug(f"MCP [{self.server_name}] -> {method} (id={request_id})")
      self._process.stdin.write(message.encode())
      await self._process.stdin.drain()
    except Exception as e:
      self._pending_requests.pop(request_id, None)
      raise MCPConnectionError(
        f"Failed to send request to '{self.server_name}': {e}",
        server_name=self.server_name,
        original_error=e,
      )

    # Wait for response
    try:
      response = await asyncio.wait_for(
        future,
        timeout=timeout or self._request_timeout,
      )
      return response
    except asyncio.TimeoutError:
      self._pending_requests.pop(request_id, None)
      raise MCPTimeoutError(
        f"Request '{method}' to '{self.server_name}' timed out",
        server_name=self.server_name,
        timeout_seconds=timeout or self._request_timeout,
      )
    except asyncio.CancelledError:
      self._pending_requests.pop(request_id, None)
      raise

  async def send_notification(
    self,
    method: str,
    params: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Send a JSON-RPC notification (no response expected).

    Args:
        method: JSON-RPC method name.
        params: Method parameters.

    Raises:
        MCPConnectionError: If not connected or write fails.
    """
    if not self._connected or not self._process or not self._process.stdin:
      raise MCPConnectionError(
        f"MCP server '{self.server_name}' not connected",
        server_name=self.server_name,
      )

    notification = JSONRPCNotification(
      method=method,
      params=params,
    )

    try:
      message = notification.model_dump_json(exclude_none=True) + "\n"
      log_debug(f"MCP [{self.server_name}] -> {method} (notification)")
      self._process.stdin.write(message.encode())
      await self._process.stdin.drain()
    except Exception as e:
      raise MCPConnectionError(
        f"Failed to send notification to '{self.server_name}': {e}",
        server_name=self.server_name,
        original_error=e,
      )

  async def _read_responses(self) -> None:
    """Background task that reads and dispatches responses."""
    if not self._process or not self._process.stdout:
      return

    try:
      while self._shutdown_event and not self._shutdown_event.is_set():
        # Read line from stdout
        try:
          line = await self._process.stdout.readline()
        except Exception as e:
          if self._shutdown_event and not self._shutdown_event.is_set():
            log_error(f"MCP [{self.server_name}] Read error: {e}")
          break

        if not line:
          if self._shutdown_event and not self._shutdown_event.is_set():
            log_warning(f"MCP [{self.server_name}] Server closed stdout")
          break

        # Parse JSON-RPC message
        try:
          data = json.loads(line.decode())
        except json.JSONDecodeError as e:
          log_warning(f"MCP [{self.server_name}] Invalid JSON: {e}")
          continue

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
        else:
          method = data.get("method", "unknown")
          log_debug(f"MCP [{self.server_name}] <- {method} (notification)")
          # Notifications are currently logged but not dispatched
          # Future: add notification handlers

    except asyncio.CancelledError:
      pass
    except Exception as e:
      if self._shutdown_event and not self._shutdown_event.is_set():
        log_error(f"MCP [{self.server_name}] Reader error: {e}")

    # Mark connection as closed
    if self._connected and (not self._shutdown_event or not self._shutdown_event.is_set()):
      self._connected = False
      # Fail all pending requests
      for future in self._pending_requests.values():
        if not future.done():
          future.set_exception(
            MCPConnectionError(
              f"Connection to '{self.server_name}' lost",
              server_name=self.server_name,
            )
          )
      self._pending_requests.clear()
