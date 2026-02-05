"""MCP client for connecting to and managing MCP servers.

Provides MCPServerConnection for single server connections and
MCPClient for managing multiple servers.
"""

import asyncio
from typing import Any, Dict, List, Optional

from definable.mcp.config import MCPConfig, MCPServerConfig
from definable.mcp.errors import (
  MCPConnectionError,
  MCPProtocolError,
  MCPServerNotFoundError,
  MCPToolNotFoundError,
)
from definable.mcp.protocol import validate_response
from definable.mcp.transports.base import BaseTransport
from definable.mcp.transports.http import HTTPTransport
from definable.mcp.transports.sse import SSETransport
from definable.mcp.transports.stdio import StdioTransport
from definable.mcp.types import (
  MCPClientInfo,
  MCPPromptDefinition,
  MCPPromptGetResult,
  MCPPromptListResult,
  MCPResource,
  MCPResourceContent,
  MCPResourceListResult,
  MCPResourceReadResult,
  MCPServerInfo,
  MCPToolCallResult,
  MCPToolDefinition,
  MCPToolListResult,
)
from definable.utils.log import log_debug, log_error, log_info, log_warning


class MCPServerConnection:
  """Connection to a single MCP server.

  Manages transport lifecycle, performs MCP handshake, and provides
  methods for interacting with the server's tools, resources, and prompts.

  Example:
      config = MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["/tmp"])
      connection = MCPServerConnection(config)
      async with connection:
          tools = await connection.list_tools()
          result = await connection.call_tool("read_file", {"path": "/tmp/test.txt"})
  """

  def __init__(self, config: MCPServerConfig) -> None:
    """Initialize server connection.

    Args:
        config: Server configuration.
    """
    self.config = config
    self.name = config.name

    self._transport: Optional[BaseTransport] = None
    self._server_info: Optional[MCPServerInfo] = None
    self._connected = False

    # Cached capabilities
    self._tools: Optional[List[MCPToolDefinition]] = None
    self._resources: Optional[List[MCPResource]] = None
    self._prompts: Optional[List[MCPPromptDefinition]] = None

  @property
  def connected(self) -> bool:
    """Check if connected to server."""
    return self._connected and self._transport is not None and self._transport.connected

  @property
  def server_info(self) -> Optional[MCPServerInfo]:
    """Get server info from initialization."""
    return self._server_info

  async def connect(self) -> None:
    """Connect to the MCP server.

    Creates transport, establishes connection, and performs
    MCP initialization handshake.

    Raises:
        MCPConnectionError: If connection fails.
        MCPTimeoutError: If connection times out.
        MCPProtocolError: If handshake fails.
    """
    if self._connected:
      return

    log_debug(f"MCP [{self.name}] Connecting...")

    # Create transport based on config
    if self.config.transport == "stdio":
      self._transport = StdioTransport(
        server_name=self.name,
        command=self.config.command,  # type: ignore
        args=self.config.args,
        env=self.config.env,
        cwd=self.config.cwd,
        connect_timeout=self.config.connect_timeout,
        request_timeout=self.config.request_timeout,
      )
    elif self.config.transport == "sse":
      self._transport = SSETransport(
        server_name=self.name,
        url=self.config.url,  # type: ignore
        headers=self.config.headers,
        connect_timeout=self.config.connect_timeout,
        request_timeout=self.config.request_timeout,
      )
    elif self.config.transport == "http":
      self._transport = HTTPTransport(
        server_name=self.name,
        url=self.config.url,  # type: ignore
        headers=self.config.headers,
        connect_timeout=self.config.connect_timeout,
        request_timeout=self.config.request_timeout,
      )
    else:
      raise MCPConnectionError(
        f"Unknown transport: {self.config.transport}",
        server_name=self.name,
      )

    # Connect transport
    await self._transport.connect()

    # Perform MCP handshake
    try:
      await self._initialize()
      self._connected = True
      log_info(f"MCP [{self.name}] Connected successfully")
    except Exception:
      await self._transport.disconnect()
      self._transport = None
      raise

  async def _initialize(self) -> None:
    """Perform MCP initialization handshake.

    Sends initialize request and notifications/initialized notification.

    Raises:
        MCPProtocolError: If handshake fails.
    """
    if not self._transport:
      raise MCPConnectionError(f"Transport not created for {self.name}", server_name=self.name)

    # Send initialize request
    client_info = MCPClientInfo()
    response = await self._transport.send_request(
      "initialize",
      params=client_info.model_dump(),
    )

    result = validate_response(response, self.name)

    try:
      self._server_info = MCPServerInfo.model_validate(result)
      log_debug(f"MCP [{self.name}] Server: {self._server_info.serverInfo.name} v{self._server_info.serverInfo.version}")
    except Exception as e:
      raise MCPProtocolError(
        f"Invalid initialize response: {e}",
        server_name=self.name,
      )

    # Send initialized notification
    await self._transport.send_notification("notifications/initialized")

  async def disconnect(self) -> None:
    """Disconnect from the MCP server.

    Idempotent - safe to call multiple times.
    """
    if not self._connected:
      return

    self._connected = False

    if self._transport:
      await self._transport.disconnect()
      self._transport = None

    # Clear cached data
    self._server_info = None
    self._tools = None
    self._resources = None
    self._prompts = None

    log_debug(f"MCP [{self.name}] Disconnected")

  async def reconnect(self) -> None:
    """Reconnect to the server with exponential backoff.

    Raises:
        MCPConnectionError: If reconnection fails after all attempts.
    """
    if not self.config.reconnect_on_failure:
      raise MCPConnectionError(
        f"Reconnection disabled for {self.name}",
        server_name=self.name,
      )

    await self.disconnect()

    delay = self.config.reconnect_delay
    for attempt in range(self.config.max_reconnect_attempts):
      try:
        log_info(f"MCP [{self.name}] Reconnecting (attempt {attempt + 1})")
        await self.connect()
        return
      except Exception as e:
        log_warning(f"MCP [{self.name}] Reconnect failed: {e}")
        if attempt < self.config.max_reconnect_attempts - 1:
          await asyncio.sleep(delay)
          delay *= 2  # Exponential backoff

    raise MCPConnectionError(
      f"Failed to reconnect to {self.name} after {self.config.max_reconnect_attempts} attempts",
      server_name=self.name,
    )

  # =========================================================================
  # Tools
  # =========================================================================

  async def list_tools(self, force_refresh: bool = False) -> List[MCPToolDefinition]:
    """List available tools from the server.

    Args:
        force_refresh: Force refresh from server (ignore cache).

    Returns:
        List of tool definitions (filtered by config).

    Raises:
        MCPConnectionError: If not connected.
        MCPProtocolError: If request fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    if self._tools is not None and not force_refresh:
      return self._tools

    response = await self._transport.send_request("tools/list")
    result = validate_response(response, self.name)

    try:
      list_result = MCPToolListResult.model_validate(result)
      all_tools = list_result.tools
    except Exception as e:
      raise MCPProtocolError(f"Invalid tools/list response: {e}", server_name=self.name)

    # Apply tool filtering
    self._tools = [t for t in all_tools if self.config.is_tool_allowed(t.name)]

    log_debug(f"MCP [{self.name}] Found {len(self._tools)} tools (filtered from {len(all_tools)})")
    return self._tools

  async def call_tool(
    self,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
  ) -> MCPToolCallResult:
    """Call a tool on the server.

    Args:
        tool_name: Name of the tool to call.
        arguments: Tool arguments.

    Returns:
        Tool call result.

    Raises:
        MCPConnectionError: If not connected.
        MCPToolNotFoundError: If tool not found or not allowed.
        MCPProtocolError: If call fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    # Check if tool is allowed
    if not self.config.is_tool_allowed(tool_name):
      raise MCPToolNotFoundError(
        f"Tool '{tool_name}' is blocked by configuration",
        tool_name=tool_name,
        server_name=self.name,
      )

    log_debug(f"MCP [{self.name}] Calling tool: {tool_name}")

    response = await self._transport.send_request(
      "tools/call",
      params={
        "name": tool_name,
        "arguments": arguments or {},
      },
    )

    result = validate_response(response, self.name)

    try:
      return MCPToolCallResult.model_validate(result)
    except Exception as e:
      raise MCPProtocolError(f"Invalid tools/call response: {e}", server_name=self.name)

  # =========================================================================
  # Resources
  # =========================================================================

  async def list_resources(self, force_refresh: bool = False) -> List[MCPResource]:
    """List available resources from the server.

    Args:
        force_refresh: Force refresh from server (ignore cache).

    Returns:
        List of resource definitions.

    Raises:
        MCPConnectionError: If not connected.
        MCPProtocolError: If request fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    if self._resources is not None and not force_refresh:
      return self._resources

    response = await self._transport.send_request("resources/list")
    result = validate_response(response, self.name)

    try:
      list_result = MCPResourceListResult.model_validate(result)
      resources = list_result.resources
      self._resources = resources
    except Exception as e:
      raise MCPProtocolError(f"Invalid resources/list response: {e}", server_name=self.name)

    log_debug(f"MCP [{self.name}] Found {len(resources)} resources")
    return resources

  async def read_resource(self, uri: str) -> List[MCPResourceContent]:
    """Read content from a resource.

    Args:
        uri: Resource URI.

    Returns:
        List of resource contents.

    Raises:
        MCPConnectionError: If not connected.
        MCPProtocolError: If read fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    log_debug(f"MCP [{self.name}] Reading resource: {uri}")

    response = await self._transport.send_request(
      "resources/read",
      params={"uri": uri},
    )

    result = validate_response(response, self.name)

    try:
      read_result = MCPResourceReadResult.model_validate(result)
      return read_result.contents
    except Exception as e:
      raise MCPProtocolError(f"Invalid resources/read response: {e}", server_name=self.name)

  # =========================================================================
  # Prompts
  # =========================================================================

  async def list_prompts(self, force_refresh: bool = False) -> List[MCPPromptDefinition]:
    """List available prompts from the server.

    Args:
        force_refresh: Force refresh from server (ignore cache).

    Returns:
        List of prompt definitions.

    Raises:
        MCPConnectionError: If not connected.
        MCPProtocolError: If request fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    if self._prompts is not None and not force_refresh:
      return self._prompts

    response = await self._transport.send_request("prompts/list")
    result = validate_response(response, self.name)

    try:
      list_result = MCPPromptListResult.model_validate(result)
      prompts = list_result.prompts
      self._prompts = prompts
    except Exception as e:
      raise MCPProtocolError(f"Invalid prompts/list response: {e}", server_name=self.name)

    log_debug(f"MCP [{self.name}] Found {len(prompts)} prompts")
    return prompts

  async def get_prompt(
    self,
    prompt_name: str,
    arguments: Optional[Dict[str, str]] = None,
  ) -> MCPPromptGetResult:
    """Get a rendered prompt from the server.

    Args:
        prompt_name: Name of the prompt.
        arguments: Prompt arguments.

    Returns:
        Rendered prompt result.

    Raises:
        MCPConnectionError: If not connected.
        MCPProtocolError: If request fails.
    """
    if not self.connected or not self._transport:
      raise MCPConnectionError(f"Not connected to {self.name}", server_name=self.name)

    log_debug(f"MCP [{self.name}] Getting prompt: {prompt_name}")

    response = await self._transport.send_request(
      "prompts/get",
      params={
        "name": prompt_name,
        "arguments": arguments or {},
      },
    )

    result = validate_response(response, self.name)

    try:
      return MCPPromptGetResult.model_validate(result)
    except Exception as e:
      raise MCPProtocolError(f"Invalid prompts/get response: {e}", server_name=self.name)

  # =========================================================================
  # Context Manager
  # =========================================================================

  async def __aenter__(self) -> "MCPServerConnection":
    """Async context manager entry."""
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit."""
    await self.disconnect()


class MCPClient:
  """Client for managing multiple MCP server connections.

  Aggregates tools, resources, and prompts across multiple servers
  and routes requests to the appropriate server.

  Example:
      config = MCPConfig(servers=[
          MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["/tmp"]),
          MCPServerConfig(name="web", transport="sse", url="http://localhost:3000"),
      ])

      client = MCPClient(config)
      async with client:
          # Get all tools from all servers
          tools = await client.list_all_tools()

          # Call tool on specific server
          result = await client.call_tool("fs", "read_file", {"path": "/tmp/test.txt"})
  """

  def __init__(self, config: MCPConfig) -> None:
    """Initialize MCP client.

    Args:
        config: MCP configuration with server definitions.
    """
    self.config = config
    self._connections: Dict[str, MCPServerConnection] = {}
    self._connected = False

  @property
  def connected(self) -> bool:
    """Check if any servers are connected."""
    return self._connected and any(c.connected for c in self._connections.values())

  @property
  def servers(self) -> List[str]:
    """Get list of configured server names."""
    return list(self._connections.keys())

  def get_connection(self, server_name: str) -> Optional[MCPServerConnection]:
    """Get connection for a specific server.

    Args:
        server_name: Server name.

    Returns:
        Server connection if exists, None otherwise.
    """
    return self._connections.get(server_name)

  async def connect(self) -> None:
    """Connect to all configured servers.

    Errors during connection are logged but don't prevent
    other servers from connecting.
    """
    if self._connected:
      return

    log_info(f"MCP Client: Connecting to {len(self.config.servers)} server(s)")

    for server_config in self.config.servers:
      connection = MCPServerConnection(server_config)
      self._connections[server_config.name] = connection

      if self.config.auto_connect:
        try:
          await connection.connect()
        except Exception as e:
          log_error(f"MCP Client: Failed to connect to '{server_config.name}': {e}")

    self._connected = True

  async def connect_server(self, server_name: str) -> None:
    """Connect to a specific server.

    Args:
        server_name: Name of server to connect.

    Raises:
        MCPServerNotFoundError: If server not in configuration.
    """
    connection = self._connections.get(server_name)
    if not connection:
      raise MCPServerNotFoundError(
        f"Server '{server_name}' not found",
        server_name=server_name,
        available_servers=list(self._connections.keys()),
      )
    await connection.connect()

  async def disconnect(self) -> None:
    """Disconnect from all servers.

    Idempotent - safe to call multiple times.
    """
    if not self._connected:
      return

    self._connected = False

    for connection in self._connections.values():
      try:
        await connection.disconnect()
      except Exception as e:
        log_error(f"MCP Client: Error disconnecting '{connection.name}': {e}")

    log_info("MCP Client: Disconnected from all servers")

  async def disconnect_server(self, server_name: str) -> None:
    """Disconnect from a specific server.

    Args:
        server_name: Name of server to disconnect.
    """
    connection = self._connections.get(server_name)
    if connection:
      await connection.disconnect()

  # =========================================================================
  # Tools
  # =========================================================================

  async def list_all_tools(self) -> Dict[str, List[MCPToolDefinition]]:
    """List tools from all connected servers.

    Returns:
        Dictionary mapping server name to list of tools.
    """
    result: Dict[str, List[MCPToolDefinition]] = {}
    for name, connection in self._connections.items():
      if connection.connected:
        try:
          tools = await connection.list_tools()
          result[name] = tools
        except Exception as e:
          log_warning(f"MCP Client: Failed to list tools from '{name}': {e}")
    return result

  async def list_tools(self, server_name: str) -> List[MCPToolDefinition]:
    """List tools from a specific server.

    Args:
        server_name: Server name.

    Returns:
        List of tool definitions.

    Raises:
        MCPServerNotFoundError: If server not found.
    """
    connection = self._connections.get(server_name)
    if not connection:
      raise MCPServerNotFoundError(
        f"Server '{server_name}' not found",
        server_name=server_name,
        available_servers=list(self._connections.keys()),
      )
    return await connection.list_tools()

  async def call_tool(
    self,
    server_name: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
  ) -> MCPToolCallResult:
    """Call a tool on a specific server.

    Args:
        server_name: Server name.
        tool_name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool call result.

    Raises:
        MCPServerNotFoundError: If server not found.
        MCPToolNotFoundError: If tool not found.
    """
    connection = self._connections.get(server_name)
    if not connection:
      raise MCPServerNotFoundError(
        f"Server '{server_name}' not found",
        server_name=server_name,
        available_servers=list(self._connections.keys()),
      )
    return await connection.call_tool(tool_name, arguments)

  def find_tool(self, tool_name: str) -> Optional[str]:
    """Find which server has a tool.

    Args:
        tool_name: Tool name to find.

    Returns:
        Server name if found, None otherwise.
    """
    for name, connection in self._connections.items():
      if connection._tools:
        for tool in connection._tools:
          if tool.name == tool_name:
            return name
    return None

  # =========================================================================
  # Resources
  # =========================================================================

  async def list_all_resources(self) -> Dict[str, List[MCPResource]]:
    """List resources from all connected servers.

    Returns:
        Dictionary mapping server name to list of resources.
    """
    result: Dict[str, List[MCPResource]] = {}
    for name, connection in self._connections.items():
      if connection.connected:
        try:
          resources = await connection.list_resources()
          result[name] = resources
        except Exception as e:
          log_warning(f"MCP Client: Failed to list resources from '{name}': {e}")
    return result

  async def read_resource(
    self,
    server_name: str,
    uri: str,
  ) -> List[MCPResourceContent]:
    """Read a resource from a specific server.

    Args:
        server_name: Server name.
        uri: Resource URI.

    Returns:
        Resource contents.

    Raises:
        MCPServerNotFoundError: If server not found.
    """
    connection = self._connections.get(server_name)
    if not connection:
      raise MCPServerNotFoundError(
        f"Server '{server_name}' not found",
        server_name=server_name,
        available_servers=list(self._connections.keys()),
      )
    return await connection.read_resource(uri)

  # =========================================================================
  # Prompts
  # =========================================================================

  async def list_all_prompts(self) -> Dict[str, List[MCPPromptDefinition]]:
    """List prompts from all connected servers.

    Returns:
        Dictionary mapping server name to list of prompts.
    """
    result: Dict[str, List[MCPPromptDefinition]] = {}
    for name, connection in self._connections.items():
      if connection.connected:
        try:
          prompts = await connection.list_prompts()
          result[name] = prompts
        except Exception as e:
          log_warning(f"MCP Client: Failed to list prompts from '{name}': {e}")
    return result

  async def get_prompt(
    self,
    server_name: str,
    prompt_name: str,
    arguments: Optional[Dict[str, str]] = None,
  ) -> MCPPromptGetResult:
    """Get a prompt from a specific server.

    Args:
        server_name: Server name.
        prompt_name: Prompt name.
        arguments: Prompt arguments.

    Returns:
        Rendered prompt.

    Raises:
        MCPServerNotFoundError: If server not found.
    """
    connection = self._connections.get(server_name)
    if not connection:
      raise MCPServerNotFoundError(
        f"Server '{server_name}' not found",
        server_name=server_name,
        available_servers=list(self._connections.keys()),
      )
    return await connection.get_prompt(prompt_name, arguments)

  # =========================================================================
  # Context Manager
  # =========================================================================

  async def __aenter__(self) -> "MCPClient":
    """Async context manager entry."""
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit."""
    await self.disconnect()
