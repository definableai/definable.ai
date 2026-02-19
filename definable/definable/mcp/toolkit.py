"""MCPToolkit - Bridge MCP servers to Agent tools.

Exposes MCP server tools as standard Function objects compatible
with the Agent's tool system.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from definable.agent.toolkit import Toolkit
from definable.mcp.client import MCPClient
from definable.mcp.config import MCPConfig
from definable.mcp.types import MCPToolDefinition
from definable.utils.log import log_debug, log_error, log_info

if TYPE_CHECKING:
  from definable.tool.function import Function


class MCPToolkit(Toolkit):
  """Toolkit that exposes MCP server tools to agents.

  Connects to configured MCP servers and exposes their tools as
  standard Function objects that can be used by Agent.

  Example:
      config = MCPConfig(servers=[
          MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["/tmp"]),
      ])

      toolkit = MCPToolkit(config)

      async def main():
          async with toolkit:  # Connects to servers
              agent = Agent(model=model, toolkits=[toolkit])
              output = await agent.arun("List files in /tmp")

      asyncio.run(main())

  Tool Naming:
      By default, tools are named "{server}_{tool}" to avoid collisions
      when multiple servers provide tools with the same name.

      With tool_name_prefix="mcp_":
          - Server "fs" tool "read_file" -> "mcp_fs_read_file"

      With tool_name_prefix="" and include_server_prefix=False:
          - Server "fs" tool "read_file" -> "read_file" (may collide!)
  """

  def __init__(
    self,
    config: MCPConfig,
    client: Optional[MCPClient] = None,
    tool_name_prefix: str = "",
    include_server_prefix: bool = True,
    require_confirmation: bool = False,
  ) -> None:
    """Initialize the MCP toolkit.

    Args:
        config: MCP configuration with server definitions.
        client: Optional pre-configured MCPClient (will create one if not provided).
        tool_name_prefix: Prefix to add to all tool names.
        include_server_prefix: Include server name in tool names.
        require_confirmation: Mark all MCP tools as requiring confirmation.
    """
    # Dependencies include the MCP client reference for tool execution
    super().__init__(dependencies={"_mcp_toolkit_client": None})

    self.config = config
    self._provided_client = client
    self._client: Optional[MCPClient] = client
    self._tool_name_prefix = tool_name_prefix
    self._include_server_prefix = include_server_prefix
    self._require_confirmation = require_confirmation

    # Cached tools
    self._mcp_tools: Optional[List["Function"]] = None
    self._initialized = False

    # Tool name to server mapping
    self._tool_server_map: Dict[str, str] = {}

  @property
  def tools(self) -> List["Function"]:
    """Return MCP tools as Function objects.

    Note: This property requires the toolkit to be initialized
    (connected to servers) before it returns tools. If not
    initialized, returns an empty list.
    """
    if not self._initialized or self._mcp_tools is None:
      return []
    return self._mcp_tools

  @property
  def client(self) -> Optional[MCPClient]:
    """Get the underlying MCP client."""
    return self._client

  async def initialize(self) -> None:
    """Connect to MCP servers and discover tools.

    Must be called before the toolkit can be used with an agent.
    Automatically called when using as async context manager.

    Raises:
        MCPConnectionError: If connection to all servers fails.
    """
    if self._initialized:
      return

    log_info("MCPToolkit: Initializing...")

    # Create client if not provided
    if self._client is None:
      self._client = MCPClient(self.config)

    # Update dependencies with client reference
    self._dependencies["_mcp_toolkit_client"] = self._client

    # Connect to servers
    await self._client.connect()

    # Discover tools from all servers
    await self._discover_tools()

    self._initialized = True
    log_info(f"MCPToolkit: Ready with {len(self._mcp_tools or [])} tools")

  async def shutdown(self) -> None:
    """Disconnect from MCP servers.

    Idempotent - safe to call multiple times.
    """
    if not self._initialized:
      return

    self._initialized = False

    # Only disconnect if we created the client
    if self._client and self._provided_client is None:
      await self._client.disconnect()
      self._client = None

    self._mcp_tools = None
    self._tool_server_map.clear()
    self._dependencies["_mcp_toolkit_client"] = None

    log_info("MCPToolkit: Shutdown complete")

  async def refresh_tools(self) -> None:
    """Refresh the list of available tools from all servers.

    Call this if you know tools have been added/removed on servers.
    """
    if not self._initialized or not self._client:
      return
    await self._discover_tools()

  async def _discover_tools(self) -> None:
    """Discover and cache tools from all connected servers."""

    if not self._client:
      return

    self._mcp_tools = []
    self._tool_server_map.clear()

    # Get tools from all servers
    all_tools = await self._client.list_all_tools()

    for server_name, server_tools in all_tools.items():
      for mcp_tool in server_tools:
        # Generate tool name
        if self._include_server_prefix:
          tool_name = f"{self._tool_name_prefix}{server_name}_{mcp_tool.name}"
        else:
          tool_name = f"{self._tool_name_prefix}{mcp_tool.name}"

        # Track server mapping
        self._tool_server_map[tool_name] = server_name

        # Create Function from MCP tool
        function = self._create_function(
          tool_name=tool_name,
          server_name=server_name,
          mcp_tool=mcp_tool,
        )
        self._mcp_tools.append(function)

    log_debug(f"MCPToolkit: Discovered {len(self._mcp_tools)} tools")

  def _create_function(
    self,
    tool_name: str,
    server_name: str,
    mcp_tool: MCPToolDefinition,
  ) -> "Function":
    """Create a Function object from an MCP tool definition.

    Args:
        tool_name: Generated tool name (with prefixes).
        server_name: Server that provides this tool.
        mcp_tool: MCP tool definition.

    Returns:
        Function object with async entrypoint.
    """
    from definable.tool.function import Function

    # Build parameters from MCP tool input schema
    parameters: Dict[str, Any] = {
      "type": "object",
      "properties": mcp_tool.inputSchema.properties,
      "required": mcp_tool.inputSchema.required or [],
    }

    # Add additionalProperties if specified
    if mcp_tool.inputSchema.additionalProperties is not None:
      parameters["additionalProperties"] = mcp_tool.inputSchema.additionalProperties
    else:
      parameters["additionalProperties"] = False

    # Create the entrypoint function
    # We need to capture server_name and original tool name in closure
    original_tool_name = mcp_tool.name

    async def mcp_tool_entrypoint(
      dependencies: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
    ) -> str:
      """Execute the MCP tool.

      Args:
          dependencies: Injected dependencies containing MCP client.
          **kwargs: Tool arguments.

      Returns:
          Tool result as string.
      """
      client = (dependencies or {}).get("_mcp_toolkit_client")
      if client is None:
        return f"Error: MCP client not available for tool '{tool_name}'"

      try:
        result = await client.call_tool(
          server_name=server_name,
          tool_name=original_tool_name,
          arguments=kwargs,
        )

        # Format result content as string
        if result.isError:
          error_content = []
          for content in result.content:
            if hasattr(content, "text"):
              error_content.append(content.text)
          return f"Error: {' '.join(error_content)}"

        # Combine content items
        output_parts = []
        for content in result.content:
          if hasattr(content, "text"):
            output_parts.append(content.text)
          elif hasattr(content, "data"):
            # Image or binary content
            output_parts.append(f"[{content.type}: {content.mimeType}]")
          elif hasattr(content, "resource"):
            # Embedded resource
            output_parts.append(f"[resource: {content.resource.uri}]")

        return "\n".join(output_parts) if output_parts else "Tool executed successfully (no output)"

      except Exception as e:
        log_error(f"MCPToolkit: Tool '{tool_name}' failed: {e}")
        return f"Error executing tool '{tool_name}': {e}"

    # Create Function with skip_entrypoint_processing=True
    # since we're providing the schema directly from MCP
    function = Function(
      name=tool_name,
      description=mcp_tool.description or f"MCP tool from {server_name}",
      parameters=parameters,
      entrypoint=mcp_tool_entrypoint,
      skip_entrypoint_processing=True,
      requires_confirmation=self._require_confirmation,
    )

    return function

  def get_tool_server(self, tool_name: str) -> Optional[str]:
    """Get the server name for a tool.

    Args:
        tool_name: Full tool name (with prefixes).

    Returns:
        Server name if found, None otherwise.
    """
    return self._tool_server_map.get(tool_name)

  # =========================================================================
  # Sync Helpers
  # =========================================================================

  def initialize_sync(self) -> None:
    """Synchronous wrapper for initialize().

    Creates a new event loop if needed.
    """
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      # We're in an async context - can't use asyncio.run
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, self.initialize())
        future.result()
    else:
      asyncio.run(self.initialize())

  def shutdown_sync(self) -> None:
    """Synchronous wrapper for shutdown().

    Creates a new event loop if needed.
    """
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, self.shutdown())
        future.result()
    else:
      asyncio.run(self.shutdown())

  # =========================================================================
  # Context Manager
  # =========================================================================

  async def __aenter__(self) -> "MCPToolkit":
    """Async context manager entry."""
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Async context manager exit."""
    await self.shutdown()

  def __enter__(self) -> "MCPToolkit":
    """Sync context manager entry (convenience wrapper)."""
    self.initialize_sync()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Sync context manager exit."""
    self.shutdown_sync()

  def __repr__(self) -> str:
    """String representation."""
    tool_count = len(self._mcp_tools) if self._mcp_tools else 0
    server_count = len(self.config.servers)
    status = "initialized" if self._initialized else "not initialized"
    return f"MCPToolkit(servers={server_count}, tools={tool_count}, {status})"
