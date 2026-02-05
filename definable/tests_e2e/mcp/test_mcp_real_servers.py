"""Tests with real MCP servers.

These tests require actual MCP servers to be installed and available.
They are skipped automatically if the required servers are not found.

To run these tests, install the required servers:
    npm install -g @modelcontextprotocol/server-filesystem

Then run:
    pytest tests_e2e/mcp/test_mcp_real_servers.py -v
"""

import shutil
from pathlib import Path

import pytest
from definable.mcp import (
  MCPClient,
  MCPConfig,
  MCPConnectionError,
  MCPServerConfig,
  MCPToolkit,
)
from definable.mcp.client import MCPServerConnection

# =============================================================================
# Skip Conditions
# =============================================================================


def npx_available() -> bool:
  """Check if npx is available."""
  return shutil.which("npx") is not None


def can_run_filesystem_server() -> bool:
  """Check if the filesystem MCP server can be started."""
  if not npx_available():
    return False
  # We don't actually test if the package is installed
  # as npx -y will auto-install it
  return True


requires_npx = pytest.mark.skipif(
  not npx_available(),
  reason="npx not available (install Node.js)",
)

requires_filesystem_server = pytest.mark.skipif(
  not can_run_filesystem_server(),
  reason="Filesystem MCP server not available (requires npx)",
)


# =============================================================================
# Fixtures for Real Servers
# =============================================================================


def get_tmp_dir() -> str:
  """Get the real tmp directory path (handles macOS symlink)."""
  import os

  # On macOS, /tmp is a symlink to /private/tmp
  # The MCP filesystem server needs the real path
  tmp = "/tmp"
  real_tmp = os.path.realpath(tmp)
  return real_tmp


@pytest.fixture
def filesystem_server_config() -> MCPServerConfig:
  """Configuration for the official filesystem MCP server."""
  tmp_dir = get_tmp_dir()
  return MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
    connect_timeout=60.0,  # npx may need time to download
    request_timeout=30.0,
  )


@pytest.fixture
def filesystem_config(filesystem_server_config: MCPServerConfig) -> MCPConfig:
  """MCPConfig with filesystem server."""
  return MCPConfig(servers=[filesystem_server_config])


# =============================================================================
# Real Server Tests
# =============================================================================


@requires_filesystem_server
class TestFilesystemServer:
  """Tests using the official filesystem MCP server."""

  @pytest.mark.asyncio
  async def test_connect_to_filesystem_server(self, filesystem_server_config: MCPServerConfig):
    """Test connecting to the real filesystem server."""
    connection = MCPServerConnection(filesystem_server_config)

    try:
      await connection.connect()

      assert connection.connected
      assert connection.server_info is not None
      # The server should identify itself
      assert connection.server_info.serverInfo.name is not None
    finally:
      await connection.disconnect()

  @pytest.mark.asyncio
  async def test_list_tools_from_filesystem_server(self, filesystem_server_config: MCPServerConfig):
    """Test listing tools from the filesystem server."""
    connection = MCPServerConnection(filesystem_server_config)

    async with connection:
      tools = await connection.list_tools()

      # Filesystem server should have some tools
      assert len(tools) > 0

      # Check that tools have proper structure
      for tool in tools:
        assert tool.name is not None
        assert tool.inputSchema is not None

      # Print discovered tools for debugging
      tool_names = [t.name for t in tools]
      print(f"\nDiscovered tools from filesystem server: {tool_names}")

  @pytest.mark.asyncio
  async def test_toolkit_with_filesystem_server(self, filesystem_config: MCPConfig):
    """Test MCPToolkit with filesystem server."""
    toolkit = MCPToolkit(filesystem_config)

    async with toolkit:
      # Should have discovered tools
      assert len(toolkit.tools) > 0

      # Tools should be properly named with server prefix
      for tool in toolkit.tools:
        assert tool.name.startswith("filesystem_")

      print(f"\nToolkit tools: {[t.name for t in toolkit.tools]}")

  @pytest.mark.asyncio
  async def test_read_directory_tool(self, filesystem_config: MCPConfig):
    """Test calling a tool to read tmp directory."""
    tmp_dir = get_tmp_dir()
    client = MCPClient(filesystem_config)

    async with client:
      # First list tools to see what's available
      tools = await client.list_tools("filesystem")
      tool_names = [t.name for t in tools]
      print(f"\nAvailable tools: {tool_names}")

      # Try to find a directory listing tool
      # Different versions may have different tool names
      list_tool = None
      for name in ["list_directory", "read_directory", "list_dir", "ls"]:
        if name in tool_names:
          list_tool = name
          break

      if list_tool:
        result = await client.call_tool("filesystem", list_tool, {"path": tmp_dir})

        # Should return content
        assert len(result.content) > 0
        assert result.isError is None or result.isError is False

        print(f"\nDirectory listing result: {result.content[0]}")
      else:
        # If no directory listing tool, try read_file on a known path
        pytest.skip(f"No directory listing tool found. Available: {tool_names}")

  @pytest.mark.asyncio
  async def test_create_and_read_file(self, filesystem_config: MCPConfig):
    """Test creating and reading a file via MCP tools."""
    import uuid

    tmp_dir = get_tmp_dir()
    client = MCPClient(filesystem_config)

    async with client:
      tools = await client.list_tools("filesystem")
      tool_names = [t.name for t in tools]

      # Find write and read tools
      write_tool = None
      read_tool = None

      for name in ["write_file", "create_file", "write"]:
        if name in tool_names:
          write_tool = name
          break

      for name in ["read_file", "read", "get_file_contents", "read_text_file"]:
        if name in tool_names:
          read_tool = name
          break

      if not write_tool or not read_tool:
        pytest.skip(f"Missing write ({write_tool}) or read ({read_tool}) tool. Available: {tool_names}")

      # Create a unique test file in the allowed directory
      test_content = f"Test content from MCP E2E test: {uuid.uuid4()}"
      test_file = f"{tmp_dir}/mcp_test_{uuid.uuid4().hex[:8]}.txt"

      try:
        # Write file
        write_result = await client.call_tool(
          "filesystem",
          write_tool,
          {"path": test_file, "content": test_content},
        )

        # Verify write succeeded
        assert write_result.isError is None or write_result.isError is False

        # Read file back
        read_result = await client.call_tool("filesystem", read_tool, {"path": test_file})

        # Verify read succeeded and content matches
        assert read_result.isError is None or read_result.isError is False
        assert len(read_result.content) > 0

        # Content should contain our test string
        read_content = read_result.content[0].text
        assert test_content in read_content

        print(f"\nSuccessfully wrote and read: {test_file}")

      finally:
        # Clean up test file
        import contextlib

        with contextlib.suppress(Exception):
          Path(test_file).unlink(missing_ok=True)


# =============================================================================
# Generic Server Tests (work with any MCP server)
# =============================================================================


class TestGenericMCPServer:
  """Generic tests that work with any MCP server configuration.

  These tests take server config as a parameter and can be reused
  with different servers.
  """

  @staticmethod
  async def verify_connection(config: MCPServerConfig) -> bool:
    """Verify a server can be connected to."""
    connection = MCPServerConnection(config)
    try:
      await connection.connect()
      connected = connection.connected
      await connection.disconnect()
      return connected
    except MCPConnectionError:
      return False

  @staticmethod
  async def verify_tools_available(config: MCPServerConfig) -> list:
    """Verify tools can be listed from server."""
    connection = MCPServerConnection(config)
    try:
      async with connection:
        tools = await connection.list_tools()
        return [t.name for t in tools]
    except Exception:
      return []

  @staticmethod
  async def run_tool_test(config: MCPServerConfig, tool_name: str, arguments: dict) -> tuple:
    """Run a tool and return (success, result_or_error)."""
    connection = MCPServerConnection(config)
    try:
      async with connection:
        result = await connection.call_tool(tool_name, arguments)
        if result.isError:
          return (False, result.content[0].text if result.content else "Error")
        return (True, result.content[0].text if result.content else "")
    except Exception as e:
      return (False, str(e))


# =============================================================================
# Parameterized Tests for Multiple Servers
# =============================================================================


def get_available_server_configs() -> list:
  """Get list of available server configurations for parameterized tests."""
  configs = []

  if can_run_filesystem_server():
    tmp_dir = get_tmp_dir()
    configs.append(
      pytest.param(
        MCPServerConfig(
          name="filesystem",
          command="npx",
          args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
          connect_timeout=60.0,
        ),
        id="filesystem",
      )
    )

  # Add more servers here as they become available
  # if can_run_memory_server():
  #     configs.append(...)

  return configs


@pytest.mark.skipif(
  len(get_available_server_configs()) == 0,
  reason="No MCP servers available for testing",
)
class TestMultipleRealServers:
  """Tests that run against all available real MCP servers."""

  @pytest.mark.asyncio
  @pytest.mark.parametrize("server_config", get_available_server_configs())
  async def test_connection(self, server_config: MCPServerConfig):
    """Test that server can be connected to."""
    result = await TestGenericMCPServer.verify_connection(server_config)
    assert result, f"Failed to connect to {server_config.name}"

  @pytest.mark.asyncio
  @pytest.mark.parametrize("server_config", get_available_server_configs())
  async def test_tools_discoverable(self, server_config: MCPServerConfig):
    """Test that tools can be discovered from server."""
    tools = await TestGenericMCPServer.verify_tools_available(server_config)
    assert len(tools) > 0, f"No tools found on {server_config.name}"
    print(f"\n{server_config.name} tools: {tools}")
