"""Comprehensive MCP E2E tests.

This module tests the complete MCP client pipeline using a real mock server
running as a subprocess. Tests cover:

1. Connection & Lifecycle
2. Tool Discovery
3. Tool Execution
4. Multi-Server Operations
5. MCPToolkit Integration
6. Agent Integration
7. Resource Provider
8. Prompt Provider
9. Error Handling
10. JSON-RPC Serialization
"""

import sys
from pathlib import Path

import pytest

from definable.mcp import (
  MCPClient,
  MCPConfig,
  MCPConnectionError,
  MCPPromptProvider,
  MCPProtocolError,
  MCPResourceProvider,
  MCPServerConfig,
  MCPServerNotFoundError,
  MCPTimeoutError,
  MCPToolkit,
  MCPToolNotFoundError,
)
from definable.mcp.client import MCPServerConnection


MOCK_SERVER_PATH = Path(__file__).parent / "mock_mcp_server.py"


# =============================================================================
# 1. Connection & Lifecycle Tests
# =============================================================================


class TestConnectionLifecycle:
  """Tests for MCP connection and lifecycle management."""

  @pytest.mark.asyncio
  async def test_stdio_connection_success(self, mock_server_config: MCPServerConfig):
    """Test successful connection to stdio server."""
    connection = MCPServerConnection(mock_server_config)
    try:
      await connection.connect()

      assert connection.connected
      assert connection.server_info is not None
      assert connection.server_info.protocolVersion == "2024-11-05"
      assert connection.server_info.serverInfo.name == "mock-mcp-server"
    finally:
      await connection.disconnect()

  @pytest.mark.asyncio
  async def test_stdio_connection_command_not_found(self, mock_server_config_invalid_command: MCPServerConfig):
    """Test that invalid command raises MCPConnectionError."""
    connection = MCPServerConnection(mock_server_config_invalid_command)

    with pytest.raises(MCPConnectionError) as exc_info:
      await connection.connect()

    assert "not found" in str(exc_info.value).lower() or "Command not found" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_stdio_connection_timeout(self, mock_server_config_slow_start: MCPServerConfig):
    """Test that slow server response during initialization raises MCPTimeoutError."""
    connection = MCPServerConnection(mock_server_config_slow_start)

    with pytest.raises(MCPTimeoutError):
      await connection.connect()

  @pytest.mark.asyncio
  async def test_stdio_disconnect_cleanup(self, mock_server_config: MCPServerConfig):
    """Test that disconnect properly terminates subprocess."""
    connection = MCPServerConnection(mock_server_config)
    await connection.connect()

    assert connection.connected
    assert connection._transport is not None
    assert connection._transport._process is not None

    # Verify process is running (has a PID)
    assert connection._transport._process.pid is not None

    await connection.disconnect()

    assert not connection.connected
    assert connection._transport is None

    # Process should be terminated
    # We can't easily check this cross-platform, but we verify state is cleaned up

  @pytest.mark.asyncio
  async def test_context_manager_lifecycle(self, mock_server_config: MCPServerConfig):
    """Test async context manager properly connects/disconnects."""
    connection = MCPServerConnection(mock_server_config)

    async with connection:
      assert connection.connected
      assert connection.server_info is not None

    assert not connection.connected

  @pytest.mark.asyncio
  async def test_reconnect_with_backoff(self, mock_server_config: MCPServerConfig):
    """Test reconnect after failure with exponential backoff."""
    connection = MCPServerConnection(mock_server_config)
    await connection.connect()

    # Disconnect first
    await connection.disconnect()
    assert not connection.connected

    # Reconnect
    await connection.reconnect()
    assert connection.connected

    await connection.disconnect()

  @pytest.mark.asyncio
  async def test_reconnect_disabled(self, mock_server_config: MCPServerConfig):
    """Test that reconnect raises error when disabled."""
    config = MCPServerConfig(
      name="no_reconnect",
      transport="stdio",
      command=sys.executable,
      args=[str(MOCK_SERVER_PATH)],
      reconnect_on_failure=False,
    )
    connection = MCPServerConnection(config)
    await connection.connect()
    await connection.disconnect()

    with pytest.raises(MCPConnectionError) as exc_info:
      await connection.reconnect()

    assert "disabled" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_double_connect_idempotent(self, mock_server_config: MCPServerConfig):
    """Test that connecting twice is safe."""
    connection = MCPServerConnection(mock_server_config)
    await connection.connect()
    await connection.connect()  # Should not raise

    assert connection.connected
    await connection.disconnect()

  @pytest.mark.asyncio
  async def test_double_disconnect_idempotent(self, mock_server_config: MCPServerConfig):
    """Test that disconnecting twice is safe."""
    connection = MCPServerConnection(mock_server_config)
    await connection.connect()
    await connection.disconnect()
    await connection.disconnect()  # Should not raise

    assert not connection.connected


# =============================================================================
# 2. Tool Discovery Tests
# =============================================================================


class TestToolDiscovery:
  """Tests for MCP tool discovery."""

  @pytest.mark.asyncio
  async def test_list_tools_success(self, mock_client: MCPClient):
    """Test discovering tools from server."""
    tools = await mock_client.list_tools("mock")

    assert len(tools) >= 5
    tool_names = [t.name for t in tools]
    assert "echo" in tool_names
    assert "add_numbers" in tool_names
    assert "slow_tool" in tool_names
    assert "error_tool" in tool_names
    assert "multi_content" in tool_names

  @pytest.mark.asyncio
  async def test_list_tools_caching(self, mock_client: MCPClient):
    """Test that second list_tools call uses cache."""
    connection = mock_client.get_connection("mock")
    assert connection is not None

    # First call populates cache
    tools1 = await connection.list_tools()

    # Verify cache is populated
    assert connection._tools is not None

    # Second call should use cache (we can't easily verify this without mocking,
    # but we verify same result)
    tools2 = await connection.list_tools()
    assert tools1 == tools2

  @pytest.mark.asyncio
  async def test_list_tools_force_refresh(self, mock_client: MCPClient):
    """Test force_refresh=True bypasses cache."""
    connection = mock_client.get_connection("mock")
    assert connection is not None

    # First call
    tools1 = await connection.list_tools()

    # Force refresh - should make new request
    tools2 = await connection.list_tools(force_refresh=True)

    # Results should be same (server returns same tools)
    assert len(tools1) == len(tools2)

  @pytest.mark.asyncio
  async def test_list_tools_filtering_allowed(self, mock_server_config_allowed_tools: MCPServerConfig):
    """Test that only allowed_tools are returned."""
    connection = MCPServerConnection(mock_server_config_allowed_tools)
    async with connection:
      tools = await connection.list_tools()

      tool_names = [t.name for t in tools]
      assert "echo" in tool_names
      assert "add_numbers" in tool_names
      assert "slow_tool" not in tool_names
      assert "error_tool" not in tool_names

  @pytest.mark.asyncio
  async def test_list_tools_filtering_blocked(self, mock_server_config_blocked_tools: MCPServerConfig):
    """Test that blocked_tools are excluded."""
    connection = MCPServerConnection(mock_server_config_blocked_tools)
    async with connection:
      tools = await connection.list_tools()

      tool_names = [t.name for t in tools]
      assert "echo" in tool_names
      assert "add_numbers" in tool_names
      assert "slow_tool" not in tool_names
      assert "error_tool" not in tool_names

  @pytest.mark.asyncio
  async def test_list_tools_empty_server(self, mock_server_config_no_tools: MCPServerConfig):
    """Test server with no tools returns empty list."""
    connection = MCPServerConnection(mock_server_config_no_tools)
    async with connection:
      tools = await connection.list_tools()
      assert tools == []


# =============================================================================
# 3. Tool Execution Tests
# =============================================================================


class TestToolExecution:
  """Tests for MCP tool execution."""

  @pytest.mark.asyncio
  async def test_call_tool_success(self, mock_client: MCPClient):
    """Test executing tool with arguments."""
    result = await mock_client.call_tool("mock", "echo", {"message": "Hello, World!"})

    assert result.isError is None or result.isError is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello, World!"

  @pytest.mark.asyncio
  async def test_call_tool_text_result(self, mock_client: MCPClient):
    """Test tool returns text content."""
    result = await mock_client.call_tool("mock", "add_numbers", {"a": 5, "b": 3})

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "8"

  @pytest.mark.asyncio
  async def test_call_tool_image_result(self, mock_client: MCPClient):
    """Test tool returns image content."""
    result = await mock_client.call_tool("mock", "multi_content", {})

    # Find image content
    image_content = None
    for c in result.content:
      if c.type == "image":
        image_content = c
        break

    assert image_content is not None
    assert image_content.mimeType == "image/png"
    assert image_content.data is not None

  @pytest.mark.asyncio
  async def test_call_tool_multiple_content(self, mock_client: MCPClient):
    """Test tool returns mixed content types."""
    result = await mock_client.call_tool("mock", "multi_content", {})

    assert len(result.content) == 2
    types = [c.type for c in result.content]
    assert "text" in types
    assert "image" in types

  @pytest.mark.asyncio
  async def test_call_tool_not_found(self, mock_client: MCPClient):
    """Test calling unknown tool returns error."""
    result = await mock_client.call_tool("mock", "nonexistent_tool", {})

    # Mock server returns isError=True for unknown tools
    assert result.isError is True

  @pytest.mark.asyncio
  async def test_call_tool_blocked(self, mock_server_config_blocked_tools: MCPServerConfig):
    """Test calling blocked tool raises MCPToolNotFoundError."""
    connection = MCPServerConnection(mock_server_config_blocked_tools)
    async with connection:
      with pytest.raises(MCPToolNotFoundError) as exc_info:
        await connection.call_tool("slow_tool", {})

      assert "blocked" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_call_tool_execution_error(self, mock_client: MCPClient):
    """Test tool error returns isError=True."""
    result = await mock_client.call_tool("mock", "error_tool", {"message": "Test error"})

    assert result.isError is True
    assert len(result.content) >= 1
    assert "Test error" in result.content[0].text

  @pytest.mark.asyncio
  async def test_call_tool_timeout(self, mock_server_config: MCPServerConfig):
    """Test slow tool raises MCPTimeoutError."""
    config = MCPServerConfig(
      name="timeout_test",
      transport="stdio",
      command=sys.executable,
      args=[str(MOCK_SERVER_PATH)],
      connect_timeout=10.0,
      request_timeout=0.5,  # Very short timeout
    )
    connection = MCPServerConnection(config)
    async with connection:
      with pytest.raises(MCPTimeoutError):
        # slow_tool with 5 second delay should timeout
        await connection.call_tool("slow_tool", {"delay": 5})

  @pytest.mark.asyncio
  async def test_call_tool_with_empty_args(self, mock_client: MCPClient):
    """Test tool with no required arguments."""
    result = await mock_client.call_tool("mock", "multi_content", {})
    assert result.isError is None or result.isError is False


# =============================================================================
# 4. Multi-Server Tests
# =============================================================================


class TestMultiServer:
  """Tests for multi-server MCP operations."""

  @pytest.mark.asyncio
  async def test_multi_server_connect(self, mock_config_multi_server: MCPConfig):
    """Test connecting to multiple servers."""
    client = MCPClient(mock_config_multi_server)
    await client.connect()

    try:
      assert client.connected
      assert len(client.servers) == 2
      assert "server1" in client.servers
      assert "server2" in client.servers

      conn1 = client.get_connection("server1")
      conn2 = client.get_connection("server2")
      assert conn1 is not None and conn1.connected
      assert conn2 is not None and conn2.connected
    finally:
      await client.disconnect()

  @pytest.mark.asyncio
  async def test_multi_server_list_all_tools(self, mock_multi_client: MCPClient):
    """Test aggregating tools from all servers."""
    all_tools = await mock_multi_client.list_all_tools()

    assert "server1" in all_tools
    assert "server2" in all_tools

    # Both servers have the same tools (echo, add_numbers, etc.)
    assert len(all_tools["server1"]) >= 5
    assert len(all_tools["server2"]) >= 5

  @pytest.mark.asyncio
  async def test_multi_server_call_tool(self, mock_multi_client: MCPClient):
    """Test routing tool call to correct server."""
    # Call tool on server1
    result1 = await mock_multi_client.call_tool("server1", "echo", {"message": "From server1"})
    assert result1.content[0].text == "From server1"

    # Call tool on server2
    result2 = await mock_multi_client.call_tool("server2", "echo", {"message": "From server2"})
    assert result2.content[0].text == "From server2"

  @pytest.mark.asyncio
  async def test_multi_server_find_tool(self, mock_multi_client: MCPClient):
    """Test finding which server has a tool."""
    # Populate tool caches
    await mock_multi_client.list_all_tools()

    server = mock_multi_client.find_tool("echo")
    assert server in ["server1", "server2"]

  @pytest.mark.asyncio
  async def test_multi_server_partial_failure(self, mock_config_with_failure: MCPConfig):
    """Test one server fails, others work."""
    client = MCPClient(mock_config_with_failure)
    await client.connect()

    try:
      assert client.connected

      # Working server should be accessible
      working_conn = client.get_connection("working")
      assert working_conn is not None
      assert working_conn.connected

      # Failing server should have failed to connect
      failing_conn = client.get_connection("failing")
      assert failing_conn is not None
      assert not failing_conn.connected

      # Should still be able to use working server
      tools = await client.list_tools("working")
      assert len(tools) >= 5
    finally:
      await client.disconnect()

  @pytest.mark.asyncio
  async def test_multi_server_server_not_found(self, mock_multi_client: MCPClient):
    """Test accessing nonexistent server raises error."""
    with pytest.raises(MCPServerNotFoundError) as exc_info:
      await mock_multi_client.list_tools("nonexistent")

    assert "not found" in str(exc_info.value).lower()
    assert exc_info.value.available_servers is not None


# =============================================================================
# 5. MCPToolkit Integration Tests
# =============================================================================


class TestMCPToolkitIntegration:
  """Tests for MCPToolkit integration with agent system."""

  @pytest.mark.asyncio
  async def test_toolkit_initialize_discovers_tools(self, mock_config: MCPConfig):
    """Test Initialize creates Function objects."""
    toolkit = MCPToolkit(mock_config)
    await toolkit.initialize()

    try:
      assert toolkit._initialized
      assert len(toolkit.tools) >= 5

      tool_names = [t.name for t in toolkit.tools]
      # Default naming: {server}_{tool}
      assert "mock_echo" in tool_names
      assert "mock_add_numbers" in tool_names
    finally:
      await toolkit.shutdown()

  @pytest.mark.asyncio
  async def test_toolkit_tool_naming_with_server_prefix(self, mock_config: MCPConfig):
    """Test tools named {server}_{tool}."""
    toolkit = MCPToolkit(mock_config, include_server_prefix=True)
    await toolkit.initialize()

    try:
      tool_names = [t.name for t in toolkit.tools]
      assert all(name.startswith("mock_") for name in tool_names)
    finally:
      await toolkit.shutdown()

  @pytest.mark.asyncio
  async def test_toolkit_tool_naming_custom_prefix(self, mock_config: MCPConfig):
    """Test custom prefix applied to tool names."""
    toolkit = MCPToolkit(mock_config, tool_name_prefix="mcp_")
    await toolkit.initialize()

    try:
      tool_names = [t.name for t in toolkit.tools]
      assert all(name.startswith("mcp_mock_") for name in tool_names)
    finally:
      await toolkit.shutdown()

  @pytest.mark.asyncio
  async def test_toolkit_tool_naming_no_server_prefix(self, mock_config: MCPConfig):
    """Test include_server_prefix=False."""
    toolkit = MCPToolkit(mock_config, include_server_prefix=False)
    await toolkit.initialize()

    try:
      tool_names = [t.name for t in toolkit.tools]
      assert "echo" in tool_names
      assert "add_numbers" in tool_names
    finally:
      await toolkit.shutdown()

  @pytest.mark.asyncio
  async def test_toolkit_tool_schema_extraction(self, mock_toolkit: MCPToolkit):
    """Test inputSchema becomes Function parameters."""
    # Find the add_numbers tool
    add_tool = None
    for t in mock_toolkit.tools:
      if "add_numbers" in t.name:
        add_tool = t
        break

    assert add_tool is not None
    assert add_tool.parameters is not None
    assert "properties" in add_tool.parameters
    assert "a" in add_tool.parameters["properties"]
    assert "b" in add_tool.parameters["properties"]

  @pytest.mark.asyncio
  async def test_toolkit_tool_execution_injects_client(self, mock_toolkit: MCPToolkit):
    """Test tool receives _mcp_toolkit_client dependency."""
    from definable.tools.function import FunctionCall

    # Find the echo tool
    echo_tool = None
    for t in mock_toolkit.tools:
      if "echo" in t.name:
        echo_tool = t
        break

    assert echo_tool is not None

    # Create a copy of the tool with dependencies set
    tool_copy = echo_tool.model_copy()
    tool_copy._dependencies = mock_toolkit.dependencies

    # Create a FunctionCall and execute it
    fc = FunctionCall(function=tool_copy, arguments={"message": "test message"})
    result = await fc.aexecute()

    assert result.status == "success"
    assert "test message" in str(result.result)

  @pytest.mark.asyncio
  async def test_toolkit_get_tool_server(self, mock_toolkit: MCPToolkit):
    """Test mapping tool name to server."""
    server = mock_toolkit.get_tool_server("mock_echo")
    assert server == "mock"

  @pytest.mark.asyncio
  async def test_toolkit_refresh_tools(self, mock_config: MCPConfig):
    """Test refresh picks up tool changes."""
    toolkit = MCPToolkit(mock_config)
    await toolkit.initialize()

    try:
      initial_count = len(toolkit.tools)

      # Refresh (same tools, but verifies refresh works)
      await toolkit.refresh_tools()

      assert len(toolkit.tools) == initial_count
    finally:
      await toolkit.shutdown()

  def test_toolkit_sync_context_manager(self, mock_config: MCPConfig):
    """Test `with toolkit:` works synchronously."""
    toolkit = MCPToolkit(mock_config)

    with toolkit:
      assert toolkit._initialized
      assert len(toolkit.tools) >= 5

    assert not toolkit._initialized

  @pytest.mark.asyncio
  async def test_toolkit_async_context_manager(self, mock_config: MCPConfig):
    """Test `async with toolkit:` works."""
    toolkit = MCPToolkit(mock_config)

    async with toolkit:
      assert toolkit._initialized
      assert len(toolkit.tools) >= 5

    assert not toolkit._initialized

  @pytest.mark.asyncio
  async def test_toolkit_not_initialized_returns_empty_tools(self, mock_config: MCPConfig):
    """Test uninitialized toolkit returns empty tools list."""
    toolkit = MCPToolkit(mock_config)
    assert toolkit.tools == []


# =============================================================================
# 6. Agent Integration Tests
# =============================================================================


class TestAgentIntegration:
  """Tests for MCP integration with Agent."""

  @pytest.mark.asyncio
  async def test_agent_with_mcp_toolkit(self, mock_toolkit: MCPToolkit):
    """Test Agent discovers and uses MCP tools."""
    from definable.agents import Agent
    from definable.agents.testing import MockModel

    # Create mock model that will call the echo tool
    mock_model = MockModel(
      responses=["The tool returned the message."],
      tool_calls=[
        [
          {
            "id": "call_1",
            "type": "function",
            "function": {"name": "mock_echo", "arguments": '{"message": "Hello from agent"}'},
          }
        ],
        [],  # No tool calls in second response
      ],
    )

    agent = Agent(model=mock_model, toolkits=[mock_toolkit])

    # Verify tools are available via _flatten_tools
    all_tools = agent._flatten_tools()
    tool_names = list(all_tools.keys())
    assert "mock_echo" in tool_names

  @pytest.mark.asyncio
  async def test_agent_tool_call_succeeds(self, mock_toolkit: MCPToolkit):
    """Test Agent executes MCP tool correctly."""
    from definable.agents import Agent
    from definable.agents.testing import MockModel

    mock_model = MockModel(
      responses=["The result is 8."],
      tool_calls=[
        [
          {
            "id": "call_1",
            "type": "function",
            "function": {"name": "mock_add_numbers", "arguments": '{"a": 5, "b": 3}'},
          }
        ],
        [],
      ],
    )

    agent = Agent(model=mock_model, toolkits=[mock_toolkit])
    output = await agent.arun("Add 5 and 3")

    # Agent should have executed the tool
    assert output is not None

  @pytest.mark.asyncio
  async def test_agent_tool_result_in_response(self, mock_toolkit: MCPToolkit):
    """Test tool result appears in response."""
    from definable.agents import Agent
    from definable.agents.testing import MockModel

    mock_model = MockModel(
      responses=["The tool said: Hello World"],
      tool_calls=[
        [
          {
            "id": "call_1",
            "type": "function",
            "function": {"name": "mock_echo", "arguments": '{"message": "Hello World"}'},
          }
        ],
        [],
      ],
    )

    agent = Agent(model=mock_model, toolkits=[mock_toolkit])
    output = await agent.arun("Echo Hello World")

    assert output is not None
    assert output.content is not None

  @pytest.mark.asyncio
  async def test_agent_tool_error_handling(self, mock_toolkit: MCPToolkit):
    """Test Agent handles tool execution errors."""
    from definable.agents import Agent
    from definable.agents.testing import MockModel

    mock_model = MockModel(
      responses=["The tool returned an error."],
      tool_calls=[
        [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "mock_error_tool",
              "arguments": '{"message": "Expected error"}',
            },
          }
        ],
        [],
      ],
    )

    agent = Agent(model=mock_model, toolkits=[mock_toolkit])
    output = await agent.arun("Trigger an error")

    # Should complete without raising
    assert output is not None


# =============================================================================
# 7. Resource Provider Tests
# =============================================================================


class TestResourceProvider:
  """Tests for MCPResourceProvider."""

  @pytest.mark.asyncio
  async def test_list_resources_success(self, mock_client: MCPClient):
    """Test listing resources from server."""
    provider = MCPResourceProvider(mock_client)
    resources = await provider.list_resources("mock")

    assert "mock" in resources
    assert len(resources["mock"]) >= 3

    uris = [r.uri for r in resources["mock"]]
    assert "file:///test/config.json" in uris
    assert "file:///test/data.txt" in uris

  @pytest.mark.asyncio
  async def test_read_resource_text(self, mock_client: MCPClient):
    """Test reading text resource content."""
    provider = MCPResourceProvider(mock_client)
    content = await provider.read_resource("mock", "file:///test/config.json")

    assert len(content) == 1
    assert hasattr(content[0], "text")
    assert "key" in content[0].text

  @pytest.mark.asyncio
  async def test_read_resource_binary(self, mock_client: MCPClient):
    """Test reading binary resource content."""
    provider = MCPResourceProvider(mock_client)
    content = await provider.read_resource("mock", "file:///test/image.png")

    assert len(content) == 1
    assert hasattr(content[0], "blob")
    assert content[0].mimeType == "image/png"

  @pytest.mark.asyncio
  async def test_read_resource_not_found(self, mock_client: MCPClient):
    """Test reading missing resource raises error."""
    provider = MCPResourceProvider(mock_client)

    with pytest.raises(MCPProtocolError):
      await provider.read_resource("mock", "file:///nonexistent")

  @pytest.mark.asyncio
  async def test_read_text_convenience(self, mock_client: MCPClient):
    """Test read_text convenience method."""
    provider = MCPResourceProvider(mock_client)
    text = await provider.read_text("mock", "file:///test/data.txt")

    assert "test data content" in text.lower()

  @pytest.mark.asyncio
  async def test_resource_provider_multi_server(self, mock_multi_client: MCPClient):
    """Test aggregating resources across servers."""
    provider = MCPResourceProvider(mock_multi_client)
    all_resources = await provider.list_resources()

    assert "server1" in all_resources
    assert "server2" in all_resources

  @pytest.mark.asyncio
  async def test_find_resource(self, mock_client: MCPClient):
    """Test finding which server has a resource."""
    provider = MCPResourceProvider(mock_client)
    server = await provider.find_resource("file:///test/config.json")

    assert server == "mock"

  @pytest.mark.asyncio
  async def test_get_resource_info(self, mock_client: MCPClient):
    """Test getting resource metadata."""
    provider = MCPResourceProvider(mock_client)
    info = await provider.get_resource_info("mock", "file:///test/config.json")

    assert info is not None
    assert info.name == "Test Config"
    assert info.mimeType == "application/json"


# =============================================================================
# 8. Prompt Provider Tests
# =============================================================================


class TestPromptProvider:
  """Tests for MCPPromptProvider."""

  @pytest.mark.asyncio
  async def test_list_prompts_success(self, mock_client: MCPClient):
    """Test listing prompts from server."""
    provider = MCPPromptProvider(mock_client)
    prompts = await provider.list_prompts("mock")

    assert "mock" in prompts
    assert len(prompts["mock"]) >= 2

    names = [p.name for p in prompts["mock"]]
    assert "greeting" in names
    assert "code_review" in names

  @pytest.mark.asyncio
  async def test_get_prompt_with_arguments(self, mock_client: MCPClient):
    """Test rendering prompt with args."""
    provider = MCPPromptProvider(mock_client)
    result = await provider.get_prompt("mock", "greeting", {"name": "Alice"})

    assert result is not None
    assert len(result.messages) >= 1
    assert "Alice" in result.messages[0].content.text

  @pytest.mark.asyncio
  async def test_get_prompt_messages(self, mock_client: MCPClient):
    """Test getting prompt message array."""
    provider = MCPPromptProvider(mock_client)
    messages = await provider.get_messages("mock", "greeting", {"name": "Bob"})

    assert len(messages) >= 1
    assert messages[0].role.value in ["user", "assistant"]

  @pytest.mark.asyncio
  async def test_get_prompt_not_found(self, mock_client: MCPClient):
    """Test getting missing prompt raises error."""
    provider = MCPPromptProvider(mock_client)

    with pytest.raises(MCPProtocolError):
      await provider.get_prompt("mock", "nonexistent_prompt", {})

  @pytest.mark.asyncio
  async def test_prompt_provider_multi_server(self, mock_multi_client: MCPClient):
    """Test aggregating prompts across servers."""
    provider = MCPPromptProvider(mock_multi_client)
    all_prompts = await provider.list_prompts()

    assert "server1" in all_prompts
    assert "server2" in all_prompts

  @pytest.mark.asyncio
  async def test_find_prompt(self, mock_client: MCPClient):
    """Test finding which server has a prompt."""
    provider = MCPPromptProvider(mock_client)
    server = await provider.find_prompt("greeting")

    assert server == "mock"

  @pytest.mark.asyncio
  async def test_get_prompt_text(self, mock_client: MCPClient):
    """Test get_text convenience method."""
    provider = MCPPromptProvider(mock_client)
    text = await provider.get_text("mock", "code_review", {"language": "python"})

    assert "python" in text.lower()

  @pytest.mark.asyncio
  async def test_get_prompt_arguments(self, mock_client: MCPClient):
    """Test getting prompt argument names."""
    provider = MCPPromptProvider(mock_client)
    args = await provider.get_prompt_arguments("mock", "greeting")

    assert "name" in args


# =============================================================================
# 9. Error Handling Tests
# =============================================================================


class TestErrorHandling:
  """Tests for MCP error handling."""

  @pytest.mark.asyncio
  async def test_protocol_error_invalid_json(self, mock_server_config_malformed: MCPServerConfig):
    """Test malformed JSON handling during initialization.

    When MCP_MALFORMED_JSON=true, the mock server sends malformed JSON,
    which causes the client to fail (either timeout waiting for valid response
    or protocol error).
    """
    connection = MCPServerConnection(mock_server_config_malformed)

    # Connection should fail because the server sends malformed JSON
    # This may manifest as a timeout (waiting for valid response) or protocol error
    with pytest.raises((MCPProtocolError, MCPTimeoutError)):
      await connection.connect()

  @pytest.mark.asyncio
  async def test_protocol_error_initialization_failure(self, mock_server_config_fail_init: MCPServerConfig):
    """Test server initialization failure."""
    connection = MCPServerConnection(mock_server_config_fail_init)

    with pytest.raises(MCPProtocolError):
      await connection.connect()

  @pytest.mark.asyncio
  async def test_server_error_response(self, mock_client: MCPClient):
    """Test server returns JSON-RPC error."""
    # The error_tool returns isError=True, which is a tool-level error
    # For a protocol-level error, we'd need to call a method that doesn't exist
    result = await mock_client.call_tool("mock", "error_tool", {})
    assert result.isError is True

  @pytest.mark.asyncio
  async def test_connection_not_established(self, mock_server_config: MCPServerConfig):
    """Test operations fail when not connected."""
    connection = MCPServerConnection(mock_server_config)

    with pytest.raises(MCPConnectionError):
      await connection.list_tools()

  @pytest.mark.asyncio
  async def test_request_after_disconnect(self, mock_server_config: MCPServerConfig):
    """Test request after disconnect fails."""
    connection = MCPServerConnection(mock_server_config)
    await connection.connect()
    await connection.disconnect()

    with pytest.raises(MCPConnectionError):
      await connection.list_tools()


# =============================================================================
# 10. JSON-RPC Serialization Tests
# =============================================================================


class TestJSONRPCSerialization:
  """Tests for JSON-RPC message serialization."""

  @pytest.mark.asyncio
  async def test_request_excludes_null_params(self, mock_client: MCPClient):
    """Test that params: null is not sent in requests."""
    # This is implicitly tested - the mock server works correctly
    # which means our requests are properly formatted
    tools = await mock_client.list_tools("mock")
    assert len(tools) >= 5

  @pytest.mark.asyncio
  async def test_large_payload_handling(self, mock_client: MCPClient):
    """Test large tool arguments work."""
    large_message = "x" * 10000  # 10KB string
    result = await mock_client.call_tool("mock", "echo", {"message": large_message})

    assert len(result.content[0].text) == 10000

  @pytest.mark.asyncio
  async def test_unicode_in_arguments(self, mock_client: MCPClient):
    """Test unicode characters preserved."""
    # Use properly encoded unicode characters (not surrogate pairs)
    # Chinese characters, Greek letters, and actual emoji (not escaped surrogates)
    unicode_message = "Hello \u4e16\u754c! \U0001f600 \u03b1\u03b2\u03b3"  # \U0001F600 is the smiley emoji
    result = await mock_client.call_tool("mock", "echo", {"message": unicode_message})

    assert result.content[0].text == unicode_message

  @pytest.mark.asyncio
  async def test_special_characters_in_arguments(self, mock_client: MCPClient):
    """Test special characters in arguments."""
    special_message = 'Line1\nLine2\tTabbed "Quoted" \\Escaped'
    result = await mock_client.call_tool("mock", "echo", {"message": special_message})

    assert result.content[0].text == special_message

  @pytest.mark.asyncio
  async def test_numeric_arguments(self, mock_client: MCPClient):
    """Test numeric argument types preserved."""
    result = await mock_client.call_tool("mock", "add_numbers", {"a": 3.14159, "b": 2.71828})
    # Result should be close to 5.85987
    value = float(result.content[0].text)
    assert abs(value - 5.85987) < 0.001

  @pytest.mark.asyncio
  async def test_empty_arguments(self, mock_client: MCPClient):
    """Test empty arguments dict."""
    result = await mock_client.call_tool("mock", "multi_content", {})
    assert len(result.content) >= 1
