"""MCP E2E test fixtures."""

import sys
from pathlib import Path

import pytest

from definable.mcp import MCPClient, MCPConfig, MCPServerConfig, MCPToolkit


# Path to the mock server script
MOCK_SERVER_PATH = Path(__file__).parent / "mock_mcp_server.py"


@pytest.fixture
def mock_server_config() -> MCPServerConfig:
  """Create a server config pointing to the mock MCP server."""
  return MCPServerConfig(
    name="mock",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_slow_start() -> MCPServerConfig:
  """Create a server config with slow startup (for timeout tests).

  This tests timeout during the initialization handshake, not subprocess creation.
  The MCP_RESPONSE_DELAY env var delays responses, which will timeout the initialize request.
  """
  return MCPServerConfig(
    name="mock_slow",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_RESPONSE_DELAY": "10"},  # 10 second delay on all responses
    connect_timeout=30.0,
    request_timeout=1.0,  # But only wait 1 second for initialize response
  )


@pytest.fixture
def mock_server_config_slow_response() -> MCPServerConfig:
  """Create a server config with slow responses (for timeout tests)."""
  return MCPServerConfig(
    name="mock_slow_response",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_RESPONSE_DELAY": "10"},  # 10 second delay
    connect_timeout=10.0,
    request_timeout=1.0,  # But only wait 1 second
  )


@pytest.fixture
def mock_server_config_fail_init() -> MCPServerConfig:
  """Create a server config that fails initialization."""
  return MCPServerConfig(
    name="mock_fail",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_FAIL_INITIALIZE": "true"},
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_malformed() -> MCPServerConfig:
  """Create a server config that returns malformed JSON."""
  return MCPServerConfig(
    name="mock_malformed",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_MALFORMED_JSON": "true"},
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_no_tools() -> MCPServerConfig:
  """Create a server config with no tools."""
  return MCPServerConfig(
    name="mock_empty",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_TOOLS": "[]"},
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_custom_tools() -> MCPServerConfig:
  """Create a server config with custom tools."""
  import json

  tools = [
    {
      "name": "custom_tool",
      "description": "A custom test tool",
      "inputSchema": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
      },
    }
  ]

  return MCPServerConfig(
    name="mock_custom",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    env={"MCP_TOOLS": json.dumps(tools)},
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_allowed_tools() -> MCPServerConfig:
  """Create a server config with allowed_tools filter."""
  return MCPServerConfig(
    name="mock_filtered",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    allowed_tools=["echo", "add_numbers"],
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_blocked_tools() -> MCPServerConfig:
  """Create a server config with blocked_tools filter."""
  return MCPServerConfig(
    name="mock_blocked",
    transport="stdio",
    command=sys.executable,
    args=[str(MOCK_SERVER_PATH)],
    blocked_tools=["error_tool", "slow_tool"],
    connect_timeout=10.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_server_config_invalid_command() -> MCPServerConfig:
  """Create a server config with an invalid command."""
  return MCPServerConfig(
    name="mock_invalid",
    transport="stdio",
    command="nonexistent_command_12345",
    args=[],
    connect_timeout=5.0,
    request_timeout=30.0,
  )


@pytest.fixture
def mock_config(mock_server_config: MCPServerConfig) -> MCPConfig:
  """Create an MCPConfig with a single mock server."""
  return MCPConfig(servers=[mock_server_config])


@pytest.fixture
def mock_config_multi_server() -> MCPConfig:
  """Create an MCPConfig with multiple mock servers."""
  return MCPConfig(
    servers=[
      MCPServerConfig(
        name="server1",
        transport="stdio",
        command=sys.executable,
        args=[str(MOCK_SERVER_PATH)],
        connect_timeout=10.0,
        request_timeout=30.0,
      ),
      MCPServerConfig(
        name="server2",
        transport="stdio",
        command=sys.executable,
        args=[str(MOCK_SERVER_PATH)],
        connect_timeout=10.0,
        request_timeout=30.0,
      ),
    ]
  )


@pytest.fixture
def mock_config_with_failure() -> MCPConfig:
  """Create an MCPConfig with one working server and one failing server."""
  return MCPConfig(
    servers=[
      MCPServerConfig(
        name="working",
        transport="stdio",
        command=sys.executable,
        args=[str(MOCK_SERVER_PATH)],
        connect_timeout=10.0,
        request_timeout=30.0,
      ),
      MCPServerConfig(
        name="failing",
        transport="stdio",
        command="nonexistent_command_12345",
        args=[],
        connect_timeout=5.0,
        request_timeout=30.0,
      ),
    ]
  )


@pytest.fixture
async def mock_client(mock_config: MCPConfig) -> MCPClient:
  """Create and connect an MCPClient with mock server."""
  client = MCPClient(mock_config)
  await client.connect()
  yield client
  await client.disconnect()


@pytest.fixture
async def mock_toolkit(mock_config: MCPConfig) -> MCPToolkit:
  """Create and initialize an MCPToolkit with mock server."""
  toolkit = MCPToolkit(mock_config)
  await toolkit.initialize()
  yield toolkit
  await toolkit.shutdown()


@pytest.fixture
async def mock_multi_client(mock_config_multi_server: MCPConfig) -> MCPClient:
  """Create and connect an MCPClient with multiple mock servers."""
  client = MCPClient(mock_config_multi_server)
  await client.connect()
  yield client
  await client.disconnect()


@pytest.fixture
async def mock_multi_toolkit(mock_config_multi_server: MCPConfig) -> MCPToolkit:
  """Create and initialize an MCPToolkit with multiple mock servers."""
  toolkit = MCPToolkit(mock_config_multi_server)
  await toolkit.initialize()
  yield toolkit
  await toolkit.shutdown()
