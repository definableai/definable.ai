"""Unit tests for ComposioToolkit."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from definable.composio.toolkit import ComposioToolkit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_composio_sdk():
  """Mock the composio SDK module."""
  mock_session = MagicMock()
  mock_session.mcp.url = "https://mcp.composio.dev/session/abc123"
  mock_session.mcp.headers = {"Authorization": "Bearer test-token"}

  mock_client = MagicMock()
  mock_client.create.return_value = mock_session

  mock_module = MagicMock()
  mock_module.Composio.return_value = mock_client

  return mock_module, mock_client, mock_session


@pytest.fixture
def mock_mcp_toolkit():
  """Mock MCPToolkit that simulates initialization."""
  mock = MagicMock()
  mock.initialize = AsyncMock()
  mock.shutdown = AsyncMock()

  # Simulate tools being available after init
  mock_tool = MagicMock()
  mock_tool.name = "COMPOSIO_SEARCH_TOOLS"
  mock.tools = [mock_tool]

  # Simulate dependencies with MCP client reference
  mock.dependencies = {"_mcp_toolkit_client": MagicMock()}

  return mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
  def test_defaults(self):
    tk = ComposioToolkit()
    assert tk.user_id == "default"
    assert tk._api_key is None
    assert tk._toolkits is None
    assert tk._initialized is False

  def test_custom_params(self):
    tk = ComposioToolkit(
      user_id="u1",
      api_key="key-123",
      toolkits=["gmail", "slack"],
      connect_timeout=10.0,
      request_timeout=60.0,
    )
    assert tk.user_id == "u1"
    assert tk._api_key == "key-123"
    assert tk._toolkits == ["gmail", "slack"]
    assert tk._connect_timeout == 10.0
    assert tk._request_timeout == 60.0

  def test_not_initialized_returns_empty_tools(self):
    tk = ComposioToolkit()
    assert tk.tools == []

  def test_not_initialized_returns_empty_dependencies(self):
    tk = ComposioToolkit()
    assert tk.dependencies == {}


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestInitialize:
  @pytest.mark.asyncio
  async def test_missing_api_key_raises(self):
    """No api_key and no env var → ValueError."""
    tk = ComposioToolkit(user_id="u1")
    with patch.dict(os.environ, {}, clear=True):
      # Remove COMPOSIO_API_KEY if it exists
      env_copy = dict(os.environ)
      env_copy.pop("COMPOSIO_API_KEY", None)
      with patch.dict(os.environ, env_copy, clear=True):
        with pytest.raises(ValueError, match="Composio API key required"):
          await tk.initialize()

  @pytest.mark.asyncio
  async def test_api_key_from_env(self, mock_composio_sdk, mock_mcp_toolkit):
    """Falls back to COMPOSIO_API_KEY env var."""
    mock_module, mock_client, _ = mock_composio_sdk

    with (
      patch.dict(os.environ, {"COMPOSIO_API_KEY": "env-key-456"}),
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit),
      patch("definable.composio.toolkit.MCPConfig"),
      patch("definable.composio.toolkit.MCPServerConfig"),
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      tk = ComposioToolkit(user_id="u1")
      await tk.initialize()

      # SDK should be called with the env key
      mock_module.Composio.assert_called_once_with(api_key="env-key-456")

  @pytest.mark.asyncio
  async def test_initialize_creates_session_and_mcp_toolkit(self, mock_composio_sdk, mock_mcp_toolkit):
    """Full happy path: session created, MCP connected, tools available."""
    mock_module, mock_client, _ = mock_composio_sdk

    with (
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit) as mock_mcp_cls,
      patch("definable.composio.toolkit.MCPServerConfig") as mock_srv_cls,
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      tk = ComposioToolkit(user_id="u1", api_key="key-123", toolkits=["gmail"])
      await tk.initialize()

      # Session created with correct params
      mock_client.create.assert_called_once_with(user_id="u1", toolkits=["gmail"])

      # MCPServerConfig built with session URL
      mock_srv_cls.assert_called_once_with(
        name="composio",
        transport="http",
        url="https://mcp.composio.dev/session/abc123",
        headers={"Authorization": "Bearer test-token"},
        connect_timeout=30.0,
        request_timeout=120.0,
      )

      # MCPToolkit created with include_server_prefix=False
      mock_mcp_cls.assert_called_once()
      _, kwargs = mock_mcp_cls.call_args
      assert kwargs["include_server_prefix"] is False

      # MCPToolkit initialized
      mock_mcp_toolkit.initialize.assert_awaited_once()

      # Tools now available
      assert len(tk.tools) == 1
      assert tk._initialized is True

      # Dependencies delegated to inner MCPToolkit
      assert "_mcp_toolkit_client" in tk.dependencies
      assert tk.dependencies is mock_mcp_toolkit.dependencies

  @pytest.mark.asyncio
  async def test_initialize_idempotent(self, mock_composio_sdk, mock_mcp_toolkit):
    """Calling initialize() twice does not recreate the session."""
    mock_module, mock_client, _ = mock_composio_sdk

    with (
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit),
      patch("definable.composio.toolkit.MCPConfig"),
      patch("definable.composio.toolkit.MCPServerConfig"),
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      tk = ComposioToolkit(user_id="u1", api_key="key-123")
      await tk.initialize()
      await tk.initialize()  # second call

      # Only one session created
      assert mock_client.create.call_count == 1
      mock_mcp_toolkit.initialize.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_import_error_without_sdk(self):
    """Missing composio package → ImportError with install hint."""
    import sys

    # Ensure composio is not importable
    with patch.dict(sys.modules, {"composio": None}):
      tk = ComposioToolkit(user_id="u1", api_key="key-123")
      with pytest.raises(ImportError, match="pip install"):
        await tk.initialize()

  @pytest.mark.asyncio
  async def test_session_creation_failure(self, mock_composio_sdk):
    """SDK error during session creation → RuntimeError."""
    mock_module, mock_client, _ = mock_composio_sdk
    mock_client.create.side_effect = Exception("API rate limited")

    with patch.dict("sys.modules", {"composio": mock_module}):
      tk = ComposioToolkit(user_id="u1", api_key="key-123")
      with pytest.raises(RuntimeError, match="Failed to create session"):
        await tk.initialize()

  @pytest.mark.asyncio
  async def test_no_toolkits_param_omitted(self, mock_composio_sdk, mock_mcp_toolkit):
    """When toolkits=None, the 'toolkits' kwarg is not passed to create()."""
    mock_module, mock_client, _ = mock_composio_sdk

    with (
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit),
      patch("definable.composio.toolkit.MCPConfig"),
      patch("definable.composio.toolkit.MCPServerConfig"),
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      tk = ComposioToolkit(user_id="u1", api_key="key-123")
      await tk.initialize()

      mock_client.create.assert_called_once_with(user_id="u1")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
  @pytest.mark.asyncio
  async def test_shutdown_delegates_to_mcp_toolkit(self, mock_composio_sdk, mock_mcp_toolkit):
    """shutdown() delegates to internal MCPToolkit."""
    mock_module, _, _ = mock_composio_sdk

    with (
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit),
      patch("definable.composio.toolkit.MCPConfig"),
      patch("definable.composio.toolkit.MCPServerConfig"),
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      tk = ComposioToolkit(user_id="u1", api_key="key-123")
      await tk.initialize()
      await tk.shutdown()

      mock_mcp_toolkit.shutdown.assert_awaited_once()
      assert tk._initialized is False
      assert tk._mcp_toolkit is None
      assert tk.tools == []

  @pytest.mark.asyncio
  async def test_shutdown_idempotent(self):
    """Calling shutdown() before initialize() is safe."""
    tk = ComposioToolkit()
    await tk.shutdown()  # no-op, no error
    assert tk._initialized is False


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


class TestContextManager:
  @pytest.mark.asyncio
  async def test_context_manager(self, mock_composio_sdk, mock_mcp_toolkit):
    """async with initializes and shuts down."""
    mock_module, _, _ = mock_composio_sdk

    with (
      patch("definable.composio.toolkit.MCPToolkit", return_value=mock_mcp_toolkit),
      patch("definable.composio.toolkit.MCPConfig"),
      patch("definable.composio.toolkit.MCPServerConfig"),
      patch.dict("sys.modules", {"composio": mock_module}),
    ):
      async with ComposioToolkit(user_id="u1", api_key="key-123") as tk:
        assert tk._initialized is True
        assert len(tk.tools) == 1

      # After exit
      assert tk._initialized is False


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
  def test_repr_not_initialized(self):
    tk = ComposioToolkit(user_id="u1")
    assert "not initialized" in repr(tk)
    assert "user='u1'" in repr(tk)
    assert "tools=0" in repr(tk)

  def test_repr_with_toolkits(self):
    tk = ComposioToolkit(user_id="u1", toolkits=["gmail"])
    assert "toolkits=['gmail']" in repr(tk)
