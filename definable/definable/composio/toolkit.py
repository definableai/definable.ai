"""ComposioToolkit — 1000+ SaaS integrations via Composio's MCP endpoint.

Creates a per-user Tool Router session through Composio's SDK, extracts
the MCP URL + auth headers, and wraps them in the battle-tested MCPToolkit.
The agent gets meta-tools for dynamic tool discovery, auth management,
and multi-tool execution.

Per-user flow:
  1. ComposioToolkit(user_id="user_123") — binds to one user
  2. await toolkit.initialize() — creates session, connects MCP
  3. Agent receives meta-tools (search, connect, execute, etc.)
  4. Agent dynamically discovers and uses any of Composio's integrations
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from definable.agent.toolkit import Toolkit
from definable.mcp.config import MCPConfig, MCPServerConfig
from definable.mcp.toolkit import MCPToolkit
from definable.utils.log import log_debug, log_info

if TYPE_CHECKING:
  from definable.tool.function import Function


class ComposioToolkit(Toolkit):
  """Agent toolkit providing 1000+ SaaS integrations via Composio.

  Wraps Composio's Tool Router MCP endpoint using the existing
  MCPToolkit infrastructure. Each toolkit instance is scoped to a
  single user, providing authentication isolation.

  Usage::

      from definable.composio import ComposioToolkit
      from definable.agent import Agent

      async with ComposioToolkit(user_id="user_123") as toolkit:
          agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
          result = await agent.arun("Send an email via Gmail to ...")

  The agent receives meta-tools for:
    - Searching available tools across 1000+ integrations
    - Managing user connections and authentication flows
    - Executing actions on connected services
    - Uploading files to connected services
  """

  def __init__(
    self,
    *,
    user_id: str = "default",
    api_key: str | None = None,
    toolkits: list[str] | None = None,
    connect_timeout: float = 30.0,
    request_timeout: float = 120.0,
  ) -> None:
    """Initialize the Composio toolkit.

    Args:
        user_id: Composio user identifier for session isolation.
        api_key: Composio API key. Falls back to ``COMPOSIO_API_KEY`` env var.
        toolkits: Optional list of toolkit names to scope the session
            (e.g. ``["gmail", "slack"]``). If ``None``, all toolkits are available.
        connect_timeout: Timeout in seconds for MCP connection.
        request_timeout: Timeout in seconds for MCP tool execution.
    """
    super().__init__()

    self._user_id = user_id
    self._api_key = api_key
    self._toolkits = toolkits
    self._connect_timeout = connect_timeout
    self._request_timeout = request_timeout

    self._mcp_toolkit: Optional[MCPToolkit] = None
    self._initialized = False

  @property
  def tools(self) -> List["Function"]:
    """Return discovered Composio tools.

    Returns an empty list until ``initialize()`` has been called.
    """
    if not self._initialized or self._mcp_toolkit is None:
      return []
    return self._mcp_toolkit.tools

  @property
  def dependencies(self) -> Dict[str, Any]:
    """Delegate dependencies to inner MCPToolkit.

    The MCPToolkit stores the MCP client reference in its dependencies
    dict (key ``_mcp_toolkit_client``). The agent merges toolkit
    dependencies into each tool at collection time, so we must
    expose them here for the client to be available at call time.
    """
    if self._mcp_toolkit is not None:
      return self._mcp_toolkit.dependencies
    return self._dependencies

  @property
  def user_id(self) -> str:
    """The Composio user ID this toolkit is scoped to."""
    return self._user_id

  async def initialize(self) -> None:
    """Create a Composio Tool Router session and connect via MCP.

    1. Imports the ``composio`` SDK (raises clear error if missing).
    2. Creates a session for the configured user.
    3. Extracts the MCP URL and auth headers.
    4. Builds and connects an internal ``MCPToolkit``.

    Idempotent — calling multiple times is safe.

    Raises:
        ImportError: If the ``composio`` package is not installed.
        ValueError: If no API key is available.
        RuntimeError: If session creation or MCP connection fails.
    """
    if self._initialized:
      return

    # Resolve API key
    api_key = self._api_key or os.environ.get("COMPOSIO_API_KEY")
    if not api_key:
      raise ValueError("Composio API key required. Pass api_key= or set COMPOSIO_API_KEY env var.")

    # Lazy-import composio SDK
    try:
      from composio import Composio  # type: ignore[import-untyped]
    except ImportError as exc:
      raise ImportError("The 'composio' package is required for ComposioToolkit. Install it with: pip install 'definable[composio]'") from exc

    log_info(f"ComposioToolkit: Creating session for user '{self._user_id}'...")

    # Create Tool Router session
    try:
      client = Composio(api_key=api_key)
      if self._toolkits:
        session = client.create(user_id=self._user_id, toolkits=self._toolkits)
      else:
        session = client.create(user_id=self._user_id)
    except Exception as exc:
      raise RuntimeError(f"ComposioToolkit: Failed to create session for user '{self._user_id}': {exc}") from exc

    # Extract MCP connection details
    mcp_url: str = session.mcp.url
    raw_headers = session.mcp.headers
    mcp_headers: dict[str, str] = {k: v for k, v in (raw_headers or {}).items() if v is not None}

    log_debug(f"ComposioToolkit: MCP URL obtained for user '{self._user_id}'")

    # Build internal MCPToolkit
    server_config = MCPServerConfig(
      name="composio",
      transport="http",
      url=mcp_url,
      headers=mcp_headers,
      connect_timeout=self._connect_timeout,
      request_timeout=self._request_timeout,
    )
    mcp_config = MCPConfig(servers=[server_config])

    self._mcp_toolkit = MCPToolkit(
      config=mcp_config,
      include_server_prefix=False,
    )
    await self._mcp_toolkit.initialize()

    self._initialized = True
    log_info(f"ComposioToolkit: Ready with {len(self._mcp_toolkit.tools)} tools for user '{self._user_id}'")

  async def shutdown(self) -> None:
    """Disconnect from the Composio MCP endpoint.

    Idempotent — safe to call multiple times.
    """
    if not self._initialized:
      return

    self._initialized = False

    if self._mcp_toolkit is not None:
      await self._mcp_toolkit.shutdown()
      self._mcp_toolkit = None

    log_info(f"ComposioToolkit: Shutdown complete for user '{self._user_id}'")

  # =========================================================================
  # Context Manager
  # =========================================================================

  async def __aenter__(self) -> "ComposioToolkit":
    """Async context manager entry — initializes the toolkit."""
    await self.initialize()
    return self

  async def __aexit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Async context manager exit — shuts down the toolkit."""
    await self.shutdown()

  def __repr__(self) -> str:
    tool_count = len(self.tools)
    status = "initialized" if self._initialized else "not initialized"
    toolkits_str = f", toolkits={self._toolkits}" if self._toolkits else ""
    return f"ComposioToolkit(user='{self._user_id}'{toolkits_str}, tools={tool_count}, {status})"
