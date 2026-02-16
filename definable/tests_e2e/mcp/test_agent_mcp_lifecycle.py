"""Tests for Agent-managed MCP toolkit lifecycle.

Verifies that Agent can initialize and shut down async toolkits
(MCPToolkit) through its own context manager and auto-init in arun().
"""

import asyncio

from definable.agents.agent import Agent, AsyncLifecycleToolkit
from definable.agents.testing import MockModel
from definable.mcp import MCPConfig, MCPToolkit


class TestAgentMCPLifecycle:
  """Core lifecycle tests: Agent owns toolkit init/shutdown."""

  async def test_agent_aenter_initializes_mcp(self, mock_config: MCPConfig):
    """async with Agent(...) should initialize uninitialized toolkits."""
    toolkit = MCPToolkit(mock_config)
    assert not toolkit._initialized

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    async with agent:
      assert toolkit._initialized
      # Tools should be available
      assert len(agent.tool_names) > 0
      assert any("echo" in name for name in agent.tool_names)

  async def test_agent_arun_auto_initializes_mcp(self, mock_config: MCPConfig):
    """arun() without context manager should auto-init toolkits."""
    toolkit = MCPToolkit(mock_config)
    assert not toolkit._initialized

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    # arun triggers auto-init
    output = await agent.arun("Hello")
    assert toolkit._initialized
    assert output is not None

    # Manual cleanup since no context manager
    await toolkit.shutdown()

  async def test_agent_skips_preinit_toolkit(self, mock_config: MCPConfig):
    """Pre-initialized toolkit should not be re-initialized by agent."""
    toolkit = MCPToolkit(mock_config)
    await toolkit.initialize()
    assert toolkit._initialized

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    async with agent:
      # Toolkit was already initialized — agent should not own it
      assert toolkit._initialized
      assert len(agent._agent_owned_toolkits) == 0

    # Toolkit should still be initialized after agent exit
    assert toolkit._initialized

    # Clean up manually
    await toolkit.shutdown()

  async def test_agent_does_not_shutdown_preinit(self, mock_config: MCPConfig):
    """Agent exit should NOT shut down pre-initialized (user-managed) toolkits."""
    toolkit = MCPToolkit(mock_config)
    await toolkit.initialize()

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    async with agent:
      pass  # Enter and immediately exit

    # Toolkit should still be initialized — agent didn't own it
    assert toolkit._initialized
    assert len(toolkit.tools) > 0

    await toolkit.shutdown()

  async def test_agent_shuts_down_own_toolkit(self, mock_config: MCPConfig):
    """Agent exit should shut down toolkits it initialized."""
    toolkit = MCPToolkit(mock_config)
    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    async with agent:
      assert toolkit._initialized
      assert len(agent._agent_owned_toolkits) == 1

    # After exit, agent should have shut it down
    assert not toolkit._initialized
    assert len(agent._agent_owned_toolkits) == 0

  async def test_concurrent_arun_single_init(self, mock_config: MCPConfig):
    """Multiple concurrent arun() calls should initialize toolkit exactly once."""
    toolkit = MCPToolkit(mock_config)
    init_count = 0
    original_init = toolkit.initialize

    async def counting_init():
      nonlocal init_count
      init_count += 1
      await original_init()

    toolkit.initialize = counting_init  # type: ignore[assignment]

    mock_model = MockModel(responses=["a", "b", "c"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    # Launch 3 concurrent arun calls
    results = await asyncio.gather(
      agent.arun("one"),
      agent.arun("two"),
      agent.arun("three"),
    )

    assert len(results) == 3
    assert init_count == 1  # Only initialized once despite 3 concurrent calls

    await toolkit.shutdown()

  async def test_toolkit_init_failure_nonfatal(self, mock_config: MCPConfig):
    """Toolkit init failure should not prevent agent from working."""
    # Create a toolkit that fails to initialize
    failing_toolkit = MCPToolkit(mock_config)

    async def failing_init():
      raise ConnectionError("Server unreachable")

    failing_toolkit.initialize = failing_init  # type: ignore[assignment]

    mock_model = MockModel(responses=["still works"])
    agent = Agent(model=mock_model, toolkits=[failing_toolkit])

    # Agent should still work, just without MCP tools
    async with agent:
      output = await agent.arun("Hello")
      assert output is not None
      assert output.content == "still works"

  async def test_agent_reuse_after_shutdown(self, mock_config: MCPConfig):
    """Agent should re-initialize toolkits when reused after shutdown."""
    toolkit = MCPToolkit(mock_config)
    mock_model = MockModel(responses=["first", "second"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    # First use
    async with agent:
      assert toolkit._initialized
      tool_count_1 = len(agent.tool_names)
      assert tool_count_1 > 0

    # After exit
    assert not toolkit._initialized

    # Second use — should re-initialize
    async with agent:
      assert toolkit._initialized
      assert len(agent.tool_names) == tool_count_1  # Same tools available

  async def test_backwards_compat_preinit_pattern(self, mock_config: MCPConfig):
    """Old pattern: async with toolkit: Agent(...) should still work."""
    toolkit = MCPToolkit(mock_config)
    mock_model = MockModel(responses=["works"])

    async with toolkit:
      agent = Agent(model=mock_model, toolkits=[toolkit])
      output = await agent.arun("Hello")
      assert output is not None
      # Agent should detect it's already initialized and not own it
      assert len(agent._agent_owned_toolkits) == 0


class TestAsyncLifecycleProtocol:
  """Tests for the AsyncLifecycleToolkit Protocol."""

  def test_mcp_toolkit_satisfies_protocol(self, mock_config: MCPConfig):
    """MCPToolkit should satisfy AsyncLifecycleToolkit protocol."""
    toolkit = MCPToolkit(mock_config)
    assert isinstance(toolkit, AsyncLifecycleToolkit)

  def test_regular_toolkit_does_not_satisfy_protocol(self):
    """Base Toolkit should NOT satisfy AsyncLifecycleToolkit protocol."""
    from definable.agents.toolkit import Toolkit

    toolkit = Toolkit()
    assert not isinstance(toolkit, AsyncLifecycleToolkit)


class TestMixedToolkits:
  """Tests for agents with both regular and async toolkits."""

  async def test_mix_of_regular_and_mcp_toolkits(self, mock_config: MCPConfig):
    """Agent should handle mix of regular Toolkit and MCPToolkit."""
    from definable.agents.toolkit import Toolkit

    regular_toolkit = Toolkit()
    mcp_toolkit = MCPToolkit(mock_config)

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[regular_toolkit, mcp_toolkit])

    async with agent:
      # MCP toolkit initialized, regular toolkit unaffected
      assert mcp_toolkit._initialized
      assert len(agent._agent_owned_toolkits) == 1
      assert agent._agent_owned_toolkits[0] is mcp_toolkit

  async def test_tool_names_before_async_init(self, mock_config: MCPConfig):
    """tool_names before async init should return sync tools only."""
    mcp_toolkit = MCPToolkit(mock_config)

    mock_model = MockModel(responses=["done"])
    agent = Agent(model=mock_model, toolkits=[mcp_toolkit])

    # Before init — MCP toolkit returns [] for tools
    names_before = agent.tool_names
    assert len(names_before) == 0  # No sync tools, MCP not initialized

    async with agent:
      names_after = agent.tool_names
      assert len(names_after) > 0  # MCP tools now available

  async def test_arun_stream_auto_initializes(self, mock_config: MCPConfig):
    """arun_stream() should also auto-init toolkits."""
    toolkit = MCPToolkit(mock_config)
    mock_model = MockModel(responses=["streamed"])
    agent = Agent(model=mock_model, toolkits=[toolkit])

    events = []
    async for event in agent.arun_stream("Hello"):
      events.append(event)

    assert toolkit._initialized
    assert len(events) > 0

    await toolkit.shutdown()
