"""
Behavioral tests for ClaudeCodeAgent — mock SubprocessTransport to simulate CLI.

Migrated from: tests_e2e/behavioral/test_claude_code_agent.py

Covers:
  - Basic run returns content and metrics
  - Session ID propagation
  - Memory integration (context injection)
  - Knowledge integration (RAG context injection)
  - Custom tools pass MCP config
  - Input guardrails block before CLI
  - Output guardrails modify response
  - Middleware wraps execution
  - Session resume passes --resume flag
"""

from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.claude_code.agent import ClaudeCodeAgent
from definable.agent.events import RunStatus


# ---------------------------------------------------------------------------
# Mock transport helper
# ---------------------------------------------------------------------------


class MockTransport:
  """Simulates SubprocessTransport for testing without the CLI."""

  def __init__(self, responses: List[dict]):
    self._responses = responses
    self._sent: List[dict] = []
    self._connected = False

  async def connect(self, args: List[str]) -> None:
    self._connected = True
    self._connect_args = args

  async def send(self, message: dict) -> None:
    self._sent.append(message)

  async def receive(self) -> AsyncIterator[dict]:
    for resp in self._responses:
      yield resp

  async def close(self) -> None:
    self._connected = False

  @property
  def is_running(self) -> bool:
    return self._connected


def _make_assistant_text(text: str, model: str = "claude-sonnet-4-6") -> dict:
  return {
    "type": "assistant",
    "message": {
      "content": [{"type": "text", "text": text}],
      "model": model,
    },
  }


def _make_result(
  session_id: str = "sess-test",
  input_tokens: int = 100,
  output_tokens: int = 50,
  cost: float = 0.01,
) -> dict:
  return {
    "type": "result",
    "subtype": "success",
    "session_id": session_id,
    "duration_ms": 5000,
    "duration_api_ms": 3000,
    "is_error": False,
    "turn_count": 1,
    "total_cost_usd": cost,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
  }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
class TestBasicRun:
  async def test_basic_run_returns_content(self):
    """Basic run returns RunOutput with content."""
    mock_transport = MockTransport([
      _make_assistant_text("Bug fixed successfully."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(model="claude-sonnet-4-6", instructions="Fix bugs.")

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("Fix the auth bug")

    assert result.content == "Bug fixed successfully."
    assert result.status == RunStatus.completed
    assert result.model == "claude-sonnet-4-6"
    assert result.metrics.input_tokens == 100  # type: ignore[union-attr]
    assert result.metrics.output_tokens == 50  # type: ignore[union-attr]
    assert result.metrics.cost == 0.01  # type: ignore[union-attr]

  async def test_run_session_id_propagated(self):
    """Session ID from ResultMessage is propagated to RunOutput."""
    mock_transport = MockTransport([
      _make_assistant_text("Done."),
      _make_result(session_id="sess-custom-123"),
    ])

    agent = ClaudeCodeAgent()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("Hello")

    assert result.session_id == "sess-custom-123"


@pytest.mark.behavioral
@pytest.mark.integration
class TestMemoryIntegration:
  async def test_run_with_memory_injects_context(self):
    """When memory is configured, session history is injected into system prompt."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore
    from definable.model.message import Message as DefMessage

    store = InMemoryStore()
    memory = Memory(store=store)
    await memory._ensure_initialized()
    await memory.add(DefMessage(role="user", content="User prefers Python."), session_id="default", user_id="user-1")

    mock_transport = MockTransport([
      _make_assistant_text("Got it."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(instructions="Help the user.", memory=True)
    # Manually set up the memory manager with pre-populated store
    agent._memory_manager = memory
    agent._initialized = True
    agent._tool_bridge = MagicMock()
    agent._tool_bridge.tool_count = 0
    agent._tool_bridge.get_mcp_config.return_value = {}

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("What do I like?", user_id="user-1", session_id="default")

    # Check that the system prompt sent to CLI includes memory
    connect_args = mock_transport._connect_args
    system_prompt_idx = connect_args.index("--system-prompt") + 1
    system_prompt = connect_args[system_prompt_idx]
    assert "User prefers Python" in system_prompt
    assert "<user_memory>" in system_prompt


@pytest.mark.behavioral
@pytest.mark.integration
class TestKnowledgeIntegration:
  async def test_run_with_knowledge_injects_context(self):
    """When knowledge is configured, RAG results are injected into system prompt."""
    mock_transport = MockTransport([
      _make_assistant_text("Based on the docs..."),
      _make_result(),
    ])

    # Create a mock Knowledge with async asearch and format_context
    mock_knowledge = MagicMock()
    mock_doc = MagicMock()
    mock_doc.content = "JWT tokens expire after 24 hours."
    mock_knowledge.asearch = AsyncMock(return_value=[mock_doc])
    mock_knowledge.format_context.return_value = "JWT tokens expire after 24 hours."

    agent = ClaudeCodeAgent(instructions="Help with auth.")
    # Directly set internal state to bypass isinstance resolution
    agent._knowledge_instance = mock_knowledge
    agent._initialized = True
    agent._tool_bridge = MagicMock()
    agent._tool_bridge.tool_count = 0
    agent._tool_bridge.get_mcp_config.return_value = {}

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("How does auth work?")

    connect_args = mock_transport._connect_args
    system_prompt_idx = connect_args.index("--system-prompt") + 1
    system_prompt = connect_args[system_prompt_idx]
    assert "JWT tokens expire" in system_prompt
    assert "<knowledge_context>" in system_prompt


@pytest.mark.behavioral
@pytest.mark.integration
class TestCustomTools:
  async def test_run_with_custom_tools_passes_mcp_config(self):
    """Custom tools result in --mcp-config in CLI args and --allowedTools."""
    from definable.tool.function import Function

    def deploy(branch: str = "main") -> str:
      """Deploy the app."""
      return f"Deployed {branch}"

    fn = Function.from_callable(deploy)

    mock_transport = MockTransport([
      _make_assistant_text("Deployed."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(tools=[fn])

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      with patch("definable.claude_code.agent.ToolServer") as MockServer:
        # Mock the tool server to avoid actually starting a Unix socket
        mock_server_instance = MockServer.return_value
        mock_server_instance.is_running = True
        mock_server_instance.start = AsyncMock()
        mock_server_instance.stop = AsyncMock()
        mock_server_instance.get_mcp_config_path.return_value = '{"mcpServers":{"definable":{"command":"python","args":["server.py"]}}}'
        await agent.arun("Deploy to production")

    # Check --mcp-config in CLI args
    connect_args = mock_transport._connect_args
    assert "--mcp-config" in connect_args

    # Check allowedTools includes MCP tool name
    if "--allowedTools" in connect_args:
      tools_str = connect_args[connect_args.index("--allowedTools") + 1]
      assert "mcp__definable__deploy" in tools_str


@pytest.mark.behavioral
@pytest.mark.integration
class TestInputGuardrails:
  async def test_input_guardrail_blocks(self):
    """Input guardrail blocks before CLI is called."""
    from definable.agent.guardrail import Guardrails, max_tokens

    agent = ClaudeCodeAgent(guardrails=Guardrails(input=[max_tokens(5)]))

    result = await agent.arun("This is a very long prompt that exceeds the limit")

    assert result.status == RunStatus.blocked
    assert "blocked" in result.content.lower()  # type: ignore[union-attr]


@pytest.mark.behavioral
@pytest.mark.integration
class TestOutputGuardrails:
  async def test_output_guardrail_modifies(self):
    """Output guardrail can modify the response."""
    from definable.agent.guardrail import Guardrails
    from definable.agent.guardrail.base import GuardrailResult
    from definable.agent.events import RunContext

    class RedactSSN:
      name = "redact_ssn"

      async def check(self, text: str, context: RunContext) -> GuardrailResult:
        if "123-45-6789" in text:
          return GuardrailResult.modify(text.replace("123-45-6789", "[REDACTED]"), "SSN redacted")
        return GuardrailResult.allow()

    mock_transport = MockTransport([
      _make_assistant_text("Your SSN is 123-45-6789."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(guardrails=Guardrails(output=[RedactSSN()]))

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("What's my SSN?")

    assert "[REDACTED]" in result.content  # type: ignore[operator]
    assert "123-45-6789" not in result.content  # type: ignore[operator]


@pytest.mark.behavioral
@pytest.mark.integration
class TestMiddleware:
  async def test_middleware_wraps_execution(self):
    """Middleware wraps the execution pipeline."""
    calls = []

    class LoggingMiddleware:
      async def __call__(self, ctx, next_handler):
        calls.append("before")
        result = await next_handler(ctx)
        calls.append("after")
        return result

    mock_transport = MockTransport([
      _make_assistant_text("Done."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(middleware=[LoggingMiddleware()])

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      result = await agent.arun("Hello")

    assert calls == ["before", "after"]
    assert result.content == "Done."


@pytest.mark.behavioral
@pytest.mark.integration
class TestSessionResume:
  async def test_session_resume_passes_flag(self):
    """When continue_conversation=True, --resume flag is passed."""
    mock_transport = MockTransport([
      _make_assistant_text("Continuing..."),
      _make_result(session_id="sess-resume"),
    ])

    agent = ClaudeCodeAgent(continue_conversation=True)

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("Continue our work", session_id="sess-resume")

    assert "--resume" in mock_transport._connect_args
    assert "sess-resume" in mock_transport._connect_args


@pytest.mark.behavioral
@pytest.mark.integration
class TestMemoryRoundTrip:
  async def test_memory_store_and_recall_same_session(self):
    """Memory store and recall use the same session_id from RunContext."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    store = InMemoryStore()
    memory = Memory(store=store)

    mock_transport = MockTransport([
      _make_assistant_text("I remember Python."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(instructions="Help.", memory=memory)
    agent._ensure_initialized()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("What do I like?", user_id="user-1", session_id="sess-round-trip")

    # Allow fire-and-forget task to complete
    import asyncio

    await asyncio.sleep(0.05)

    # Now verify the memory was stored with the correct session_id
    entries = await memory.get_entries("sess-round-trip", "user-1")
    assert len(entries) >= 1  # At least the user message was stored
    contents = [e.content for e in entries]
    assert any("What do I like?" in c for c in contents)

  async def test_memory_works_without_user_id(self):
    """Memory store works even when user_id is not provided (defaults to 'default')."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore

    store = InMemoryStore()
    memory = Memory(store=store)

    mock_transport = MockTransport([
      _make_assistant_text("Done."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(memory=memory)
    agent._ensure_initialized()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("Hello", session_id="no-user-sess")

    import asyncio

    await asyncio.sleep(0.05)

    # Should store under user_id="default"
    entries = await memory.get_entries("no-user-sess", "default")
    assert len(entries) >= 1


@pytest.mark.behavioral
@pytest.mark.integration
class TestToolkitIntegrationBehavioral:
  async def test_toolkit_tools_available_in_run(self):
    """Toolkit tools are registered and available during a run."""
    from definable.tool.function import Function

    def tk_tool(x: int) -> str:
      """Toolkit tool."""
      return f"result: {x}"

    fn = Function.from_callable(tk_tool)

    mock_toolkit = MagicMock()
    mock_toolkit._initialized = False
    mock_toolkit.initialize = AsyncMock()
    mock_toolkit.shutdown = AsyncMock()
    mock_toolkit.tools = [fn]

    mock_transport = MockTransport([
      _make_assistant_text("Used toolkit."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(toolkits=[mock_toolkit])
    agent._ensure_initialized()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      with patch("definable.claude_code.agent.ToolServer") as MockServer:
        mock_server_instance = MockServer.return_value
        mock_server_instance.is_running = True
        mock_server_instance.start = AsyncMock()
        mock_server_instance.stop = AsyncMock()
        mock_server_instance.get_mcp_config_path.return_value = "/tmp/config.json"
        await agent.arun("Use the toolkit")

    mock_toolkit.initialize.assert_called_once()
    # Toolkit tool should be in bridge
    assert agent._tool_bridge is not None
    assert "tk_tool" in agent._tool_bridge._tools


@pytest.mark.behavioral
@pytest.mark.integration
class TestMemoryPersistsAcrossAruns:
  async def test_memory_persists_across_arun_calls(self):
    """Two arun() calls without explicit session_id share the same session — memory from first is visible in second."""
    from definable.memory.manager import Memory
    from definable.memory.store.in_memory import InMemoryStore
    import asyncio

    store = InMemoryStore()
    memory = Memory(store=store)

    # First call — stores "hello" in memory
    mock_transport_1 = MockTransport([
      _make_assistant_text("Hi there!"),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(instructions="Help.", memory=memory)
    agent._ensure_initialized()

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport_1):
      await agent.arun("hello")

    # Let fire-and-forget memory store complete
    await asyncio.sleep(0.05)

    # Second call — memory recall should find entries from the first call
    mock_transport_2 = MockTransport([
      _make_assistant_text("You said hello earlier."),
      _make_result(),
    ])

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport_2):
      await agent.arun("how many messages?")

    # The system prompt of the second call should contain memory from the first
    connect_args = mock_transport_2._connect_args
    system_prompt_idx = connect_args.index("--system-prompt") + 1
    system_prompt = connect_args[system_prompt_idx]
    assert "<user_memory>" in system_prompt
    assert "hello" in system_prompt


@pytest.mark.behavioral
@pytest.mark.integration
class TestMCPToolkitPassthrough:
  async def test_mcp_toolkit_passthrough_in_run(self):
    """Full arun() with MCPToolkit produces --mcp-config in CLI args."""
    from definable.mcp.toolkit import MCPToolkit
    from definable.mcp.config import MCPConfig, MCPServerConfig

    mcp_toolkit = MCPToolkit(
      config=MCPConfig(
        servers=[
          MCPServerConfig(name="gmail", transport="http", url="https://backend.composio.dev/mcp"),
        ]
      )
    )

    mock_transport = MockTransport([
      _make_assistant_text("Email sent."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(toolkits=[mcp_toolkit])

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("List my emails")

    # Verify --mcp-config appears in CLI args
    connect_args = mock_transport._connect_args
    assert "--mcp-config" in connect_args

    # Find the mcp-config path and verify its content
    mcp_idx = connect_args.index("--mcp-config")
    config_path = connect_args[mcp_idx + 1]

    # File should still exist during the run, but cleaned up after
    # (the finally block runs _cleanup_mcp_temps, so file may be gone)
    # Just verify the config was correct by checking the args were set
    assert config_path.endswith(".json")

    # MCPToolkit should NOT have been initialized (no double-proxy)
    assert not mcp_toolkit._initialized
