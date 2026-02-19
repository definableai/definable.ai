"""Behavioral tests for ClaudeCodeAgent — mock SubprocessTransport to simulate CLI."""

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


class TestBasicRun:
  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
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


class TestMemoryIntegration:
  @pytest.mark.asyncio
  async def test_run_with_memory_injects_context(self):
    """When memory is configured, memories are injected into system prompt."""
    from definable.memory.store.in_memory import InMemoryStore
    from definable.memory.types import UserMemory

    store = InMemoryStore()
    await store.upsert_user_memory(
      UserMemory(memory_id="m1", user_id="user-1", memory="User prefers Python.", topics=["preferences"]),
    )

    mock_transport = MockTransport([
      _make_assistant_text("Got it."),
      _make_result(),
    ])

    agent = ClaudeCodeAgent(instructions="Help the user.", memory=True)
    # Manually set up the memory manager with pre-populated store
    from definable.memory.manager import MemoryManager

    agent._memory_manager = MemoryManager(store=store)
    agent._initialized = True
    agent._tool_bridge = MagicMock()
    agent._tool_bridge.tool_count = 0
    agent._tool_bridge.get_mcp_config.return_value = {}

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=mock_transport):
      await agent.arun("What do I like?", user_id="user-1")

    # Check that the system prompt sent to CLI includes memory
    connect_args = mock_transport._connect_args
    system_prompt_idx = connect_args.index("--system-prompt") + 1
    system_prompt = connect_args[system_prompt_idx]
    assert "User prefers Python" in system_prompt
    assert "<user_memory>" in system_prompt


class TestKnowledgeIntegration:
  @pytest.mark.asyncio
  async def test_run_with_knowledge_injects_context(self):
    """When knowledge is configured, RAG results are injected into system prompt."""
    mock_transport = MockTransport([
      _make_assistant_text("Based on the docs..."),
      _make_result(),
    ])

    # Create a mock Knowledge with a search method
    mock_knowledge = MagicMock()
    mock_doc = MagicMock()
    mock_doc.content = "JWT tokens expire after 24 hours."
    mock_knowledge.search.return_value = [mock_doc]

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


class TestCustomTools:
  @pytest.mark.asyncio
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


class TestInputGuardrails:
  @pytest.mark.asyncio
  async def test_input_guardrail_blocks(self):
    """Input guardrail blocks before CLI is called."""
    from definable.agent.guardrail import Guardrails, max_tokens

    agent = ClaudeCodeAgent(guardrails=Guardrails(input=[max_tokens(5)]))

    result = await agent.arun("This is a very long prompt that exceeds the limit")

    assert result.status == RunStatus.blocked
    assert "blocked" in result.content.lower()  # type: ignore[union-attr]


class TestOutputGuardrails:
  @pytest.mark.asyncio
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


class TestMiddleware:
  @pytest.mark.asyncio
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


class TestSessionResume:
  @pytest.mark.asyncio
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
