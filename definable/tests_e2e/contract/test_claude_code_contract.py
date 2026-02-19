"""Contract tests for ClaudeCodeAgent — verifies API surface compatibility."""

from inspect import iscoroutinefunction
from unittest.mock import patch

import pytest

from definable.claude_code.agent import ClaudeCodeAgent
from definable.agent.events import RunOutput


class TestClaudeCodeAgentContract:
  """Ensures ClaudeCodeAgent satisfies the same contract as Agent."""

  def test_has_arun_method(self):
    agent = ClaudeCodeAgent()
    assert hasattr(agent, "arun")
    assert iscoroutinefunction(agent.arun)

  def test_has_arun_stream_method(self):
    agent = ClaudeCodeAgent()
    assert hasattr(agent, "arun_stream")

  def test_async_context_manager(self):
    agent = ClaudeCodeAgent()
    assert hasattr(agent, "__aenter__")
    assert hasattr(agent, "__aexit__")

  def test_accepts_all_features(self):
    """Constructor accepts all documented feature parameters."""
    from definable.agent.guardrail import Guardrails
    from definable.memory import Memory
    from definable.memory.store.in_memory import InMemoryStore
    from definable.tool.function import Function

    def my_tool(x: str) -> str:
      """A tool."""
      return x

    fn = Function.from_callable(my_tool)

    agent = ClaudeCodeAgent(
      model="claude-sonnet-4-6",
      instructions="Test agent",
      allowed_tools=["Read", "Write"],
      disallowed_tools=["Bash"],
      permission_mode="bypassPermissions",
      max_turns=10,
      max_budget_usd=5.0,
      cwd="/tmp",
      cli_path="/usr/local/bin/claude",
      env={"MY_VAR": "value"},
      agent_id="test-id",
      agent_name="TestAgent",
      memory=Memory(store=InMemoryStore()),
      guardrails=Guardrails(),
      middleware=[],
      tools=[fn],
      toolkits=[],
      skills=[],
      tracing=True,
      thinking_budget_tokens=4096,
      continue_conversation=True,
    )

    assert agent.model == "claude-sonnet-4-6"
    assert agent.agent_name == "TestAgent"
    assert agent.max_turns == 10

  @pytest.mark.asyncio
  async def test_returns_run_output(self):
    """arun() returns a RunOutput instance."""

    # Create a mock transport that returns a simple response
    class MockTransport:
      async def connect(self, args):
        pass

      async def send(self, msg):
        pass

      async def receive(self):
        yield {
          "type": "assistant",
          "message": {"content": [{"type": "text", "text": "Hello"}], "model": "claude-sonnet-4-6"},
        }
        yield {
          "type": "result",
          "subtype": "success",
          "session_id": "s1",
          "duration_ms": 100,
          "duration_api_ms": 50,
          "is_error": False,
          "turn_count": 1,
          "input_tokens": 10,
          "output_tokens": 5,
        }

      async def close(self):
        pass

      @property
      def is_running(self):
        return True

    with patch("definable.claude_code.agent.SubprocessTransport", return_value=MockTransport()):
      agent = ClaudeCodeAgent()
      result = await agent.arun("Hello")

    assert isinstance(result, RunOutput)
    assert result.content == "Hello"
    assert result.run_id is not None
    assert result.session_id is not None

  def test_on_event_registration(self):
    """Can register event handlers."""
    agent = ClaudeCodeAgent()
    events = []
    agent.on_event(lambda e: events.append(e))
    assert len(agent._event_handlers) == 1
