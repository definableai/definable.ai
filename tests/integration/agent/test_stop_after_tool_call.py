"""
Behavioral tests: stop_after_tool_call flag on tools.

Migrated from: tests_e2e/behavioral/test_stop_after_tool_call.py

Strategy:
  - MockModel with side_effect to simulate tool calls
  - Assert on OUTCOMES: loop termination, model call count

Covers:
  - Tool with stop_after_tool_call=True ends the agent loop immediately
  - stop_after_tool_call on one tool does not affect unrelated tools
  - Normal tools (no stop flag) allow the loop to continue for a final answer
"""

from unittest.mock import MagicMock

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_response(**overrides):
  """Create a MagicMock response with all required fields."""
  response = MagicMock()
  response.response_usage = Metrics()
  response.reasoning_content = None
  response.citations = None
  response.images = None
  response.videos = None
  response.audios = None
  response.content = overrides.get("content", "")
  response.tool_calls = overrides.get("tool_calls", [])
  return response


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def stop_tool() -> str:
  """Tool that should stop the loop."""
  return "stop_result"


# Set stop_after_tool_call on the Function object
stop_tool.stop_after_tool_call = True


@tool
def normal_tool() -> str:
  """Normal tool that does not stop."""
  return "normal_result"


@tool
def other_tool() -> str:
  """Another normal tool."""
  return "other_result"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
class TestStopAfterToolCall:
  """Verify that stop_after_tool_call terminates the agentic loop early."""

  async def test_stop_after_tool_call_ends_loop(self):
    """When a tool with stop_after_tool_call=True is called,
    the loop should end without a subsequent model call for a final answer."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content="",
          tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "stop_tool", "arguments": "{}"}},
          ],
        )
      else:
        # This should NOT be reached if stop works correctly
        return _make_response(content="Should not see this")

    model = MockModel(side_effect=side_effect)
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[stop_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Use the stop tool")

    # Only 1 model call should have happened — no second call for final answer
    assert call_count == 1, f"Expected 1 model call, got {call_count}"

    # The tool should still have been executed
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "stop_tool" in tool_names

  async def test_stop_does_not_affect_other_tools(self):
    """When stop_tool and normal_tool are both available but only
    stop_tool is called, the loop ends. normal_tool is unaffected."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        # Model only calls stop_tool
        return _make_response(
          content="",
          tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "stop_tool", "arguments": "{}"}},
          ],
        )
      else:
        return _make_response(content="Should not reach here")

    model = MockModel(side_effect=side_effect)
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[stop_tool, normal_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Use the stop tool")

    assert call_count == 1, f"Expected 1 model call, got {call_count}"
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "stop_tool" in tool_names
    assert "normal_tool" not in tool_names

  async def test_no_stop_continues_loop(self):
    """A normal tool (without stop_after_tool_call) should allow the model to
    be called again to produce a final answer."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _make_response(
          content="",
          tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "other_tool", "arguments": "{}"}},
          ],
        )
      else:
        # Second call: model produces the final answer
        return _make_response(content="Final answer after tool")

    model = MockModel(side_effect=side_effect)
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[other_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Use the other tool")

    # Model should have been called twice: once for tool call, once for final answer
    assert call_count == 2, f"Expected 2 model calls, got {call_count}"
    assert output.content == "Final answer after tool"
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "other_tool" in tool_names
