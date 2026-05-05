"""
Regression tests for the AgentLoop rewrite backward compatibility.

Migrated from tests_e2e/regression/test_loop_backward_compat.py.

These tests verify that the new AgentLoop-based implementation preserves
the existing public API surface: RunOutput shape, tool execution records,
middleware chaining, multi-turn conversation, streaming events, and tool
result propagation.

Any future refactor of the agent loop must keep these tests green.
"""

from unittest.mock import MagicMock

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.model.response import ToolExecution
from definable.agent.events import RunOutput, RunStatus
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_tool_side_effect():
  """Return a side_effect function that simulates one tool call then a final answer."""
  call_count = 0

  def _side_effect(messages, tools, **kwargs):
    nonlocal call_count
    call_count += 1
    response = MagicMock()
    response.response_usage = Metrics()
    response.reasoning_content = None
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None
    if call_count == 1:
      response.content = ""
      response.tool_calls = [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "greet", "arguments": '{"name": "Alice"}'},
        }
      ]
    else:
      response.content = "Hello Alice!"
      response.tool_calls = []
    return response

  return _side_effect


@tool
def greet(name: str) -> str:
  """Greet someone by name."""
  return f"Hi, {name}!"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestLoopBackwardCompat:
  """Ensure the AgentLoop rewrite preserves the public RunOutput contract."""

  async def test_run_output_shape(self):
    """RunOutput must expose the canonical fields with correct types."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=_NO_TRACE)  # type: ignore[arg-type]
    output = await agent.arun("Hi")

    # Core field existence and types
    assert isinstance(output, RunOutput)
    assert isinstance(output.content, str)
    assert output.content == "Hello!"
    assert isinstance(output.messages, list)
    assert isinstance(output.model, str)
    assert isinstance(output.run_id, str)
    assert isinstance(output.session_id, str)
    assert output.status == RunStatus.completed

    # metrics may be Metrics or None, but should exist as an attribute
    assert hasattr(output, "metrics")
    if output.metrics is not None:
      assert isinstance(output.metrics, Metrics)

  async def test_run_output_with_tools(self):
    """Tool executions must be recorded as ToolExecution objects in output.tools."""
    model = MockModel(side_effect=_make_tool_side_effect())
    agent = Agent(model=model, tools=[greet], config=_NO_TRACE)  # type: ignore[arg-type]
    output = await agent.arun("Greet Alice")

    assert output.status == RunStatus.completed
    assert output.content == "Hello Alice!"

    # tools list must be populated
    assert output.tools is not None
    assert len(output.tools) >= 1

    te = output.tools[0]
    assert isinstance(te, ToolExecution)
    assert te.tool_name == "greet"
    assert te.tool_args == {"name": "Alice"}
    assert te.result is not None
    assert "Hi, Alice!" in te.result
    assert te.tool_call_id == "call_1"

  async def test_middleware_still_wraps_execution(self):
    """agent.use(middleware) must inject middleware into the execution path."""

    class FlagMiddleware:
      def __init__(self):
        self.called = False

      async def __call__(self, context, next_handler):
        self.called = True
        return await next_handler(context)

    mw = FlagMiddleware()
    model = MockModel(responses=["Done"])
    agent = Agent(model=model, config=_NO_TRACE)  # type: ignore[arg-type]
    agent.use(mw)

    output = await agent.arun("Run with middleware")

    assert mw.called is True
    assert output.status == RunStatus.completed
    assert output.content == "Done"

  async def test_multi_turn_with_messages(self):
    """Passing messages= from a prior run must carry conversation history."""
    model = MockModel(responses=["First answer", "Second answer"])
    agent = Agent(model=model, config=_NO_TRACE)  # type: ignore[arg-type]

    output1 = await agent.arun("Turn 1")
    assert output1.content == "First answer"
    assert output1.messages is not None

    output2 = await agent.arun("Turn 2", messages=output1.messages)
    assert output2.content == "Second answer"
    assert output2.messages is not None

    # Second output must include messages from the first turn plus the new ones
    assert len(output2.messages) > len(output1.messages)

  async def test_streaming_yields_events(self):
    """arun_stream() must yield RunStartedEvent and RunCompletedEvent."""
    model = MockModel(responses=["Streamed answer"])
    agent = Agent(model=model, config=_NO_TRACE)  # type: ignore[arg-type]

    events = []
    async for event in agent.arun_stream("Hello"):
      events.append(event)

    event_types = [type(e).__name__ for e in events]
    assert "RunStartedEvent" in event_types
    assert "RunCompletedEvent" in event_types

  async def test_tool_execution_in_output(self):
    """When a tool is invoked, ToolExecution in output.tools has correct name and result."""

    @tool
    def double(x: int) -> str:
      """Double a number."""
      return str(x * 2)

    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      response = MagicMock()
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None
      if call_count == 1:
        response.content = ""
        response.tool_calls = [
          {
            "id": "call_d",
            "type": "function",
            "function": {"name": "double", "arguments": '{"x": 21}'},
          }
        ]
      else:
        response.content = "The answer is 42"
        response.tool_calls = []
      return response

    model = MockModel(side_effect=side_effect)
    agent = Agent(model=model, tools=[double], config=_NO_TRACE)  # type: ignore[arg-type]
    output = await agent.arun("What is 21 doubled?")

    assert output.status == RunStatus.completed
    assert output.tools is not None
    assert len(output.tools) == 1

    te = output.tools[0]
    assert te.tool_name == "double"
    assert te.tool_args == {"x": 21}
    assert te.result == "42"
