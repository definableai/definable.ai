"""
Behavioral tests: Parallel tool dispatch and sequential tool ordering.

Strategy:
  - MockModel with side_effect to simulate tool calls
  - Assert on OUTCOMES: timing for parallel dispatch, ordering for sequential
  - Verify error isolation between parallel tools

Covers:
  - Parallel tools execute concurrently (timing-based assertion)
  - Sequential tools execute in declaration order
  - Error in one parallel tool does not crash the agent
  - Mixed parallel and sequential tools all execute
  - Single tool call basic success
  - Tool execution entries appear in RunOutput.tools
"""

import asyncio
import time
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


def _make_side_effect(tool_names, final_content="Done"):
  """Build a side_effect that returns tool_calls on the first invocation,
  then a plain text response on subsequent calls."""
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
        {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": "{}"}} for i, name in enumerate(tool_names, start=1)
      ]
    else:
      response.content = final_content
      response.tool_calls = []
    return response

  return side_effect


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def slow_tool_a() -> str:
  """Tool A that sleeps."""
  await asyncio.sleep(0.5)
  return "result_a"


@tool
async def slow_tool_b() -> str:
  """Tool B that sleeps."""
  await asyncio.sleep(0.5)
  return "result_b"


execution_order: list[str] = []


@tool
def ordered_tool_x() -> str:
  """Sequential tool X."""
  execution_order.append("x")
  return "x_done"


@tool
def ordered_tool_y() -> str:
  """Sequential tool Y."""
  execution_order.append("y")
  return "y_done"


@tool
def failing_tool() -> str:
  """Tool that always raises."""
  raise RuntimeError("intentional failure")


@tool
def succeeding_tool() -> str:
  """Tool that always succeeds."""
  return "success"


@tool
async def parallel_fast() -> str:
  """Fast parallel tool."""
  return "fast_done"


@tool
def sequential_slow() -> str:
  """Sequential tool."""
  return "seq_done"


@tool
def single_tool() -> str:
  """A simple tool."""
  return "single_result"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestParallelToolDispatch:
  """Verify that the agent dispatches non-sequential tools in parallel."""

  @pytest.mark.asyncio
  async def test_parallel_tools_execute_concurrently(self):
    """Two tools that each sleep 0.5s should complete in < 1.0s when dispatched in parallel."""
    model = MockModel(side_effect=_make_side_effect(["slow_tool_a", "slow_tool_b"]))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[slow_tool_a, slow_tool_b],
      config=NO_TRACE,
    )

    start = time.monotonic()
    output = await agent.arun("Run both tools")
    elapsed = time.monotonic() - start

    # If executed in parallel, total time should be ~0.5s, not ~1.0s
    assert elapsed < 1.0, f"Parallel execution took {elapsed:.2f}s — expected < 1.0s"
    assert output.content == "Done"

  @pytest.mark.asyncio
  async def test_sequential_tools_execute_in_order(self):
    """Tools marked sequential=True should execute in the order they were called."""
    # Mark both tools as sequential
    ordered_tool_x.sequential = True
    ordered_tool_y.sequential = True

    try:
      execution_order.clear()

      model = MockModel(side_effect=_make_side_effect(["ordered_tool_x", "ordered_tool_y"]))
      agent = Agent(
        model=model,  # type: ignore[arg-type]
        tools=[ordered_tool_x, ordered_tool_y],
        config=NO_TRACE,
      )

      output = await agent.arun("Run tools in order")

      assert execution_order == ["x", "y"], f"Expected ['x', 'y'], got {execution_order}"
      assert output.content == "Done"
    finally:
      ordered_tool_x.sequential = False
      ordered_tool_y.sequential = False

  @pytest.mark.asyncio
  async def test_parallel_tool_error_isolated(self):
    """When one parallel tool raises, the other should still succeed and the agent should not crash."""
    model = MockModel(side_effect=_make_side_effect(["failing_tool", "succeeding_tool"]))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[failing_tool, succeeding_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Run both tools")

    # Agent should not crash — it gets the error as a tool result message
    assert output.content is not None
    # The succeeding tool should have executed
    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "succeeding_tool" in tool_names
    assert "failing_tool" in tool_names

  @pytest.mark.asyncio
  async def test_mixed_parallel_and_sequential(self):
    """Mix of parallel and sequential tools should all execute successfully."""
    sequential_slow.sequential = True

    try:
      model = MockModel(side_effect=_make_side_effect(["parallel_fast", "slow_tool_a", "sequential_slow"]))
      agent = Agent(
        model=model,  # type: ignore[arg-type]
        tools=[parallel_fast, slow_tool_a, sequential_slow],
        config=NO_TRACE,
      )

      output = await agent.arun("Run all three tools")

      tool_names = [t.tool_name for t in (output.tools or [])]
      assert "parallel_fast" in tool_names
      assert "slow_tool_a" in tool_names
      assert "sequential_slow" in tool_names
      assert output.content == "Done"
    finally:
      sequential_slow.sequential = False

  @pytest.mark.asyncio
  async def test_single_tool_works(self):
    """A single tool call should succeed without issues."""
    model = MockModel(side_effect=_make_side_effect(["single_tool"]))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[single_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Use the tool")

    tool_names = [t.tool_name for t in (output.tools or [])]
    assert "single_tool" in tool_names
    assert output.content == "Done"

  @pytest.mark.asyncio
  async def test_tools_results_in_output(self):
    """After tools execute, RunOutput.tools should contain ToolExecution entries."""
    model = MockModel(side_effect=_make_side_effect(["slow_tool_a", "succeeding_tool"]))
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[slow_tool_a, succeeding_tool],
      config=NO_TRACE,
    )

    output = await agent.arun("Run both tools")

    assert output.tools is not None, "RunOutput.tools should not be None after tool execution"
    assert len(output.tools) >= 2, f"Expected at least 2 ToolExecution entries, got {len(output.tools)}"

    tool_names = [t.tool_name for t in output.tools]
    assert "slow_tool_a" in tool_names
    assert "succeeding_tool" in tool_names

    # Each ToolExecution should have a tool_name and call id
    for te in output.tools:
      assert te.tool_name is not None
      assert te.tool_call_id is not None
