"""Tests for the workflow multi-step orchestration system."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.event_bus import EventBus
from definable.agent.workflow.condition import Condition
from definable.agent.workflow.context import StepInput, StepOutput, StepStatus, WorkflowOutput
from definable.agent.workflow.events import (
  BaseWorkflowEvent,
  LoopIterationEvent,
  StepCompletedEvent,
  StepErrorEvent,
  StepSkippedEvent,
  StepStartedEvent,
  WorkflowRunCompletedEvent,
  WorkflowRunErrorEvent,
  WorkflowRunStartedEvent,
)
from definable.agent.workflow.loop import Loop
from definable.agent.workflow.parallel import Parallel
from definable.agent.workflow.router import Router
from definable.agent.workflow.step import BaseStep, Step, Steps, _default_input_builder, _normalize_step
from definable.agent.workflow.workflow import Workflow


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(name: str = "mock_agent", response: str = "mock response") -> MagicMock:
  """Create a mock agent that returns a fixed response from arun()."""
  agent = MagicMock()
  agent.name = name
  agent.instructions = f"You are {name}."

  run_output = MagicMock()
  run_output.content = response
  agent.arun = AsyncMock(return_value=run_output)
  return agent


def _make_failing_agent(name: str = "failing", error_msg: str = "agent failed") -> MagicMock:
  """Create a mock agent that raises an exception from arun()."""
  agent = MagicMock()
  agent.name = name
  agent.arun = AsyncMock(side_effect=RuntimeError(error_msg))
  return agent


def _make_mock_team(name: str = "mock_team", response: str = "team response") -> MagicMock:
  """Create a mock team that returns a fixed response from arun()."""
  team = MagicMock()
  team.name = name

  run_output = MagicMock()
  run_output.content = response
  team.arun = AsyncMock(return_value=run_output)
  return team


# ---------------------------------------------------------------------------
# StepStatus tests
# ---------------------------------------------------------------------------


class TestStepStatus:
  def test_values(self):
    assert StepStatus.pending == "pending"
    assert StepStatus.running == "running"  # type: ignore[unreachable]
    assert StepStatus.completed == "completed"
    assert StepStatus.failed == "failed"
    assert StepStatus.skipped == "skipped"

  def test_is_string_enum(self):
    assert isinstance(StepStatus.completed, str)


# ---------------------------------------------------------------------------
# StepInput tests
# ---------------------------------------------------------------------------


class TestStepInput:
  def test_default_construction(self):
    si = StepInput()
    assert si.input is None
    assert si.previous_step_content is None
    assert si.previous_step_outputs == {}
    assert si.additional_data == {}
    assert si.session_state == {}

  def test_get_step_output(self):
    output = StepOutput(step_name="step1", content="hello")
    si = StepInput(previous_step_outputs={"step1": output})
    assert si.get_step_output("step1") is output
    assert si.get_step_output("nonexistent") is None

  def test_get_step_content(self):
    output = StepOutput(step_name="step1", content="hello")
    si = StepInput(previous_step_outputs={"step1": output})
    assert si.get_step_content("step1") == "hello"
    assert si.get_step_content("nonexistent") is None

  def test_get_last_step_content(self):
    si = StepInput(previous_step_content="last output")
    assert si.get_last_step_content() == "last output"

  def test_get_all_previous_content(self):
    si = StepInput(
      previous_step_outputs={
        "a": StepOutput(step_name="a", content="content_a"),
        "b": StepOutput(step_name="b", content="content_b"),
      }
    )
    result = si.get_all_previous_content()
    assert result == {"a": "content_a", "b": "content_b"}


# ---------------------------------------------------------------------------
# StepOutput tests
# ---------------------------------------------------------------------------


class TestStepOutput:
  def test_default_construction(self):
    so = StepOutput()
    assert so.step_name == ""
    assert so.step_type == "step"
    assert so.content is None
    assert so.status == StepStatus.completed
    assert so.success is True
    assert so.error is None
    assert so.stop is False
    assert so.steps == []

  def test_to_dict(self):
    so = StepOutput(step_name="test", content="output", duration_ms=42.5)
    d = so.to_dict()
    assert d["step_name"] == "test"
    assert d["content"] == "output"
    assert d["status"] == "completed"
    assert d["duration_ms"] == 42.5

  def test_to_dict_with_nested_steps(self):
    inner = StepOutput(step_name="inner", content="nested")
    outer = StepOutput(step_name="outer", steps=[inner])
    d = outer.to_dict()
    assert len(d["steps"]) == 1
    assert d["steps"][0]["step_name"] == "inner"

  def test_auto_generated_step_id(self):
    so1 = StepOutput()
    so2 = StepOutput()
    assert so1.step_id != so2.step_id
    assert len(so1.step_id) == 8


# ---------------------------------------------------------------------------
# WorkflowOutput tests
# ---------------------------------------------------------------------------


class TestWorkflowOutput:
  def test_get_step_output(self):
    so = StepOutput(step_name="researcher", content="findings")
    wo = WorkflowOutput(step_outputs=[so])
    assert wo.get_step_output("researcher") is so
    assert wo.get_step_output("nonexistent") is None

  def test_get_step_output_nested(self):
    inner = StepOutput(step_name="inner", content="nested content")
    outer = StepOutput(step_name="outer", steps=[inner])
    wo = WorkflowOutput(step_outputs=[outer])
    assert wo.get_step_output("inner") is inner

  def test_get_step_content(self):
    so = StepOutput(step_name="writer", content="article text")
    wo = WorkflowOutput(step_outputs=[so])
    assert wo.get_step_content("writer") == "article text"
    assert wo.get_step_content("nonexistent") is None


# ---------------------------------------------------------------------------
# _default_input_builder tests
# ---------------------------------------------------------------------------


class TestDefaultInputBuilder:
  def test_input_only(self):
    si = StepInput(input="hello")
    assert _default_input_builder(si) == "hello"

  def test_previous_step_only(self):
    si = StepInput(previous_step_content="previous")
    result = _default_input_builder(si)
    assert "Previous step output:" in result
    assert "previous" in result

  def test_both_input_and_previous(self):
    si = StepInput(input="hello", previous_step_content="previous")
    result = _default_input_builder(si)
    assert "hello" in result
    assert "Previous step output:" in result
    assert "previous" in result

  def test_empty_input(self):
    si = StepInput()
    assert _default_input_builder(si) == ""


# ---------------------------------------------------------------------------
# _normalize_step tests
# ---------------------------------------------------------------------------


class TestNormalizeStep:
  def test_basestep_passthrough(self):
    step = Step(name="test", executor=lambda x: "ok")
    assert _normalize_step(step) is step

  def test_callable_wrapped(self):
    async def my_fn(step_input: StepInput) -> str:
      return "hello"

    result = _normalize_step(my_fn)
    assert isinstance(result, Step)
    assert result.name == "my_fn"
    assert result.executor is my_fn

  def test_invalid_type_raises(self):
    with pytest.raises(TypeError, match="Invalid step type"):
      _normalize_step(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------


class TestStep:
  def test_step_type(self):
    step = Step(name="test", executor=lambda x: "ok")
    assert step.step_type == "step"

  @pytest.mark.asyncio
  async def test_agent_execution(self):
    agent = _make_mock_agent("researcher", "research findings")
    step = Step(name="researcher", agent=agent)
    si = StepInput(input="research quantum computing")

    result = await step.execute(si)

    assert result.success is True
    assert result.status == StepStatus.completed
    assert result.content == "research findings"
    assert result.step_name == "researcher"
    assert result.duration_ms > 0
    agent.arun.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_team_execution(self):
    team = _make_mock_team("research_team", "team findings")
    step = Step(name="team_step", team=team)
    si = StepInput(input="analyze data")

    result = await step.execute(si)

    assert result.success is True
    assert result.content == "team findings"
    team.arun.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_executor_async(self):
    async def my_executor(step_input: StepInput) -> str:
      return f"processed: {step_input.input}"

    step = Step(name="custom", executor=my_executor)
    si = StepInput(input="data")

    result = await step.execute(si)

    assert result.success is True
    assert result.content == "processed: data"

  @pytest.mark.asyncio
  async def test_executor_sync(self):
    def my_executor(step_input: StepInput) -> str:
      return f"sync: {step_input.input}"

    step = Step(name="sync_step", executor=my_executor)
    si = StepInput(input="test")

    result = await step.execute(si)

    assert result.success is True
    assert result.content == "sync: test"

  def test_no_executor_raises(self):
    with pytest.raises(ValueError, match="exactly one of agent, team, or executor"):
      Step(name="empty")

  @pytest.mark.asyncio
  async def test_failure_returns_failed_output(self):
    agent = _make_failing_agent("broken", "kaboom")
    step = Step(name="broken", agent=agent)
    si = StepInput(input="test")

    result = await step.execute(si)

    assert result.success is False
    assert result.status == StepStatus.failed
    assert "kaboom" in (result.error or "")

  @pytest.mark.asyncio
  async def test_retry_on_failure(self):
    call_count = 0

    async def flaky_executor(step_input: StepInput) -> str:
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise RuntimeError("transient error")
      return "success"

    step = Step(name="flaky", executor=flaky_executor, retries=2)
    si = StepInput(input="test")

    result = await step.execute(si)

    assert result.success is True
    assert result.content == "success"
    assert call_count == 3

  @pytest.mark.asyncio
  async def test_retry_exhaustion(self):
    async def always_fail(step_input: StepInput) -> str:
      raise RuntimeError("permanent failure")

    step = Step(name="fail", executor=always_fail, retries=2)
    si = StepInput(input="test")

    result = await step.execute(si)

    assert result.success is False
    assert "permanent failure" in (result.error or "")

  @pytest.mark.asyncio
  async def test_custom_input_builder(self):
    agent = _make_mock_agent("agent", "response")
    step = Step(
      name="custom_input",
      agent=agent,
      input_builder=lambda si: f"CUSTOM: {si.input}",
    )
    si = StepInput(input="hello")

    await step.execute(si)

    agent.arun.assert_awaited_once_with("CUSTOM: hello")

  @pytest.mark.asyncio
  async def test_timeout(self):
    async def slow_executor(step_input: StepInput) -> str:
      await asyncio.sleep(10)
      return "slow"

    step = Step(name="slow", executor=slow_executor, timeout=0.01)
    si = StepInput(input="test")

    result = await step.execute(si)

    assert result.success is False
    # asyncio.TimeoutError

  @pytest.mark.asyncio
  async def test_events_emitted(self):
    events: List[Any] = []
    bus = EventBus()
    bus.on(object, lambda e: events.append(e))

    agent = _make_mock_agent("agent", "result")
    step = Step(name="test_step", agent=agent)
    si = StepInput(input="hello")

    await step.execute(si, event_bus=bus, run_id="run1", workflow_id="wf1")

    started_events = [e for e in events if isinstance(e, StepStartedEvent)]
    completed_events = [e for e in events if isinstance(e, StepCompletedEvent)]
    assert len(started_events) == 1
    assert started_events[0].step_name == "test_step"
    assert started_events[0].run_id == "run1"
    assert len(completed_events) == 1
    assert completed_events[0].content == "result"

  @pytest.mark.asyncio
  async def test_error_event_emitted(self):
    events: List[Any] = []
    bus = EventBus()
    bus.on(object, lambda e: events.append(e))

    agent = _make_failing_agent("broken", "error msg")
    step = Step(name="broken_step", agent=agent)
    si = StepInput(input="hello")

    await step.execute(si, event_bus=bus, run_id="run1")

    error_events = [e for e in events if isinstance(e, StepErrorEvent)]
    assert len(error_events) == 1
    assert "error msg" in error_events[0].error


# ---------------------------------------------------------------------------
# Steps (sequential) tests
# ---------------------------------------------------------------------------


class TestSteps:
  def test_step_type(self):
    seq = Steps(name="seq")
    assert seq.step_type == "steps"

  @pytest.mark.asyncio
  async def test_sequential_execution(self):
    agent1 = _make_mock_agent("a1", "output1")
    agent2 = _make_mock_agent("a2", "output2")

    seq = Steps(
      steps=[
        Step(name="step1", agent=agent1),
        Step(name="step2", agent=agent2),
      ]
    )
    si = StepInput(input="hello")

    result = await seq.execute(si)

    assert result.success is True
    assert result.content == "output2"  # Last step's content
    assert len(result.steps) == 2
    assert result.steps[0].step_name == "step1"
    assert result.steps[1].step_name == "step2"

  @pytest.mark.asyncio
  async def test_context_chaining(self):
    """Second step receives first step's output as context."""
    prompts_received: List[str] = []

    async def capture_executor(step_input: StepInput) -> str:
      prompt = _default_input_builder(step_input)
      prompts_received.append(prompt)
      return f"result from {len(prompts_received)}"

    seq = Steps(
      steps=[
        Step(name="step1", executor=capture_executor),
        Step(name="step2", executor=capture_executor),
      ]
    )
    si = StepInput(input="original input")

    await seq.execute(si)

    assert len(prompts_received) == 2
    assert "original input" in prompts_received[0]
    assert "result from 1" in prompts_received[1]

  @pytest.mark.asyncio
  async def test_stops_on_failure(self):
    agent1 = _make_failing_agent("broken", "fail")
    agent2 = _make_mock_agent("a2", "output2")

    seq = Steps(
      steps=[
        Step(name="step1", agent=agent1),
        Step(name="step2", agent=agent2),
      ]
    )
    si = StepInput(input="hello")

    result = await seq.execute(si)

    assert result.success is False
    assert len(result.steps) == 1  # Only step1 ran
    agent2.arun.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_early_termination_via_stop_flag(self):
    async def stop_executor(step_input: StepInput) -> StepOutput:
      return StepOutput(step_name="stopper", content="stopping", stop=True)

    # Create a Step that wraps this — but the executor returns StepOutput directly.
    # We need to handle this at the Step level. Let's use a different approach.
    agent = _make_mock_agent("agent", "agent output")

    # Manually create StepOutput with stop=True
    step1_output = StepOutput(step_name="step1", content="continue", stop=True)
    step1 = MagicMock(spec=BaseStep)
    step1.name = "step1"
    step1._id = "id1"
    step1.step_type = "step"
    step1.execute = AsyncMock(return_value=step1_output)

    step2 = Step(name="step2", agent=agent)

    seq = Steps(steps=[step1, step2])
    si = StepInput(input="hello")

    result = await seq.execute(si)

    assert result.stop is True
    assert len(result.steps) == 1
    agent.arun.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_empty_steps(self):
    seq = Steps(steps=[])
    si = StepInput(input="hello")

    result = await seq.execute(si)

    assert result.success is True
    assert result.content is None
    assert len(result.steps) == 0

  @pytest.mark.asyncio
  async def test_default_name(self):
    seq = Steps(steps=[])
    si = StepInput()
    result = await seq.execute(si)
    assert result.step_name == "sequence"


# ---------------------------------------------------------------------------
# Parallel tests
# ---------------------------------------------------------------------------


class TestParallel:
  def test_step_type(self):
    par = Parallel()
    assert par.step_type == "parallel"

  def test_default_name(self):
    par = Parallel()
    assert par.name == "parallel"

  @pytest.mark.asyncio
  async def test_parallel_execution(self):
    agent1 = _make_mock_agent("a1", "output1")
    agent2 = _make_mock_agent("a2", "output2")

    par = Parallel(
      name="par",
      steps=[
        Step(name="step1", agent=agent1),
        Step(name="step2", agent=agent2),
      ],
    )
    si = StepInput(input="hello")

    result = await par.execute(si)

    assert result.success is True
    assert len(result.steps) == 2
    assert "[step1]: output1" in (result.content or "")
    assert "[step2]: output2" in (result.content or "")

  @pytest.mark.asyncio
  async def test_parallel_all_receive_same_input(self):
    """All parallel steps receive the same original input."""
    inputs_received: List[str] = []

    async def capture_executor(step_input: StepInput) -> str:
      inputs_received.append(step_input.input or "")
      return "done"

    par = Parallel(
      steps=[
        Step(name="s1", executor=capture_executor),
        Step(name="s2", executor=capture_executor),
        Step(name="s3", executor=capture_executor),
      ]
    )
    si = StepInput(input="shared input")

    await par.execute(si)

    assert len(inputs_received) == 3
    assert all(inp == "shared input" for inp in inputs_received)

  @pytest.mark.asyncio
  async def test_partial_failure(self):
    agent_ok = _make_mock_agent("ok", "success")
    agent_fail = _make_failing_agent("fail", "error")

    par = Parallel(
      steps=[
        Step(name="ok_step", agent=agent_ok),
        Step(name="fail_step", agent=agent_fail),
      ]
    )
    si = StepInput(input="hello")

    result = await par.execute(si)

    assert result.success is False
    assert len(result.steps) == 2
    ok_step = next(s for s in result.steps if s.step_name == "ok_step")
    fail_step = next(s for s in result.steps if s.step_name == "fail_step")
    assert ok_step.success is True
    assert fail_step.success is False

  @pytest.mark.asyncio
  async def test_max_concurrency(self):
    """Semaphore limits concurrent execution."""
    active = 0
    max_active = 0

    async def track_executor(step_input: StepInput) -> str:
      nonlocal active, max_active
      active += 1
      max_active = max(max_active, active)
      await asyncio.sleep(0.01)
      active -= 1
      return "done"

    par = Parallel(
      steps=[Step(name=f"s{i}", executor=track_executor) for i in range(5)],
      max_concurrency=2,
    )
    si = StepInput(input="test")

    await par.execute(si)

    assert max_active <= 2

  @pytest.mark.asyncio
  async def test_exception_in_gather_handled(self):
    """An exception raised directly (not returned as StepOutput) is caught."""

    # Create a step mock that raises when execute is called
    step = MagicMock(spec=BaseStep)
    step.name = "explosive"
    step._id = "ex1"
    step.step_type = "step"
    step.execute = AsyncMock(side_effect=RuntimeError("boom"))

    par = Parallel(steps=[step])
    si = StepInput(input="test")

    result = await par.execute(si)

    assert result.success is False
    assert len(result.steps) == 1
    assert "boom" in (result.steps[0].error or "")


# ---------------------------------------------------------------------------
# Loop tests
# ---------------------------------------------------------------------------


class TestLoop:
  def test_step_type(self):
    loop = Loop(steps=[Step(name="s", executor=lambda x: "ok")])
    assert loop.step_type == "loop"

  def test_default_name(self):
    loop = Loop(steps=[Step(name="s", executor=lambda x: "ok")])
    assert loop.name == "loop"

  def test_empty_steps_raises(self):
    with pytest.raises(ValueError, match="at least one step"):
      Loop(name="empty")

  def test_zero_max_iterations_raises(self):
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
      Loop(steps=[Step(name="s", executor=lambda x: "ok")], max_iterations=0)

  @pytest.mark.asyncio
  async def test_basic_loop(self):
    agent = _make_mock_agent("agent", "iteration output")

    loop = Loop(
      name="test_loop",
      steps=[Step(name="step1", agent=agent)],
      max_iterations=3,
    )
    si = StepInput(input="hello")

    result = await loop.execute(si)

    assert result.success is True
    assert len(result.steps) == 3  # Ran 3 iterations
    assert agent.arun.await_count == 3

  @pytest.mark.asyncio
  async def test_end_condition(self):
    call_count = 0

    async def counting_executor(step_input: StepInput) -> str:
      nonlocal call_count
      call_count += 1
      return f"iteration {call_count}"

    loop = Loop(
      name="counted_loop",
      steps=[Step(name="counter", executor=counting_executor)],
      end_condition=lambda outputs: len(outputs) >= 2,
      max_iterations=10,
    )
    si = StepInput(input="test")

    result = await loop.execute(si)

    assert result.success is True
    assert len(result.steps) == 2  # Stopped at 2, not 10

  @pytest.mark.asyncio
  async def test_async_end_condition(self):
    async def async_condition(outputs: List[StepOutput]) -> bool:
      return len(outputs) >= 2

    loop = Loop(
      name="async_loop",
      steps=[Step(name="step", executor=AsyncMock(return_value="ok"))],
      end_condition=async_condition,  # type: ignore[arg-type]
      max_iterations=10,
    )
    si = StepInput(input="test")

    result = await loop.execute(si)

    assert len(result.steps) == 2

  @pytest.mark.asyncio
  async def test_stops_on_failure(self):
    agent = _make_failing_agent("fail", "error")

    loop = Loop(
      name="failing_loop",
      steps=[Step(name="step", agent=agent)],
      max_iterations=5,
    )
    si = StepInput(input="test")

    result = await loop.execute(si)

    assert result.success is False
    assert len(result.steps) == 1  # Only first iteration

  @pytest.mark.asyncio
  async def test_iteration_events(self):
    events: List[Any] = []
    bus = EventBus()
    bus.on(object, lambda e: events.append(e))

    agent = _make_mock_agent("agent", "output")

    loop = Loop(
      name="event_loop",
      steps=[Step(name="step", agent=agent)],
      max_iterations=2,
    )
    si = StepInput(input="test")

    await loop.execute(si, event_bus=bus)

    iteration_events = [e for e in events if isinstance(e, LoopIterationEvent)]
    assert len(iteration_events) == 2
    assert iteration_events[0].iteration == 0
    assert iteration_events[1].iteration == 1

  @pytest.mark.asyncio
  async def test_context_chains_across_iterations(self):
    """Each iteration receives the previous iteration's output."""
    inputs_seen: List[Optional[str]] = []

    async def capture_executor(step_input: StepInput) -> str:
      inputs_seen.append(step_input.previous_step_content)
      return f"iter_{len(inputs_seen)}"

    loop = Loop(
      name="chain_loop",
      steps=[Step(name="step", executor=capture_executor)],
      max_iterations=3,
    )
    si = StepInput(input="start")

    await loop.execute(si)

    assert len(inputs_seen) == 3
    assert inputs_seen[0] is None  # First iteration has no previous
    # Second iteration gets first's output (wrapped in Steps, so content from Steps output)
    assert inputs_seen[1] is not None


# ---------------------------------------------------------------------------
# Condition tests
# ---------------------------------------------------------------------------


class TestCondition:
  def test_step_type(self):
    cond = Condition()
    assert cond.step_type == "condition"

  @pytest.mark.asyncio
  async def test_true_branch(self):
    agent = _make_mock_agent("true_agent", "true output")

    cond = Condition(
      name="gate",
      condition=lambda si: True,
      true_steps=Step(name="true_step", agent=agent),
      false_steps=Step(name="false_step", agent=_make_mock_agent("f", "false")),
    )
    si = StepInput(input="test")

    result = await cond.execute(si)

    assert result.success is True
    assert result.content == "true output"
    assert len(result.steps) == 1

  @pytest.mark.asyncio
  async def test_false_branch(self):
    true_agent = _make_mock_agent("true", "true output")
    false_agent = _make_mock_agent("false", "false output")

    cond = Condition(
      name="gate",
      condition=lambda si: False,
      true_steps=Step(name="true_step", agent=true_agent),
      false_steps=Step(name="false_step", agent=false_agent),
    )
    si = StepInput(input="test")

    result = await cond.execute(si)

    assert result.content == "false output"
    true_agent.arun.assert_not_awaited()
    false_agent.arun.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_condition_based_on_previous_step(self):
    agent = _make_mock_agent("agent", "output")

    cond = Condition(
      name="quality_gate",
      condition=lambda si: "PASS" in (si.get_last_step_content() or ""),
      true_steps=Step(name="publish", agent=agent),
    )
    si = StepInput(previous_step_content="Quality: PASS")

    result = await cond.execute(si)

    assert result.success is True
    assert result.content == "output"

  @pytest.mark.asyncio
  async def test_async_condition(self):
    agent = _make_mock_agent("agent", "output")

    async def async_check(si: StepInput) -> bool:
      return si.input == "yes"

    cond = Condition(
      name="async_gate",
      condition=async_check,  # type: ignore[arg-type]
      true_steps=Step(name="true_step", agent=agent),
    )
    si = StepInput(input="yes")

    result = await cond.execute(si)

    assert result.content == "output"

  @pytest.mark.asyncio
  async def test_no_branch_skips(self):
    events: List[Any] = []
    bus = EventBus()
    bus.on(object, lambda e: events.append(e))

    cond = Condition(
      name="no_else",
      condition=lambda si: False,
      true_steps=Step(name="true", agent=_make_mock_agent("a", "x")),
      false_steps=None,
    )
    si = StepInput(input="test")

    result = await cond.execute(si, event_bus=bus)

    assert result.status == StepStatus.skipped
    assert result.success is True

    skip_events = [e for e in events if isinstance(e, StepSkippedEvent)]
    assert len(skip_events) == 1

  @pytest.mark.asyncio
  async def test_no_condition_raises(self):
    cond = Condition(name="broken")
    si = StepInput(input="test")

    with pytest.raises(ValueError, match="no condition callable"):
      await cond.execute(si)

  @pytest.mark.asyncio
  async def test_list_of_steps_as_branch(self):
    agent1 = _make_mock_agent("a1", "output1")
    agent2 = _make_mock_agent("a2", "output2")

    cond = Condition(
      name="multi_branch",
      condition=lambda si: True,
      true_steps=[
        Step(name="step1", agent=agent1),
        Step(name="step2", agent=agent2),
      ],
    )
    si = StepInput(input="test")

    result = await cond.execute(si)

    assert result.success is True
    assert result.content == "output2"  # Last step's content


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestRouter:
  def test_step_type(self):
    router = Router()
    assert router.step_type == "router"

  @pytest.mark.asyncio
  async def test_single_route(self):
    tech_agent = _make_mock_agent("tech", "tech response")
    gen_agent = _make_mock_agent("general", "general response")

    router = Router(
      name="support",
      selector=lambda si: "technical",
      routes={
        "technical": Step(name="tech_support", agent=tech_agent),
        "general": Step(name="general_support", agent=gen_agent),
      },
    )
    si = StepInput(input="bug report")

    result = await router.execute(si)

    assert result.success is True
    assert result.content == "tech response"
    tech_agent.arun.assert_awaited_once()
    gen_agent.arun.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_dynamic_routing(self):
    tech_agent = _make_mock_agent("tech", "tech response")
    gen_agent = _make_mock_agent("general", "general response")

    router = Router(
      name="support",
      selector=lambda si: "technical" if "bug" in (si.input or "") else "general",
      routes={
        "technical": Step(name="tech", agent=tech_agent),
        "general": Step(name="general", agent=gen_agent),
      },
    )

    # Route to technical
    result1 = await router.execute(StepInput(input="bug in login"))
    assert "tech response" in (result1.content or "")

    # Route to general
    result2 = await router.execute(StepInput(input="how to sign up"))
    assert "general response" in (result2.content or "")

  @pytest.mark.asyncio
  async def test_multiple_routes(self):
    agent1 = _make_mock_agent("a1", "output1")
    agent2 = _make_mock_agent("a2", "output2")

    router = Router(
      name="multi",
      selector=lambda si: ["route1", "route2"],
      routes={
        "route1": Step(name="r1", agent=agent1),
        "route2": Step(name="r2", agent=agent2),
      },
    )
    si = StepInput(input="test")

    result = await router.execute(si)

    assert result.success is True
    assert len(result.steps) == 2
    assert "[r1]: output1" in (result.content or "")
    assert "[r2]: output2" in (result.content or "")

  @pytest.mark.asyncio
  async def test_unknown_route(self):
    router = Router(
      name="broken",
      selector=lambda si: "nonexistent",
      routes={"valid": Step(name="v", agent=_make_mock_agent("a", "x"))},
    )
    si = StepInput(input="test")

    result = await router.execute(si)

    assert result.success is False
    assert "not found" in (result.steps[0].error or "")

  @pytest.mark.asyncio
  async def test_async_selector(self):
    agent = _make_mock_agent("agent", "response")

    async def async_selector(si: StepInput) -> str:
      return "route_a"

    router = Router(
      name="async",
      selector=async_selector,  # type: ignore[arg-type]
      routes={"route_a": Step(name="a", agent=agent)},
    )
    si = StepInput(input="test")

    result = await router.execute(si)

    assert result.content == "response"

  @pytest.mark.asyncio
  async def test_no_selector_raises(self):
    router = Router(name="broken")
    si = StepInput(input="test")

    with pytest.raises(ValueError, match="no selector callable"):
      await router.execute(si)

  @pytest.mark.asyncio
  async def test_empty_selection_skips(self):
    router = Router(
      name="empty",
      selector=lambda si: [],
      routes={"a": Step(name="a", agent=_make_mock_agent("a", "x"))},
    )
    si = StepInput(input="test")

    result = await router.execute(si)

    assert result.status == StepStatus.skipped
    assert result.success is True


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


class TestWorkflow:
  @pytest.mark.asyncio
  async def test_basic_workflow(self):
    agent1 = _make_mock_agent("researcher", "research findings")
    agent2 = _make_mock_agent("writer", "final article")

    workflow = Workflow(
      name="research-pipeline",
      steps=[
        Step(name="researcher", agent=agent1),
        Step(name="writer", agent=agent2),
      ],
    )

    result = await workflow.arun("Write about quantum computing")

    assert result.success is True
    assert result.content == "final article"
    assert result.workflow_name == "research-pipeline"
    assert len(result.step_outputs) == 2
    assert result.duration_ms > 0

  @pytest.mark.asyncio
  async def test_single_step_workflow(self):
    agent = _make_mock_agent("agent", "response")

    workflow = Workflow(
      name="simple",
      steps=Step(name="only_step", agent=agent),
    )

    result = await workflow.arun("hello")

    assert result.success is True
    assert result.content == "response"

  @pytest.mark.asyncio
  async def test_workflow_with_parallel(self):
    agent1 = _make_mock_agent("tech", "technical analysis")
    agent2 = _make_mock_agent("biz", "business analysis")

    workflow = Workflow(
      name="multi-analysis",
      steps=Parallel(
        name="analysis",
        steps=[
          Step(name="technical", agent=agent1),
          Step(name="business", agent=agent2),
        ],
      ),
    )

    result = await workflow.arun("analyze the market")

    assert result.success is True
    assert "technical analysis" in (result.content or "")
    assert "business analysis" in (result.content or "")

  @pytest.mark.asyncio
  async def test_workflow_with_condition(self):
    agent = _make_mock_agent("agent", "published")

    workflow = Workflow(
      name="conditional",
      steps=Condition(
        name="gate",
        condition=lambda si: True,
        true_steps=Step(name="publish", agent=agent),
      ),
    )

    result = await workflow.arun("test")

    assert result.success is True
    assert result.content == "published"

  @pytest.mark.asyncio
  async def test_workflow_with_loop(self):
    call_count = 0

    async def executor(si: StepInput) -> str:
      nonlocal call_count
      call_count += 1
      return f"iteration {call_count}"

    workflow = Workflow(
      name="looping",
      steps=Loop(
        name="improve",
        steps=[Step(name="generate", executor=executor)],
        end_condition=lambda outputs: len(outputs) >= 2,
        max_iterations=5,
      ),
    )

    result = await workflow.arun("test")

    assert result.success is True
    assert call_count == 2

  @pytest.mark.asyncio
  async def test_workflow_with_router(self):
    tech = _make_mock_agent("tech", "technical answer")
    general = _make_mock_agent("general", "general answer")

    workflow = Workflow(
      name="support",
      steps=Router(
        name="router",
        selector=lambda si: "tech" if "bug" in (si.input or "") else "general",
        routes={
          "tech": Step(name="tech_support", agent=tech),
          "general": Step(name="general_support", agent=general),
        },
      ),
    )

    result = await workflow.arun("I found a bug")

    assert result.success is True
    assert "technical answer" in (result.content or "")

  @pytest.mark.asyncio
  async def test_workflow_session_state(self):
    received_state: Dict[str, Any] = {}

    async def state_executor(si: StepInput) -> str:
      received_state.update(si.session_state)
      return "done"

    workflow = Workflow(
      name="stateful",
      steps=[Step(name="step", executor=state_executor)],
      session_state={"key1": "val1"},
    )

    result = await workflow.arun("test", session_state={"key2": "val2"})

    assert result.success is True
    assert received_state["key1"] == "val1"
    assert received_state["key2"] == "val2"
    assert result.session_state["key1"] == "val1"
    assert result.session_state["key2"] == "val2"

  @pytest.mark.asyncio
  async def test_workflow_failure(self):
    agent = _make_failing_agent("broken", "workflow error")

    workflow = Workflow(
      name="failing",
      steps=[Step(name="broken_step", agent=agent)],
    )

    result = await workflow.arun("test")

    assert result.success is False
    # Step errors propagate up through the sequence wrapper
    assert "workflow error" in (result.error or "")

  @pytest.mark.asyncio
  async def test_workflow_events(self):
    events: List[Any] = []
    agent = _make_mock_agent("agent", "output")

    workflow = Workflow(
      name="event-test",
      steps=[Step(name="step1", agent=agent)],
    )
    workflow.events.on(object, lambda e: events.append(e))

    await workflow.arun("test")

    started = [e for e in events if isinstance(e, WorkflowRunStartedEvent)]
    completed = [e for e in events if isinstance(e, WorkflowRunCompletedEvent)]
    step_started = [e for e in events if isinstance(e, StepStartedEvent)]
    step_completed = [e for e in events if isinstance(e, StepCompletedEvent)]

    assert len(started) == 1
    assert started[0].workflow_name == "event-test"
    assert len(completed) == 1
    assert completed[0].success is True
    assert len(step_started) == 1
    assert len(step_completed) == 1

  @pytest.mark.asyncio
  async def test_workflow_error_event(self):
    events: List[Any] = []

    workflow = Workflow(
      name="error-test",
      steps=42,  # type: ignore[arg-type]  # invalid steps type
    )
    workflow.events.on(object, lambda e: events.append(e))

    result = await workflow.arun("test")

    assert result.success is False
    error_events = [e for e in events if isinstance(e, WorkflowRunErrorEvent)]
    assert len(error_events) == 1

  @pytest.mark.asyncio
  async def test_workflow_callable_step(self):
    """Callables are auto-wrapped in Step."""

    async def my_step(si: StepInput) -> str:
      return f"processed: {si.input}"

    workflow = Workflow(
      name="callable",
      steps=[my_step],
    )

    result = await workflow.arun("test input")

    assert result.success is True
    assert result.content == "processed: test input"

  @pytest.mark.asyncio
  async def test_workflow_id_and_run_id(self):
    workflow = Workflow(name="id-test", steps=[Step(name="s", executor=AsyncMock(return_value="ok"))])

    result = await workflow.arun("test")

    assert result.workflow_id == workflow.id
    assert result.run_id  # Non-empty UUID
    assert result.workflow_id != result.run_id

  @pytest.mark.asyncio
  async def test_workflow_additional_data(self):
    received_data: Dict[str, Any] = {}

    async def capture_executor(si: StepInput) -> str:
      received_data.update(si.additional_data)
      return "done"

    workflow = Workflow(
      name="data-test",
      steps=[Step(name="step", executor=capture_executor)],
    )

    await workflow.arun("test", additional_data={"custom_key": "custom_val"})

    assert received_data["custom_key"] == "custom_val"


# ---------------------------------------------------------------------------
# Composition / nesting tests
# ---------------------------------------------------------------------------


class TestComposition:
  @pytest.mark.asyncio
  async def test_nested_steps_in_parallel(self):
    """Steps inside Parallel."""
    a1 = _make_mock_agent("a1", "o1")
    a2 = _make_mock_agent("a2", "o2")
    a3 = _make_mock_agent("a3", "o3")

    workflow = Workflow(
      name="nested",
      steps=Parallel(
        steps=[
          Steps(
            steps=[
              Step(name="seq1_s1", agent=a1),
              Step(name="seq1_s2", agent=a2),
            ]
          ),
          Step(name="standalone", agent=a3),
        ]
      ),
    )

    result = await workflow.arun("test")

    assert result.success is True

  @pytest.mark.asyncio
  async def test_condition_in_sequence(self):
    """Condition as part of a sequential workflow."""
    agent1 = _make_mock_agent("draft", "APPROVED: good article")
    publish_agent = _make_mock_agent("publish", "published!")

    workflow = Workflow(
      name="review-pipeline",
      steps=[
        Step(name="draft", agent=agent1),
        Condition(
          name="quality-gate",
          condition=lambda si: "APPROVED" in (si.get_last_step_content() or ""),
          true_steps=Step(name="publish", agent=publish_agent),
        ),
      ],
    )

    result = await workflow.arun("write about AI")

    assert result.success is True
    assert result.content == "published!"

  @pytest.mark.asyncio
  async def test_loop_with_condition(self):
    """Loop containing a condition."""
    iteration = 0

    async def draft_executor(si: StepInput) -> str:
      nonlocal iteration
      iteration += 1
      return "PASS" if iteration >= 2 else "FAIL"

    workflow = Workflow(
      name="iterate-review",
      steps=Loop(
        name="improve",
        steps=[
          Step(name="draft", executor=draft_executor),
          Condition(
            name="check",
            condition=lambda si: "PASS" in (si.get_last_step_content() or ""),
            true_steps=Step(name="publish", executor=AsyncMock(return_value="done")),
          ),
        ],
        end_condition=lambda outputs: any(s.content == "done" for o in outputs for s in (o.steps or []) if s.step_name == "publish"),
        max_iterations=5,
      ),
    )

    result = await workflow.arun("test")

    assert result.success is True


# ---------------------------------------------------------------------------
# Event tests (detailed)
# ---------------------------------------------------------------------------


class TestWorkflowEvents:
  def test_base_event_fields(self):
    event = BaseWorkflowEvent(workflow_id="wf1", workflow_name="test")
    assert event.workflow_id == "wf1"
    assert event.workflow_name == "test"
    assert event.created_at > 0

  def test_started_event(self):
    event = WorkflowRunStartedEvent(
      run_id="r1",
      workflow_id="wf1",
      workflow_name="test",
      step_count=3,
      step_names=["a", "b", "c"],
    )
    assert event.event == "workflow_run_started"
    assert event.step_count == 3
    assert event.step_names == ["a", "b", "c"]

  def test_completed_event(self):
    event = WorkflowRunCompletedEvent(
      run_id="r1",
      workflow_id="wf1",
      workflow_name="test",
      success=True,
      content="result",
      duration_ms=100.0,
    )
    assert event.event == "workflow_run_completed"
    assert event.success is True

  def test_error_event(self):
    event = WorkflowRunErrorEvent(
      run_id="r1",
      workflow_id="wf1",
      workflow_name="test",
      error="bad things",
    )
    assert event.event == "workflow_run_error"
    assert event.error == "bad things"

  def test_step_events(self):
    started = StepStartedEvent(step_id="s1", step_name="step1", step_type="step", step_index=0)
    assert started.event == "step_started"

    completed = StepCompletedEvent(step_id="s1", step_name="step1", step_type="step", content="done")
    assert completed.event == "step_completed"

    error = StepErrorEvent(step_id="s1", step_name="step1", step_type="step", error="fail")
    assert error.event == "step_error"

    skipped = StepSkippedEvent(step_id="s1", step_name="step1", step_type="condition", reason="no branch")
    assert skipped.event == "step_skipped"

  def test_loop_iteration_event(self):
    event = LoopIterationEvent(step_name="loop1", iteration=2, max_iterations=5, should_continue=True)
    assert event.event == "loop_iteration"
    assert event.iteration == 2


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
  def test_import_from_workflow_package(self):
    from definable.agent.workflow import (
      Condition,
      Loop,
      Parallel,
      Router,
      Step,
      Steps,
      Workflow,
    )

    assert Workflow is not None
    assert Step is not None
    assert Steps is not None
    assert Parallel is not None
    assert Loop is not None
    assert Condition is not None
    assert Router is not None

  def test_import_from_agent_package(self):
    from definable.agent import Workflow, Step, Steps, Parallel, Loop, Condition, Router  # noqa: F401

    assert Workflow is not None
    assert Step is not None
    assert Steps is not None
    assert Parallel is not None
    assert Loop is not None
    assert Condition is not None
    assert Router is not None

  def test_import_events_from_agent_events(self):
    from definable.agent.events import (  # noqa: F401
      WorkflowRunStartedEvent,
      WorkflowRunCompletedEvent,
      WorkflowRunErrorEvent,
      WorkflowStepStartedEvent,
      WorkflowStepCompletedEvent,
      WorkflowStepErrorEvent,
      WorkflowStepSkippedEvent,
      WorkflowLoopIterationEvent,
    )

    assert WorkflowRunStartedEvent is not None
    assert WorkflowRunCompletedEvent is not None
    assert WorkflowRunErrorEvent is not None
    assert WorkflowStepStartedEvent is not None
    assert WorkflowStepCompletedEvent is not None
    assert WorkflowStepErrorEvent is not None
    assert WorkflowStepSkippedEvent is not None
    assert WorkflowLoopIterationEvent is not None

  def test_import_context_types(self):
    from definable.agent.workflow import StepInput, StepOutput, StepStatus, WorkflowOutput

    assert StepInput is not None
    assert StepOutput is not None
    assert StepStatus is not None
    assert WorkflowOutput is not None
