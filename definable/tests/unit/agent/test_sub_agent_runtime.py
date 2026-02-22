"""Tests for sub-agent spawn runtime — the actual spawn_agent tool execution."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.events import (
  SubAgentCompletedEvent,
  SubAgentFailedEvent,
  SubAgentSpawnedEvent,
)
from definable.agent.pipeline.sub_agent import (
  SubAgentPolicy,
  _build_spawn_agent_function,
)
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.tool.decorator import tool


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _mock_response(content="done"):
  """Build a standard MagicMock model response."""
  response = MagicMock()
  response.content = content
  response.tool_calls = []
  response.tool_executions = []
  response.response_usage = Metrics()
  response.reasoning_content = None
  response.citations = None
  response.images = None
  response.videos = None
  response.audios = None
  return response


def _make_ctx(run_id="test"):
  """Build a minimal RunContext for _prepare_tools_for_run tests."""
  from definable.agent.run.base import RunContext

  return RunContext(run_id=run_id, session_id="s1", session_state={})


# ═══════════════════════════════════════════════════════════════════════
# Function injection
# ═══════════════════════════════════════════════════════════════════════


class TestSpawnAgentInjection:
  def test_spawn_agent_injected_when_sub_agents_true(self):
    """spawn_agent tool appears in prepared tools when sub_agents=True."""
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    tools = agent._prepare_tools_for_run(_make_ctx())
    assert "spawn_agent" in tools

  def test_spawn_agent_not_injected_when_sub_agents_none(self):
    """spawn_agent tool does NOT appear without sub_agents."""
    agent = Agent(model=MockModel(responses=["hi"]), config=NO_TRACE)
    tools = agent._prepare_tools_for_run(_make_ctx())
    assert "spawn_agent" not in tools

  def test_spawn_agent_injected_with_policy(self):
    """spawn_agent appears when sub_agents is a SubAgentPolicy."""
    policy = SubAgentPolicy(max_concurrent=2)
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=policy, config=NO_TRACE)
    tools = agent._prepare_tools_for_run(_make_ctx())
    assert "spawn_agent" in tools

  def test_spawn_agent_fresh_per_run(self):
    """Each call to _prepare_tools_for_run creates a fresh spawn_agent."""
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    t1 = agent._prepare_tools_for_run(_make_ctx("r1"))
    t2 = agent._prepare_tools_for_run(_make_ctx("r2"))
    assert t1["spawn_agent"] is not t2["spawn_agent"]


# ═══════════════════════════════════════════════════════════════════════
# Function object properties
# ═══════════════════════════════════════════════════════════════════════


class TestSpawnAgentFunctionObject:
  def test_function_name(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    assert fn.name == "spawn_agent"

  def test_function_description(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    assert "sub-agent" in fn.description.lower()

  def test_function_parameters_schema(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    params = fn.parameters
    assert params["type"] == "object"
    assert "goal" in params["properties"]
    assert "instructions" in params["properties"]
    assert "context" in params["properties"]
    assert "goal" in params["required"]

  def test_function_has_entrypoint(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    assert fn.entrypoint is not None
    assert asyncio.iscoroutinefunction(fn.entrypoint)

  def test_function_skip_entrypoint_processing(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    assert fn.skip_entrypoint_processing is True

  def test_function_has_tcb_list(self):
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(agent, SubAgentPolicy())
    assert hasattr(fn, "_tcb_list")
    assert isinstance(fn._tcb_list, list)
    assert len(fn._tcb_list) == 0


# ═══════════════════════════════════════════════════════════════════════
# Entrypoint execution — direct call
# ═══════════════════════════════════════════════════════════════════════


class TestSpawnAgentEntrypoint:
  @pytest.mark.asyncio
  async def test_spawn_returns_child_content(self):
    """spawn_agent entrypoint creates child and returns its content."""
    parent = Agent(model=MockModel(responses=["child answer"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    result = await fn.entrypoint(goal="What is 2+2?")
    assert result == "child answer"

  @pytest.mark.asyncio
  async def test_spawn_with_instructions(self):
    """spawn_agent passes instructions to child agent."""
    parent = Agent(model=MockModel(responses=["expert answer"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    result = await fn.entrypoint(goal="Analyze data", instructions="You are a data scientist.")
    assert result == "expert answer"

  @pytest.mark.asyncio
  async def test_spawn_with_context(self):
    """spawn_agent passes context to child agent."""
    parent = Agent(model=MockModel(responses=["contextual answer"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    result = await fn.entrypoint(goal="Summarize", context="Here is some background info.")
    assert result == "contextual answer"

  @pytest.mark.asyncio
  async def test_spawn_handles_failure_gracefully(self):
    """spawn_agent catches exceptions and returns error string."""

    def failing_side_effect(messages, tools, **kwargs):
      raise RuntimeError("Model exploded")

    parent = Agent(model=MockModel(side_effect=failing_side_effect), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    result = await fn.entrypoint(goal="This will fail")
    assert "Sub-agent failed" in result
    assert "Model exploded" in result

  @pytest.mark.asyncio
  async def test_spawn_returns_empty_on_none_content(self):
    """If child returns None content, spawn_agent returns empty string."""

    def none_content(messages, tools, **kwargs):
      r = _mock_response(content=None)
      return r

    parent = Agent(model=MockModel(side_effect=none_content), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    result = await fn.entrypoint(goal="Something")
    assert result == ""


# ═══════════════════════════════════════════════════════════════════════
# TCB tracking
# ═══════════════════════════════════════════════════════════════════════


class TestTCBTracking:
  @pytest.mark.asyncio
  async def test_tcb_created_on_spawn(self):
    parent = Agent(model=MockModel(responses=["done"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="test")
    assert len(fn._tcb_list) == 1

  @pytest.mark.asyncio
  async def test_tcb_completed_state(self):
    parent = Agent(model=MockModel(responses=["result"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="test")
    tcb = fn._tcb_list[0]
    assert tcb.state == "completed"
    assert tcb.result == "result"
    assert tcb.error is None

  @pytest.mark.asyncio
  async def test_tcb_failed_state(self):
    def failing(messages, tools, **kwargs):
      raise ValueError("bad input")

    parent = Agent(model=MockModel(side_effect=failing), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="test")
    tcb = fn._tcb_list[0]
    assert tcb.state == "failed"
    assert "bad input" in tcb.error
    assert tcb.result is None

  @pytest.mark.asyncio
  async def test_tcb_goal_recorded(self):
    parent = Agent(model=MockModel(responses=["ok"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="Research quantum computing")
    tcb = fn._tcb_list[0]
    assert tcb.goal == "Research quantum computing"

  @pytest.mark.asyncio
  async def test_tcb_has_unique_id(self):
    parent = Agent(model=MockModel(responses=["ok"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="task 1")
    await fn.entrypoint(goal="task 2")
    assert len(fn._tcb_list) == 2
    assert fn._tcb_list[0].id != fn._tcb_list[1].id

  @pytest.mark.asyncio
  async def test_tcb_run_output_stored_on_success(self):
    parent = Agent(model=MockModel(responses=["ok"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="test")
    tcb = fn._tcb_list[0]
    assert tcb.run_output is not None
    assert tcb.run_output.content == "ok"


# ═══════════════════════════════════════════════════════════════════════
# Event emission
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentEventEmission:
  @pytest.mark.asyncio
  async def test_spawned_event_emitted(self):
    events_received = []

    parent = Agent(model=MockModel(responses=["done"]), sub_agents=True, config=NO_TRACE)

    @parent.events.on(SubAgentSpawnedEvent)
    def on_spawn(event):
      events_received.append(event)

    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="Research X")
    assert len(events_received) == 1
    assert events_received[0].goal == "Research X"

  @pytest.mark.asyncio
  async def test_completed_event_emitted(self):
    events_received = []

    parent = Agent(model=MockModel(responses=["answer"]), sub_agents=True, config=NO_TRACE)

    @parent.events.on(SubAgentCompletedEvent)
    def on_complete(event):
      events_received.append(event)

    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="Do something")
    assert len(events_received) == 1
    assert events_received[0].result == "answer"

  @pytest.mark.asyncio
  async def test_failed_event_emitted(self):
    events_received = []

    def failing(messages, tools, **kwargs):
      raise RuntimeError("boom")

    parent = Agent(model=MockModel(side_effect=failing), sub_agents=True, config=NO_TRACE)

    @parent.events.on(SubAgentFailedEvent)
    def on_fail(event):
      events_received.append(event)

    fn = _build_spawn_agent_function(parent, SubAgentPolicy())
    await fn.entrypoint(goal="Will fail")
    assert len(events_received) == 1
    assert "boom" in events_received[0].error


# ═══════════════════════════════════════════════════════════════════════
# Concurrency — verified via max-inflight tracking (no async side_effect)
# ═══════════════════════════════════════════════════════════════════════


class TestConcurrency:
  @pytest.mark.asyncio
  async def test_semaphore_limits_concurrent(self):
    """Semaphore with max_concurrent=1 forces sequential execution."""
    max_inflight = 0
    current_inflight = 0
    lock = asyncio.Lock()

    original_arun = Agent.arun

    async def patched_arun(self, *args, **kwargs):
      nonlocal max_inflight, current_inflight
      async with lock:
        current_inflight += 1
        if current_inflight > max_inflight:
          max_inflight = current_inflight
      await asyncio.sleep(0.05)
      async with lock:
        current_inflight -= 1
      return await original_arun(self, *args, **kwargs)

    parent = Agent(model=MockModel(responses=["done"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy(max_concurrent=1))

    # Monkeypatch Agent.arun for inflight tracking
    Agent.arun = patched_arun
    try:
      await asyncio.gather(
        fn.entrypoint(goal="Task A"),
        fn.entrypoint(goal="Task B"),
      )
    finally:
      Agent.arun = original_arun

    # With max_concurrent=1, at most 1 should be in-flight at a time
    assert max_inflight == 1

  @pytest.mark.asyncio
  async def test_parallel_spawn_agents(self):
    """With sufficient concurrency, spawn_agents run in parallel."""
    original_arun = Agent.arun

    async def slow_arun(self, *args, **kwargs):
      await asyncio.sleep(0.1)
      return await original_arun(self, *args, **kwargs)

    parent = Agent(model=MockModel(responses=["done"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, SubAgentPolicy(max_concurrent=5))

    Agent.arun = slow_arun
    try:
      start = time.monotonic()
      await asyncio.gather(
        fn.entrypoint(goal="A"),
        fn.entrypoint(goal="B"),
        fn.entrypoint(goal="C"),
      )
      elapsed = time.monotonic() - start
    finally:
      Agent.arun = original_arun

    # 3 parallel tasks at 0.1s each: should take ~0.1s, not ~0.3s
    assert elapsed < 0.25


# ═══════════════════════════════════════════════════════════════════════
# Tool and knowledge inheritance
# ═══════════════════════════════════════════════════════════════════════


@tool
def sample_tool() -> str:
  """A sample tool for testing."""
  return "sample"


class TestInheritance:
  @pytest.mark.asyncio
  async def test_inherit_tools_true(self):
    """When inherit_tools=True, child gets parent's tools."""
    child_tools_seen = []

    def capture_tools(messages, tools, **kwargs):
      child_tools_seen.extend(tools or [])
      return _mock_response("done")

    parent = Agent(
      model=MockModel(side_effect=capture_tools),
      tools=[sample_tool],
      sub_agents=True,
      config=NO_TRACE,
    )
    fn = _build_spawn_agent_function(parent, SubAgentPolicy(inherit_tools=True))
    await fn.entrypoint(goal="Use tools")

    # Child should have the sample_tool (tools are passed as dicts to model)
    tool_names = [t.get("function", {}).get("name", t.get("name", "")) for t in child_tools_seen]
    assert "sample_tool" in tool_names

  @pytest.mark.asyncio
  async def test_inherit_tools_false(self):
    """When inherit_tools=False, child gets no tools."""
    child_tools_seen = []

    def capture_tools(messages, tools, **kwargs):
      child_tools_seen.extend(tools or [])
      return _mock_response("done")

    parent = Agent(
      model=MockModel(side_effect=capture_tools),
      tools=[sample_tool],
      sub_agents=True,
      config=NO_TRACE,
    )
    fn = _build_spawn_agent_function(parent, SubAgentPolicy(inherit_tools=False))
    await fn.entrypoint(goal="No tools")

    # Child should have no tools (empty list)
    assert len(child_tools_seen) == 0


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════


class TestCallbacks:
  @pytest.mark.asyncio
  async def test_on_spawn_callback(self):
    """on_spawn fires with TCB before the child runs."""
    spawn_states = []
    policy = SubAgentPolicy(on_spawn=lambda tcb: spawn_states.append(tcb.state))

    parent = Agent(model=MockModel(responses=["ok"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, policy)
    await fn.entrypoint(goal="test")

    # on_spawn fires before arun, so state should be "running" at callback time
    assert len(spawn_states) == 1
    assert spawn_states[0] == "running"

  @pytest.mark.asyncio
  async def test_on_complete_callback_success(self):
    completed_tcbs = []
    policy = SubAgentPolicy(on_complete=lambda tcb: completed_tcbs.append(tcb))

    parent = Agent(model=MockModel(responses=["result"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, policy)
    await fn.entrypoint(goal="test")

    assert len(completed_tcbs) == 1
    assert completed_tcbs[0].state == "completed"
    assert completed_tcbs[0].result == "result"

  @pytest.mark.asyncio
  async def test_on_complete_callback_on_failure(self):
    """on_complete fires even when sub-agent fails."""
    completed_tcbs = []
    policy = SubAgentPolicy(on_complete=lambda tcb: completed_tcbs.append(tcb))

    def failing(messages, tools, **kwargs):
      raise RuntimeError("fail")

    parent = Agent(model=MockModel(side_effect=failing), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, policy)
    await fn.entrypoint(goal="test")

    assert len(completed_tcbs) == 1
    assert completed_tcbs[0].state == "failed"

  @pytest.mark.asyncio
  async def test_async_callbacks(self):
    """Async callbacks are awaited."""
    spawned = []

    async def async_on_spawn(tcb):
      await asyncio.sleep(0.01)
      spawned.append(tcb)

    policy = SubAgentPolicy(on_spawn=async_on_spawn)
    parent = Agent(model=MockModel(responses=["ok"]), sub_agents=True, config=NO_TRACE)
    fn = _build_spawn_agent_function(parent, policy)
    await fn.entrypoint(goal="async test")

    assert len(spawned) == 1


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: full agent.arun with spawn_agent
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEnd:
  @pytest.mark.asyncio
  async def test_agent_arun_with_spawn(self):
    """Full agent.arun where the model calls spawn_agent."""
    call_count = 0

    def parent_side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      response = _mock_response()

      if call_count == 1:
        # First call: model wants to spawn a sub-agent
        response.content = ""
        response.tool_calls = [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "spawn_agent",
              "arguments": '{"goal": "What is 2+2?"}',
            },
          }
        ]
      else:
        # Subsequent calls: model sees the spawn result and gives final answer
        response.content = "The answer is 4"
        response.tool_calls = []
      return response

    parent = Agent(
      model=MockModel(side_effect=parent_side_effect),
      sub_agents=True,
      config=NO_TRACE,
    )
    result = await parent.arun("Calculate 2+2 using a sub-agent")
    assert result.content == "The answer is 4"
