"""Tests for sub-agent spawning infrastructure — events, types, TCB tracking."""

from definable.agent.events import (
  SubAgentCompletedEvent,
  SubAgentFailedEvent,
  SubAgentKilledEvent,
  SubAgentSpawnedEvent,
)
from definable.agent.pipeline.sub_agent import SubAgentConfig, SubAgentPolicy, ThreadControlBlock
from definable.run.agent import RUN_EVENT_TYPE_REGISTRY, RunEvent


# ═══════════════════════════════════════════════════════════════════════
# SubAgentConfig
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentConfig:
  def test_minimal(self):
    cfg = SubAgentConfig(instructions="Research topic X")
    assert cfg.instructions == "Research topic X"
    assert cfg.tools is None
    assert cfg.model is None
    assert cfg.max_tool_rounds == 15
    assert cfg.knowledge is None
    assert cfg.context is None

  def test_full(self):
    cfg = SubAgentConfig(
      instructions="Analyze data",
      model="openai/gpt-4o-mini",
      max_tool_rounds=10,
      context="Parent context here",
    )
    assert cfg.model == "openai/gpt-4o-mini"
    assert cfg.max_tool_rounds == 10
    assert cfg.context == "Parent context here"


# ═══════════════════════════════════════════════════════════════════════
# ThreadControlBlock
# ═══════════════════════════════════════════════════════════════════════


class TestThreadControlBlock:
  def test_default_state(self):
    tcb = ThreadControlBlock(id="t1", goal="Analyze X")
    assert tcb.state == "running"
    assert tcb.result is None
    assert tcb.error is None
    assert tcb.metrics is None

  def test_start_time_auto(self):
    tcb = ThreadControlBlock(id="t1", goal="test")
    assert tcb.start_time > 0

  def test_completed_state(self):
    tcb = ThreadControlBlock(id="t1", goal="test")
    tcb.state = "completed"
    tcb.result = "Found 5 results"
    assert tcb.state == "completed"
    assert tcb.result == "Found 5 results"

  def test_failed_state(self):
    tcb = ThreadControlBlock(id="t1", goal="test")
    tcb.state = "failed"
    tcb.error = "API timeout"
    assert tcb.state == "failed"
    assert tcb.error == "API timeout"


# ═══════════════════════════════════════════════════════════════════════
# SubAgentPolicy
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentPolicy:
  def test_defaults(self):
    p = SubAgentPolicy()
    assert p.max_concurrent == 5
    assert p.max_tool_rounds == 15
    assert p.inherit_tools is True
    assert p.inherit_knowledge is False
    assert p.allowed_models is None
    assert p.on_spawn is None
    assert p.on_complete is None

  def test_restricted_models(self):
    p = SubAgentPolicy(allowed_models=["openai/gpt-4o-mini", "openai/gpt-4o"])
    assert p.allowed_models is not None
    assert len(p.allowed_models) == 2

  def test_callbacks(self):
    def on_spawn(tcb):
      return None

    def on_complete(tcb):
      return None

    p = SubAgentPolicy(on_spawn=on_spawn, on_complete=on_complete)
    assert p.on_spawn is on_spawn
    assert p.on_complete is on_complete


# ═══════════════════════════════════════════════════════════════════════
# Sub-agent events
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentEvents:
  def test_spawned_event(self):
    e = SubAgentSpawnedEvent(
      run_id="run1",
      thread_id="t1",
      goal="Research X",
      instructions="You are a researcher",
    )
    assert e.event == RunEvent.sub_agent_spawned.value
    assert e.thread_id == "t1"
    assert e.goal == "Research X"

  def test_completed_event(self):
    e = SubAgentCompletedEvent(
      run_id="run1",
      thread_id="t1",
      result="Found 5 papers",
    )
    assert e.event == RunEvent.sub_agent_completed.value
    assert e.result == "Found 5 papers"

  def test_failed_event(self):
    e = SubAgentFailedEvent(
      run_id="run1",
      thread_id="t1",
      error="API rate limited",
    )
    assert e.event == RunEvent.sub_agent_failed.value
    assert e.error == "API rate limited"

  def test_killed_event(self):
    e = SubAgentKilledEvent(
      run_id="run1",
      thread_id="t1",
      reason="Timeout exceeded",
    )
    assert e.event == RunEvent.sub_agent_killed.value
    assert e.reason == "Timeout exceeded"


# ═══════════════════════════════════════════════════════════════════════
# Event registry
# ═══════════════════════════════════════════════════════════════════════


class TestEventRegistry:
  def test_spawned_in_registry(self):
    assert RunEvent.sub_agent_spawned.value in RUN_EVENT_TYPE_REGISTRY

  def test_completed_in_registry(self):
    assert RunEvent.sub_agent_completed.value in RUN_EVENT_TYPE_REGISTRY

  def test_failed_in_registry(self):
    assert RunEvent.sub_agent_failed.value in RUN_EVENT_TYPE_REGISTRY

  def test_killed_in_registry(self):
    assert RunEvent.sub_agent_killed.value in RUN_EVENT_TYPE_REGISTRY

  def test_registry_maps_to_correct_types(self):
    assert RUN_EVENT_TYPE_REGISTRY[RunEvent.sub_agent_spawned.value] is SubAgentSpawnedEvent
    assert RUN_EVENT_TYPE_REGISTRY[RunEvent.sub_agent_completed.value] is SubAgentCompletedEvent
    assert RUN_EVENT_TYPE_REGISTRY[RunEvent.sub_agent_failed.value] is SubAgentFailedEvent
    assert RUN_EVENT_TYPE_REGISTRY[RunEvent.sub_agent_killed.value] is SubAgentKilledEvent


# ═══════════════════════════════════════════════════════════════════════
# RunEvent enum values
# ═══════════════════════════════════════════════════════════════════════


class TestRunEventEnum:
  def test_sub_agent_spawned_value(self):
    assert RunEvent.sub_agent_spawned.value == "SubAgentSpawned"

  def test_sub_agent_completed_value(self):
    assert RunEvent.sub_agent_completed.value == "SubAgentCompleted"

  def test_sub_agent_failed_value(self):
    assert RunEvent.sub_agent_failed.value == "SubAgentFailed"

  def test_sub_agent_killed_value(self):
    assert RunEvent.sub_agent_killed.value == "SubAgentKilled"


# ═══════════════════════════════════════════════════════════════════════
# Agent integration
# ═══════════════════════════════════════════════════════════════════════


class TestAgentSubAgentIntegration:
  def test_agent_accepts_sub_agents_true(self):
    from definable.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=True)  # type: ignore[arg-type]
    assert agent._sub_agent_policy is not None
    assert agent._sub_agent_policy.max_concurrent == 5

  def test_agent_accepts_sub_agents_policy(self):
    from definable.agent import Agent
    from definable.agent.testing import MockModel

    policy = SubAgentPolicy(max_concurrent=2, inherit_knowledge=True)
    agent = Agent(model=MockModel(responses=["hi"]), sub_agents=policy)  # type: ignore[arg-type]
    assert agent._sub_agent_policy is not None
    assert agent._sub_agent_policy.max_concurrent == 2
    assert agent._sub_agent_policy.inherit_knowledge is True

  def test_agent_sub_agents_none_by_default(self):
    from definable.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(model=MockModel(responses=["hi"]))  # type: ignore[arg-type]
    assert agent._sub_agent_policy is None
