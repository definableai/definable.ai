"""Tests for default pipeline phase implementations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.pipeline.phases import (
  ComposePhase,
  GuardInputPhase,
  GuardOutputPhase,
  InvokeLoopPhase,
  PreparePhase,
  RecallPhase,
  StorePhase,
  ThinkPhase,
)
from definable.agent.pipeline.state import LoopState, LoopStatus
from definable.agent.run.base import RunContext


def _make_state(**overrides) -> LoopState:
  defaults = {
    "run_id": "test-run",
    "session_id": "test-session",
    "context": RunContext(run_id="test-run", session_id="test-session"),
  }
  defaults.update(overrides)
  return LoopState(**defaults)  # type: ignore[arg-type]


def _mock_agent(**attrs):
  """Create a mock Agent with reasonable defaults."""
  agent = MagicMock()
  agent.guardrails = None
  agent.memory = None
  agent._knowledge = None
  agent._thinking = None
  agent._ensure_toolkits_initialized = AsyncMock()
  agent._prepare_tools_for_run = MagicMock(return_value={})
  agent._should_store_memory = MagicMock(return_value=False)
  agent._memory_store = MagicMock(return_value=[])
  agent._drain_memory_tasks = AsyncMock()
  agent._run_pre_execution_pipeline = AsyncMock(return_value=([], [], None, None, None, None, None))
  agent._build_invoke_messages = MagicMock(return_value=[])
  agent._thinking_should_run = AsyncMock(return_value=False)
  agent._execute_thinking = AsyncMock(return_value=(None, None, None))
  for k, v in attrs.items():
    setattr(agent, k, v)
  return agent


# ═══════════════════════════════════════════════════════════════════════
# Phase name checks
# ═══════════════════════════════════════════════════════════════════════


class TestPhaseNames:
  def test_prepare_name(self):
    assert PreparePhase(_mock_agent()).name == "prepare"

  def test_recall_name(self):
    assert RecallPhase(_mock_agent()).name == "recall"

  def test_think_name(self):
    assert ThinkPhase(_mock_agent()).name == "think"

  def test_guard_input_name(self):
    assert GuardInputPhase(_mock_agent()).name == "guard_input"

  def test_guard_output_name(self):
    assert GuardOutputPhase(_mock_agent()).name == "guard_output"

  def test_compose_name(self):
    assert ComposePhase(_mock_agent()).name == "compose"

  def test_invoke_loop_name(self):
    assert InvokeLoopPhase(_mock_agent()).name == "invoke_loop"

  def test_store_name(self):
    assert StorePhase(_mock_agent()).name == "store"


# ═══════════════════════════════════════════════════════════════════════
# PreparePhase
# ═══════════════════════════════════════════════════════════════════════


class TestPreparePhase:
  @pytest.mark.asyncio
  async def test_calls_prepare_tools(self):
    agent = _mock_agent()
    agent._prepare_tools_for_run = MagicMock(return_value={"search": MagicMock()})
    phase = PreparePhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    agent._prepare_tools_for_run.assert_called_once()
    assert "search" in state.tools


# ═══════════════════════════════════════════════════════════════════════
# RecallPhase
# ═══════════════════════════════════════════════════════════════════════


class TestRecallPhase:
  @pytest.mark.asyncio
  async def test_yields_state(self):
    agent = _mock_agent()
    phase = RecallPhase(agent)
    state = _make_state()
    results = []
    async for s, e in phase.execute(state):
      results.append((s, e))
    assert len(results) >= 1
    assert results[-1][0] is state


# ═══════════════════════════════════════════════════════════════════════
# ThinkPhase
# ═══════════════════════════════════════════════════════════════════════


class TestThinkPhase:
  def test_should_run_false_without_thinking(self):
    agent = _mock_agent(_thinking=None)
    phase = ThinkPhase(agent)
    state = _make_state()
    assert phase.should_run(state) is False

  @pytest.mark.asyncio
  async def test_skips_when_trigger_says_no(self):
    agent = _mock_agent(_thinking=MagicMock())
    agent._thinking_should_run = AsyncMock(return_value=False)
    phase = ThinkPhase(agent)
    state = _make_state()
    results = []
    async for s, e in phase.execute(state):
      results.append((s, e))
    assert len(results) == 1
    assert results[0][1] is None
    # Thinking output not populated
    assert state.thinking_output is None


# ═══════════════════════════════════════════════════════════════════════
# GuardInputPhase / GuardOutputPhase
# ═══════════════════════════════════════════════════════════════════════


class TestGuardPhases:
  def test_guard_input_should_run_false_no_guardrails(self):
    agent = _mock_agent(guardrails=None)
    phase = GuardInputPhase(agent)
    state = _make_state()
    assert phase.should_run(state) is False

  def test_guard_output_should_run_false_no_guardrails(self):
    agent = _mock_agent(guardrails=None)
    phase = GuardOutputPhase(agent)
    state = _make_state()
    assert phase.should_run(state) is False

  @pytest.mark.asyncio
  async def test_guard_input_blocks(self):
    """GuardInputPhase sets blocked status when guardrail blocks."""
    block_result = MagicMock()
    block_result.content = "Input blocked by guardrail"

    guardrails_mock = MagicMock()
    guardrails_mock.input = [MagicMock()]

    agent = _mock_agent(guardrails=guardrails_mock)
    agent._run_input_guardrails = AsyncMock(return_value=block_result)

    phase = GuardInputPhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    assert state.status == LoopStatus.blocked
    assert state.content == "Input blocked by guardrail"

  @pytest.mark.asyncio
  async def test_guard_input_passes(self):
    """GuardInputPhase allows when guardrails pass."""
    guardrails_mock = MagicMock()
    guardrails_mock.input = [MagicMock()]

    agent = _mock_agent(guardrails=guardrails_mock)
    agent._run_input_guardrails = AsyncMock(return_value=None)

    phase = GuardInputPhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    assert state.status != LoopStatus.blocked


# ═══════════════════════════════════════════════════════════════════════
# ComposePhase
# ═══════════════════════════════════════════════════════════════════════


class TestComposePhase:
  @pytest.mark.asyncio
  async def test_calls_build_invoke_messages(self):
    from definable.model.message import Message

    agent = _mock_agent()
    agent._build_invoke_messages = AsyncMock(return_value=([Message(role="system", content="You are helpful")], None, None))

    phase = ComposePhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    agent._build_invoke_messages.assert_called_once()
    assert len(state.invoke_messages) == 1


# ═══════════════════════════════════════════════════════════════════════
# InvokeLoopPhase
# ═══════════════════════════════════════════════════════════════════════


class TestInvokeLoopPhase:
  @pytest.mark.asyncio
  async def test_name(self):
    agent = _mock_agent()
    phase = InvokeLoopPhase(agent)
    assert phase.name == "invoke_loop"


# ═══════════════════════════════════════════════════════════════════════
# StorePhase
# ═══════════════════════════════════════════════════════════════════════


class TestStorePhase:
  def test_should_run_false_when_no_memory(self):
    agent = _mock_agent()
    agent._should_store_memory = MagicMock(return_value=False)
    phase = StorePhase(agent)
    state = _make_state()
    assert phase.should_run(state) is False

  @pytest.mark.asyncio
  async def test_stores_when_memory_enabled(self):
    agent = _mock_agent()
    agent._should_store_memory = MagicMock(return_value=True)
    agent._memory_store = MagicMock(return_value=[])
    agent._drain_memory_tasks = AsyncMock()

    phase = StorePhase(agent)
    state = _make_state(content="Hello response")
    state.new_messages = []
    async for s, e in phase.execute(state):
      pass
    agent._memory_store.assert_called_once()
    agent._drain_memory_tasks.assert_awaited_once()
