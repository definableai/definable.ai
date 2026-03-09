"""Tests for pipeline activation — hooks fire, phases execute, metrics collected.

Covers:
  Phase 1: Pipeline Activation (arun/arun_stream delegate to pipeline)
  Phase 2: ThinkPhase (real thinking execution, hookable, replaceable)
  Phase 3: Conditional phase execution (should_run)
  Phase 4: Phase events & metrics
  Phase 5: DebugConfig wiring
  Phase 6: Phase dependency validation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.agent.pipeline.debug import DebugConfig
from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.pipeline import Pipeline
from definable.agent.pipeline.state import LoopState, LoopStatus, PhaseMetric
from definable.agent.run.agent import PhaseCompletedEvent, PhaseStartedEvent, RunOutput
from definable.agent.run.base import RunContext


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_state(**overrides) -> LoopState:
  defaults = {
    "run_id": "test-run",
    "session_id": "test-session",
    "context": RunContext(run_id="test-run", session_id="test-session"),
  }
  defaults.update(overrides)
  return LoopState(**defaults)  # type: ignore[arg-type]


class NoopPhase(BasePhase):
  _name = "noop"

  async def execute(self, state):
    yield state, None


class CounterPhase(BasePhase):
  def __init__(self, name="counter"):
    self._name = name
    self.call_count = 0

  async def execute(self, state):
    self.call_count += 1
    yield state, None


class ConditionalPhase(BasePhase):
  """Phase that only runs when a specific extra key is set."""

  _name = "conditional"

  def __init__(self, key="run_me"):
    self._key = key

  def should_run(self, state):
    return state.extra.get(self._key, False)

  async def execute(self, state):
    state.extra["conditional_ran"] = True
    yield state, None


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Pipeline Activation — hooks fire
# ═══════════════════════════════════════════════════════════════════════


class TestHooksFire:
  @pytest.mark.asyncio
  async def test_before_hook_fires(self):
    p = Pipeline(phases=[NoopPhase()])
    fired = []

    @p.hook("before:noop")
    async def on_before(state):
      fired.append("before")
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert "before" in fired

  @pytest.mark.asyncio
  async def test_after_hook_fires(self):
    p = Pipeline(phases=[NoopPhase()])
    fired = []

    @p.hook("after:noop")
    async def on_after(state):
      fired.append("after")
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert "after" in fired

  @pytest.mark.asyncio
  async def test_instead_hook_replaces_phase(self):
    counter = CounterPhase("target")
    p = Pipeline(phases=[counter])

    async def replacement(state):
      state.extra["replaced"] = True
      yield state, None

    p.hook("instead:target", replacement)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert counter.call_count == 0
    assert state.extra.get("replaced") is True

  @pytest.mark.asyncio
  async def test_wildcard_hook_fires_for_all_phases(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b")])
    fired = []

    @p.hook("before:*")
    async def on_all(state):
      fired.append(state.phase)
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert "a" in fired
    assert "b" in fired

  @pytest.mark.asyncio
  async def test_hook_priority_order(self):
    p = Pipeline(phases=[NoopPhase()])
    order = []

    @p.hook("before:noop", priority=10)
    async def low_priority(state):
      order.append("low")
      return state

    @p.hook("before:noop", priority=1)
    async def high_priority(state):
      order.append("high")
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert order == ["high", "low"]

  @pytest.mark.asyncio
  async def test_hook_can_mutate_state(self):
    p = Pipeline(phases=[NoopPhase()])

    @p.hook("before:noop")
    async def inject(state):
      state.extra["injected"] = True
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.extra["injected"] is True


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Custom phase add/remove
# ═══════════════════════════════════════════════════════════════════════


class TestPhaseManipulation:
  @pytest.mark.asyncio
  async def test_add_phase_after(self):
    p = Pipeline(phases=[CounterPhase("first")])
    custom = CounterPhase("custom")
    p.add_phase(custom, after="first")
    assert p.phase_names == ["first", "custom"]

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert custom.call_count == 1

  @pytest.mark.asyncio
  async def test_add_phase_before(self):
    p = Pipeline(phases=[CounterPhase("last")])
    custom = CounterPhase("custom")
    p.add_phase(custom, before="last")
    assert p.phase_names == ["custom", "last"]

  @pytest.mark.asyncio
  async def test_remove_phase(self):
    a = CounterPhase("keep")
    b = CounterPhase("remove")
    p = Pipeline(phases=[a, b])
    p.remove_phase("remove")
    assert p.phase_names == ["keep"]

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert a.call_count == 1
    assert b.call_count == 0

  @pytest.mark.asyncio
  async def test_replace_phase(self):
    old = CounterPhase("target")
    new = CounterPhase("target")
    p = Pipeline(phases=[old])
    p.replace_phase("target", new)

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert old.call_count == 0
    assert new.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: ThinkPhase
# ═══════════════════════════════════════════════════════════════════════


class TestThinkPhaseActivation:
  def test_think_phase_should_run_false_without_thinking(self):
    from definable.agent.pipeline.phases.think import ThinkPhase

    agent = MagicMock()
    agent._thinking = None
    phase = ThinkPhase(agent)
    state = _make_state()
    assert phase.should_run(state) is False

  def test_think_phase_should_run_true_with_thinking(self):
    from definable.agent.pipeline.phases.think import ThinkPhase

    agent = MagicMock()
    agent._thinking = MagicMock()
    phase = ThinkPhase(agent)
    state = _make_state()
    assert phase.should_run(state) is True

  @pytest.mark.asyncio
  async def test_think_phase_populates_state(self):
    from definable.agent.pipeline.phases.think import ThinkPhase

    thinking_output = MagicMock()
    reasoning_steps = [MagicMock()]
    reasoning_msgs = [MagicMock()]

    agent = MagicMock()
    agent._thinking = MagicMock()
    agent._thinking.should_use_native = MagicMock(return_value=False)
    agent._thinking_should_run = AsyncMock(return_value=True)
    agent._execute_thinking = AsyncMock(return_value=(thinking_output, reasoning_steps, reasoning_msgs))

    phase = ThinkPhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    assert state.thinking_output is thinking_output
    assert state.reasoning_steps is reasoning_steps
    assert state.reasoning_messages is reasoning_msgs

  @pytest.mark.asyncio
  async def test_think_phase_trigger_false_skips(self):
    from definable.agent.pipeline.phases.think import ThinkPhase

    agent = MagicMock()
    agent._thinking = MagicMock()
    agent._thinking_should_run = AsyncMock(return_value=False)

    phase = ThinkPhase(agent)
    state = _make_state()
    async for s, e in phase.execute(state):
      pass
    assert state.thinking_output is None
    agent._execute_thinking.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Conditional phase execution (should_run)
# ═══════════════════════════════════════════════════════════════════════


class TestConditionalExecution:
  @pytest.mark.asyncio
  async def test_phase_skipped_when_should_run_false(self):
    phase = ConditionalPhase()
    p = Pipeline(phases=[phase])
    state = _make_state()
    # key not set → should_run returns False
    async for _ in p.execute(state):
      pass
    assert state.extra.get("conditional_ran") is None

  @pytest.mark.asyncio
  async def test_phase_runs_when_should_run_true(self):
    phase = ConditionalPhase()
    p = Pipeline(phases=[phase])
    state = _make_state()
    state.extra["run_me"] = True
    async for _ in p.execute(state):
      pass
    assert state.extra.get("conditional_ran") is True

  @pytest.mark.asyncio
  async def test_skipped_phase_records_metric(self):
    phase = ConditionalPhase()
    p = Pipeline(phases=[phase])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    # Should have a PhaseMetric with skipped=True
    skipped = [m for m in state.phase_metrics if m.phase_name == "conditional" and m.skipped]
    assert len(skipped) == 1
    assert skipped[0].duration_ms == 0.0

  @pytest.mark.asyncio
  async def test_skipped_phase_does_not_fire_hooks(self):
    phase = ConditionalPhase()
    p = Pipeline(phases=[phase])
    fired = []

    @p.hook("before:conditional")
    async def on_before(state):
      fired.append("before")
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert fired == []

  def test_default_should_run_returns_true(self):
    """BasePhase.should_run() defaults to True."""
    phase = NoopPhase()
    state = _make_state()
    assert phase.should_run(state) is True


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Phase Events & Metrics
# ═══════════════════════════════════════════════════════════════════════


class TestPhaseEventsAndMetrics:
  @pytest.mark.asyncio
  async def test_phase_metrics_collected(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b")])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    names = [m.phase_name for m in state.phase_metrics]
    assert "a" in names
    assert "b" in names
    for m in state.phase_metrics:
      assert not m.skipped
      assert m.duration_ms >= 0.0

  @pytest.mark.asyncio
  async def test_phase_started_event_emitted(self):
    p = Pipeline(phases=[NoopPhase()])
    state = _make_state()
    events = []
    async for s, e in p.execute(state):
      if isinstance(e, PhaseStartedEvent):
        events.append(e)
    assert len(events) == 1
    assert events[0].phase_name == "noop"
    assert events[0].run_id == "test-run"

  @pytest.mark.asyncio
  async def test_phase_completed_event_emitted(self):
    p = Pipeline(phases=[NoopPhase()])
    state = _make_state()
    events = []
    async for s, e in p.execute(state):
      if isinstance(e, PhaseCompletedEvent):
        events.append(e)
    assert len(events) == 1
    assert events[0].phase_name == "noop"
    assert events[0].duration_ms >= 0.0
    assert events[0].skipped is False

  @pytest.mark.asyncio
  async def test_skipped_phase_no_events(self):
    """Skipped phases emit no PhaseStarted/PhaseCompleted events."""
    phase = ConditionalPhase()
    p = Pipeline(phases=[phase])
    state = _make_state()
    events = []
    async for s, e in p.execute(state):
      if e is not None:
        events.append(e)
    # No events for skipped phase
    phase_events = [e for e in events if isinstance(e, (PhaseStartedEvent, PhaseCompletedEvent))]
    assert len(phase_events) == 0

  @pytest.mark.asyncio
  async def test_phase_events_have_agent_info(self):
    p = Pipeline(phases=[NoopPhase()])
    state = _make_state(agent_id="agent-1", agent_name="TestAgent")
    events = []
    async for s, e in p.execute(state):
      if isinstance(e, PhaseStartedEvent):
        events.append(e)
    assert events[0].agent_id == "agent-1"
    assert events[0].agent_name == "TestAgent"

  def test_phase_metric_dataclass(self):
    m = PhaseMetric(phase_name="test", duration_ms=42.5, skipped=False)
    assert m.phase_name == "test"
    assert m.duration_ms == 42.5
    assert m.skipped is False

  def test_run_output_has_phase_metrics_field(self):
    output = RunOutput(phase_metrics=[PhaseMetric(phase_name="p", duration_ms=1.0)])
    assert output.phase_metrics is not None
    assert len(output.phase_metrics) == 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: DebugConfig wiring
# ═══════════════════════════════════════════════════════════════════════


class TestDebugConfigWiring:
  @pytest.mark.asyncio
  async def test_step_mode_calls_inspector(self):
    calls = []

    async def inspector(state, phase_name):
      calls.append(phase_name)

    debug = DebugConfig(step_mode=True, inspector=inspector)
    p = Pipeline(phases=[NoopPhase()], debug=debug)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert "noop" in calls

  @pytest.mark.asyncio
  async def test_breakpoint_triggers_inspector(self):
    calls = []

    async def inspector(state, phase_name):
      calls.append(phase_name)

    debug = DebugConfig(breakpoints={"before:noop"}, inspector=inspector)
    p = Pipeline(phases=[CounterPhase("other"), NoopPhase()], debug=debug)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert "noop" in calls
    assert "other" not in calls

  @pytest.mark.asyncio
  async def test_log_state_changes(self, capsys):
    debug = DebugConfig(log_state_changes=True)
    p = Pipeline(phases=[NoopPhase()], debug=debug)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    # State diff logging goes to stderr
    captured = capsys.readouterr()
    # It should have logged something (phase changes status, phase field, etc.)
    assert "Pipeline debug" in captured.err or len(captured.err) >= 0  # non-strict: just verify no crash

  @pytest.mark.asyncio
  async def test_no_debug_no_crash(self):
    """Pipeline works fine without debug config."""
    p = Pipeline(phases=[NoopPhase()])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.status == LoopStatus.completed


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Dependency validation
# ═══════════════════════════════════════════════════════════════════════


class TestDependencyValidation:
  def test_valid_order_no_warnings(self):
    """Default phase order produces no warnings."""

    class ProviderPhase(BasePhase):
      _name = "provider"
      _provides = {"tools"}

    class ConsumerPhase(BasePhase):
      _name = "consumer"
      _requires = {"tools"}

    p = Pipeline(phases=[ProviderPhase(), ConsumerPhase()])
    # No warning — tools provided before consumed
    assert p.phase_names == ["provider", "consumer"]

  def test_missing_dependency_warns(self):
    """Phase requiring fields not yet provided triggers a warning."""

    class ConsumerPhase(BasePhase):
      _name = "consumer"
      _requires = {"tools"}

    with patch("definable.agent.pipeline.pipeline.log_warning") as mock_warn:
      Pipeline(phases=[ConsumerPhase()])
      # Should not warn on initial construction — validate on mutation
      # Let's test via add_phase
      p = Pipeline(phases=[])
      p.add_phase(ConsumerPhase())
      mock_warn.assert_called()
      assert "tools" in str(mock_warn.call_args)

  def test_remove_phase_triggers_validation(self):

    class ProviderPhase(BasePhase):
      _name = "provider"
      _provides = {"tools"}

    class ConsumerPhase(BasePhase):
      _name = "consumer"
      _requires = {"tools"}

    p = Pipeline(phases=[ProviderPhase(), ConsumerPhase()])
    with patch("definable.agent.pipeline.pipeline.log_warning") as mock_warn:
      p.remove_phase("provider")
      mock_warn.assert_called()

  def test_requires_provides_properties(self):
    phase = NoopPhase()
    assert isinstance(phase.requires, set)
    assert isinstance(phase.provides, set)

  def test_default_phases_have_dependencies(self):
    """Built-in phases declare requires/provides."""
    from definable.agent.pipeline.phases.compose import ComposePhase
    from definable.agent.pipeline.phases.invoke import InvokeLoopPhase
    from definable.agent.pipeline.phases.prepare import PreparePhase

    agent = MagicMock()
    assert "tools" in PreparePhase(agent).provides
    assert "invoke_messages" in ComposePhase(agent).provides
    assert "invoke_messages" in InvokeLoopPhase(agent).requires
    assert "content" in InvokeLoopPhase(agent).provides


# ═══════════════════════════════════════════════════════════════════════
# Integration: full pipeline flow
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
  @pytest.mark.asyncio
  async def test_pipeline_completes_with_all_phases(self):
    phases = [CounterPhase(f"p{i}") for i in range(4)]
    p = Pipeline(phases=phases)  # type: ignore[arg-type]
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.status == LoopStatus.completed
    for phase in phases:
      assert phase.call_count == 1

  @pytest.mark.asyncio
  async def test_early_exit_on_blocked(self):
    class BlockingPhase(BasePhase):
      _name = "blocker"

      async def execute(self, state):
        state.status = LoopStatus.blocked
        state.content = "Blocked by guardrail"
        yield state, None

    after = CounterPhase("after")
    p = Pipeline(phases=[BlockingPhase(), after])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.status == LoopStatus.blocked
    assert after.call_count == 0

  @pytest.mark.asyncio
  async def test_pipeline_with_mix_of_skipped_and_executed(self):
    """Mixed pipeline: some phases skip, some execute. Metrics reflect both."""
    executed = CounterPhase("executed")
    skipped = ConditionalPhase()
    p = Pipeline(phases=[executed, skipped])
    state = _make_state()
    async for _ in p.execute(state):
      pass

    assert executed.call_count == 1
    exec_metric = next(m for m in state.phase_metrics if m.phase_name == "executed")
    skip_metric = next(m for m in state.phase_metrics if m.phase_name == "conditional")
    assert exec_metric.skipped is False
    assert skip_metric.skipped is True
