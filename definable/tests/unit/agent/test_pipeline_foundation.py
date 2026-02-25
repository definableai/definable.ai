"""Tests for pipeline foundation — LoopState, Pipeline, Phase, EventStream, ToolRetry, DebugConfig."""

from dataclasses import dataclass

import pytest

from definable.agent.pipeline.debug import DebugConfig
from definable.agent.pipeline.event_stream import EventStream
from definable.agent.pipeline.phase import BasePhase, Phase
from definable.agent.pipeline.pipeline import Pipeline
from definable.agent.pipeline.state import LoopState, LoopStatus
from definable.agent.pipeline.sub_agent import SubAgentConfig, SubAgentPolicy, ThreadControlBlock
from definable.agent.pipeline.tool_retry import ToolRetry
from definable.agent.run.base import BaseRunOutputEvent


# ── Helpers ────────────────────────────────────────────────────────────


class NoOpPhase(BasePhase):
  _name = "noop"

  async def execute(self, state):
    yield state, None


class CounterPhase(BasePhase):
  """Phase that increments a counter in state.extra."""

  def __init__(self, name: str = "counter"):
    self._name = name
    self.call_count = 0

  async def execute(self, state):
    self.call_count += 1
    state.extra[f"{self._name}_count"] = self.call_count
    yield state, None


@dataclass
class FakeEvent(BaseRunOutputEvent):
  event: str = "fake"
  run_id: str = ""
  data: str = ""


class EventEmittingPhase(BasePhase):
  _name = "emitter"

  async def execute(self, state):
    yield state, FakeEvent(run_id=state.run_id, data="from_emitter")


def _make_state(**overrides) -> LoopState:
  defaults = {"run_id": "test-run", "session_id": "test-session"}
  defaults.update(overrides)
  return LoopState(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# LoopState
# ═══════════════════════════════════════════════════════════════════════


class TestLoopState:
  def test_create_minimal(self):
    state = LoopState(run_id="r1", session_id="s1")
    assert state.run_id == "r1"
    assert state.session_id == "s1"
    assert state.status == LoopStatus.pending
    assert state.content is None
    assert state.tools == {}
    assert state.new_messages == []

  def test_default_collections_are_independent(self):
    s1 = LoopState(run_id="r1", session_id="s1")
    s2 = LoopState(run_id="r2", session_id="s2")
    s1.new_messages.append("msg")  # type: ignore[arg-type]
    assert len(s2.new_messages) == 0

  def test_mutable_fields(self):
    state = _make_state()
    state.content = "hello"
    state.status = LoopStatus.completed
    state.extra["key"] = "val"
    assert state.content == "hello"
    assert state.status == LoopStatus.completed
    assert state.extra["key"] == "val"

  def test_extra_dict(self):
    state = _make_state()
    state.extra["custom"] = 42
    assert state.extra["custom"] == 42

  def test_thread_control_blocks_default(self):
    state = _make_state()
    assert state.thread_control_blocks == []

  def test_streaming_and_cancellation(self):
    state = _make_state(streaming=True, cancellation_token="tok")
    assert state.streaming is True
    assert state.cancellation_token == "tok"  # type: ignore[comparison-overlap]


# ═══════════════════════════════════════════════════════════════════════
# LoopStatus
# ═══════════════════════════════════════════════════════════════════════


class TestLoopStatus:
  def test_values(self):
    assert LoopStatus.pending == "pending"
    assert LoopStatus.running == "running"  # type: ignore[unreachable]
    assert LoopStatus.completed == "completed"
    assert LoopStatus.paused == "paused"
    assert LoopStatus.cancelled == "cancelled"
    assert LoopStatus.blocked == "blocked"
    assert LoopStatus.error == "error"

  def test_is_string(self):
    assert isinstance(LoopStatus.completed, str)


# ═══════════════════════════════════════════════════════════════════════
# Phase protocol
# ═══════════════════════════════════════════════════════════════════════


class TestPhase:
  def test_base_phase_conforms_to_protocol(self):
    assert isinstance(NoOpPhase(), Phase)

  def test_custom_phase_conforms(self):
    class MyPhase:
      @property
      def name(self):
        return "my"

      async def execute(self, state):
        yield state, None

    assert isinstance(MyPhase(), Phase)

  def test_base_phase_name(self):
    p = NoOpPhase()
    assert p.name == "noop"

  @pytest.mark.asyncio
  async def test_base_phase_yields_state(self):
    p = NoOpPhase()
    state = _make_state()
    results = []
    async for s, e in p.execute(state):
      results.append((s, e))
    assert len(results) == 1
    assert results[0][0] is state
    assert results[0][1] is None


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — phase ordering
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineOrdering:
  def test_initial_phase_order(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b"), CounterPhase("c")])
    assert p.phase_names == ["a", "b", "c"]

  def test_add_phase_append(self):
    p = Pipeline(phases=[CounterPhase("a")])
    p.add_phase(CounterPhase("b"))
    assert p.phase_names == ["a", "b"]

  def test_add_phase_after(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("c")])
    p.add_phase(CounterPhase("b"), after="a")
    assert p.phase_names == ["a", "b", "c"]

  def test_add_phase_before(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("c")])
    p.add_phase(CounterPhase("b"), before="c")
    assert p.phase_names == ["a", "b", "c"]

  def test_add_phase_after_and_before_raises(self):
    p = Pipeline(phases=[CounterPhase("a")])
    with pytest.raises(ValueError, match="not both"):
      p.add_phase(CounterPhase("b"), after="a", before="a")

  def test_add_phase_unknown_anchor_raises(self):
    p = Pipeline(phases=[CounterPhase("a")])
    with pytest.raises(ValueError, match="not found"):
      p.add_phase(CounterPhase("b"), after="nonexistent")

  def test_remove_phase(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b"), CounterPhase("c")])
    p.remove_phase("b")
    assert p.phase_names == ["a", "c"]

  def test_remove_phase_unknown_raises(self):
    p = Pipeline(phases=[CounterPhase("a")])
    with pytest.raises(ValueError, match="not found"):
      p.remove_phase("nonexistent")

  def test_replace_phase(self):
    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b")])
    p.replace_phase("b", CounterPhase("new_b"))
    assert p.phase_names == ["a", "new_b"]

  def test_replace_phase_unknown_raises(self):
    p = Pipeline(phases=[CounterPhase("a")])
    with pytest.raises(ValueError, match="not found"):
      p.replace_phase("nonexistent", CounterPhase("b"))

  def test_chaining(self):
    p = Pipeline()
    result = p.add_phase(CounterPhase("a")).add_phase(CounterPhase("b"))
    assert result is p
    assert p.phase_names == ["a", "b"]

  def test_phases_property_is_copy(self):
    p = Pipeline(phases=[CounterPhase("a")])
    phases_copy = p.phases
    phases_copy.append(CounterPhase("b"))
    assert len(p.phases) == 1


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — execution
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineExecution:
  @pytest.mark.asyncio
  async def test_execute_all_phases(self):
    a, b = CounterPhase("a"), CounterPhase("b")
    p = Pipeline(phases=[a, b])
    state = _make_state()
    async for s, _ in p.execute(state):
      pass
    assert a.call_count == 1
    assert b.call_count == 1
    assert state.status == LoopStatus.completed

  @pytest.mark.asyncio
  async def test_execute_sets_running_then_completed(self):
    observed = []

    class ObserverPhase(BasePhase):
      _name = "observer"

      async def execute(self, state):
        observed.append(state.status)
        yield state, None

    p = Pipeline(phases=[ObserverPhase()])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert LoopStatus.running in observed
    assert state.status == LoopStatus.completed

  @pytest.mark.asyncio
  async def test_execute_emits_events(self):
    p = Pipeline(phases=[EventEmittingPhase()])
    state = _make_state()
    events = []
    async for s, e in p.execute(state):
      if e is not None:
        events.append(e)
    # Phase lifecycle events (PhaseStarted + PhaseCompleted) + the custom event
    assert len(events) == 3
    assert events[0].event == "PhaseStarted"  # type: ignore[attr-defined]
    assert events[1].data == "from_emitter"  # type: ignore[attr-defined]
    assert events[2].event == "PhaseCompleted"  # type: ignore[attr-defined]

  @pytest.mark.asyncio
  async def test_execute_empty_pipeline(self):
    p = Pipeline(phases=[])
    state = _make_state()
    results = []
    async for s, e in p.execute(state):
      results.append((s, e))
    assert state.status == LoopStatus.completed
    assert len(results) == 0

  @pytest.mark.asyncio
  async def test_execute_stops_on_error_status(self):
    class ErrorPhase(BasePhase):
      _name = "error_phase"

      async def execute(self, state):
        state.status = LoopStatus.error
        yield state, None

    after_phase = CounterPhase("after")
    p = Pipeline(phases=[ErrorPhase(), after_phase])
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert after_phase.call_count == 0
    assert state.status == LoopStatus.error

  @pytest.mark.asyncio
  async def test_phase_exception_sets_error_status(self):
    class BrokenPhase(BasePhase):
      _name = "broken"

      async def execute(self, state):
        raise RuntimeError("boom")
        yield  # type: ignore[unreachable]  # noqa: F841

    p = Pipeline(phases=[BrokenPhase()])
    state = _make_state()
    with pytest.raises(RuntimeError, match="boom"):
      async for _ in p.execute(state):
        pass
    assert state.status == LoopStatus.error


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — hooks
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineHooks:
  @pytest.mark.asyncio
  async def test_before_hook(self):
    order = []

    async def before_hook(state):
      order.append("before")
      return state

    class TrackedPhase(BasePhase):
      _name = "tracked"

      async def execute(self, state):
        order.append("phase")
        yield state, None

    p = Pipeline(phases=[TrackedPhase()])
    p.hook("before:tracked", before_hook)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert order == ["before", "phase"]

  @pytest.mark.asyncio
  async def test_after_hook(self):
    order = []

    async def after_hook(state):
      order.append("after")
      return state

    class TrackedPhase(BasePhase):
      _name = "tracked"

      async def execute(self, state):
        order.append("phase")
        yield state, None

    p = Pipeline(phases=[TrackedPhase()])
    p.hook("after:tracked", after_hook)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert order == ["phase", "after"]

  @pytest.mark.asyncio
  async def test_instead_hook_replaces_phase(self):
    phase = CounterPhase("target")

    async def replacement(state):
      state.extra["replaced"] = True
      yield state, None

    p = Pipeline(phases=[phase])
    p.hook("instead:target", replacement)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert phase.call_count == 0
    assert state.extra.get("replaced") is True

  @pytest.mark.asyncio
  async def test_wildcard_hook(self):
    calls = []

    async def wildcard(state):
      calls.append(state.phase)
      return state

    p = Pipeline(phases=[CounterPhase("a"), CounterPhase("b")])
    p.hook("before:*", wildcard)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert calls == ["a", "b"]

  @pytest.mark.asyncio
  async def test_hook_priority(self):
    order = []

    async def hook_low(state):
      order.append("low")
      return state

    async def hook_high(state):
      order.append("high")
      return state

    p = Pipeline(phases=[CounterPhase("a")])
    p.hook("before:a", hook_high, priority=10)
    p.hook("before:a", hook_low, priority=0)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert order == ["low", "high"]

  @pytest.mark.asyncio
  async def test_hook_decorator_syntax(self):
    p = Pipeline(phases=[CounterPhase("a")])
    called = []

    @p.hook("before:a")
    async def my_hook(state):
      called.append(True)
      return state

    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert len(called) == 1

  def test_invalid_hook_spec_raises(self):
    p = Pipeline()
    with pytest.raises(ValueError, match="Invalid hook spec"):
      p.hook("invalid", lambda s: s)

  def test_invalid_hook_timing_raises(self):
    p = Pipeline()
    with pytest.raises(ValueError, match="Invalid hook timing"):
      p.hook("unknown:phase", lambda s: s)

  @pytest.mark.asyncio
  async def test_hook_modifies_state(self):
    p = Pipeline(phases=[CounterPhase("a")])

    async def modifier(state):
      state.extra["injected"] = True
      return state

    p.hook("before:a", modifier)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.extra["injected"] is True

  @pytest.mark.asyncio
  async def test_hook_returns_none_keeps_state(self):
    p = Pipeline(phases=[CounterPhase("a")])

    async def null_hook(state):
      return None  # explicitly return None

    p.hook("before:a", null_hook)
    state = _make_state()
    async for _ in p.execute(state):
      pass
    assert state.status == LoopStatus.completed


# ═══════════════════════════════════════════════════════════════════════
# EventStream
# ═══════════════════════════════════════════════════════════════════════


class TestEventStream:
  @pytest.mark.asyncio
  async def test_subscribe_and_emit(self):
    stream = EventStream()
    received = []

    async def handler(event):
      received.append(event)

    stream.subscribe(handler)
    event = FakeEvent(data="test")
    await stream.emit(event)
    assert len(received) == 1
    assert received[0].data == "test"

  @pytest.mark.asyncio
  async def test_sync_handler(self):
    stream = EventStream()
    received = []
    stream.subscribe(lambda e: received.append(e))
    await stream.emit(FakeEvent(data="sync"))
    assert len(received) == 1

  @pytest.mark.asyncio
  async def test_unsubscribe(self):
    stream = EventStream()
    received = []

    async def handler(event):
      received.append(event)

    stream.subscribe(handler)
    stream.unsubscribe(handler)
    await stream.emit(FakeEvent(data="test"))
    assert len(received) == 0

  @pytest.mark.asyncio
  async def test_multiple_handlers(self):
    stream = EventStream()
    r1, r2 = [], []
    stream.subscribe(lambda e: r1.append(e))
    stream.subscribe(lambda e: r2.append(e))
    await stream.emit(FakeEvent(data="multi"))
    assert len(r1) == 1
    assert len(r2) == 1

  @pytest.mark.asyncio
  async def test_handler_error_doesnt_kill_others(self):
    stream = EventStream()
    r1 = []

    def bad_handler(event):
      raise RuntimeError("boom")

    stream.subscribe(bad_handler)
    stream.subscribe(lambda e: r1.append(e))
    await stream.emit(FakeEvent(data="test"))
    assert len(r1) == 1

  def test_handler_count(self):
    stream = EventStream()
    assert stream.handler_count == 0
    stream.subscribe(lambda e: None)
    assert stream.handler_count == 1

  def test_clear(self):
    stream = EventStream()
    stream.subscribe(lambda e: None)
    stream.clear()
    assert stream.handler_count == 0


# ═══════════════════════════════════════════════════════════════════════
# ToolRetry
# ═══════════════════════════════════════════════════════════════════════


class TestToolRetry:
  def test_basic(self):
    tr = ToolRetry("bad args")
    assert tr.message == "bad args"
    assert tr.max_retries == 3

  def test_custom_max_retries(self):
    tr = ToolRetry("try again", max_retries=5)
    assert tr.max_retries == 5

  def test_is_exception(self):
    tr = ToolRetry("msg")
    assert isinstance(tr, Exception)

  def test_str(self):
    tr = ToolRetry("query too short")
    assert str(tr) == "query too short"


# ═══════════════════════════════════════════════════════════════════════
# DebugConfig
# ═══════════════════════════════════════════════════════════════════════


class TestDebugConfig:
  def test_defaults(self):
    dc = DebugConfig()
    assert dc.breakpoints == set()
    assert dc.step_mode is False
    assert dc.inspector is None
    assert dc.log_state_changes is False
    assert dc.enable_trace is True

  def test_custom(self):
    dc = DebugConfig(
      breakpoints={"before:invoke_loop", "after:guard_output"},
      step_mode=True,
      log_state_changes=True,
    )
    assert "before:invoke_loop" in dc.breakpoints
    assert dc.step_mode is True


# ═══════════════════════════════════════════════════════════════════════
# SubAgent types
# ═══════════════════════════════════════════════════════════════════════


class TestSubAgentTypes:
  def test_sub_agent_config(self):
    cfg = SubAgentConfig(instructions="You are a researcher.")
    assert cfg.instructions == "You are a researcher."
    assert cfg.tools is None
    assert cfg.model is None
    assert cfg.max_tool_rounds == 15

  def test_thread_control_block(self):
    tcb = ThreadControlBlock(id="t1", goal="Research X")
    assert tcb.state == "running"
    assert tcb.result is None
    assert tcb.start_time > 0

  def test_sub_agent_policy_defaults(self):
    policy = SubAgentPolicy()
    assert policy.max_concurrent == 5
    assert policy.inherit_tools is True
    assert policy.inherit_knowledge is False
    assert policy.max_tool_rounds == 15

  def test_sub_agent_policy_custom(self):
    policy = SubAgentPolicy(max_concurrent=3, allowed_models=["openai/gpt-4o-mini"])
    assert policy.max_concurrent == 3
    assert policy.allowed_models == ["openai/gpt-4o-mini"]


# ═══════════════════════════════════════════════════════════════════════
# Pipeline event_stream integration
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineEventStream:
  @pytest.mark.asyncio
  async def test_events_flow_to_stream(self):
    p = Pipeline(phases=[EventEmittingPhase()])
    received = []
    p.subscribe(lambda e: received.append(e))
    state = _make_state()
    async for _ in p.execute(state):
      pass
    # PhaseStarted + custom event + PhaseCompleted
    assert len(received) == 3
    assert received[1].data == "from_emitter"

  @pytest.mark.asyncio
  async def test_event_stream_property(self):
    p = Pipeline()
    assert isinstance(p.event_stream, EventStream)
