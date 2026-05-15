from __future__ import annotations

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.events import (
  MemoryAccessed,
  ModelResponded,
  RunCompleted,
  RunErrored,
  ToolCall,
  ToolCallCompleted,
  ToolCallFailed,
  ToolCallStarted,
  TurnStarted,
)
from definable.observability.projection import RunState, project
from definable.observability.schema import RunRow


def _state(run_id: str = "r1", *, provider: str = "openai", model: str = "gpt-4o") -> RunState:
  run = RunRow(id=run_id, agent_id="a", started_at=0.0)
  return RunState(run=run, provider=provider, model=model)


def _snapshot(turn: int) -> TurnSnapshot:
  return TurnSnapshot(turn=turn, context_tokens=0, messages=0, available_tools=[], memory_files=[])


def test_turn_started_opens_llm_span() -> None:
  state = _state()
  result = project(TurnStarted(run_id="r1", timestamp=1.0, snapshot=_snapshot(1)), state)
  assert len(result.new_spans) == 1
  assert result.new_spans[0].kind == "llm"
  assert result.new_spans[0].end_ts is None
  assert state.open_llm_span is not None
  assert state.run.turns == 1


def test_model_responded_closes_llm_span_and_adds_usage() -> None:
  state = _state()
  project(TurnStarted(run_id="r1", timestamp=1.0, snapshot=_snapshot(1)), state)
  result = project(
    ModelResponded(
      run_id="r1",
      timestamp=1.5,
      content="hi",
      tool_calls=[],
      usage={"input_tokens": 100, "output_tokens": 50},
    ),
    state,
  )
  assert state.open_llm_span is None
  assert len(result.closed_spans) == 1
  closed = result.closed_spans[0]
  assert closed.kind == "llm"
  assert closed.duration_ms == 500.0
  assert closed.status == "ok"
  assert state.run.total_input_tokens == 100
  assert state.run.total_output_tokens == 50
  # Known model — cost is non-zero.
  assert state.run.total_cost_usd > 0


def test_tool_call_open_then_complete() -> None:
  state = _state()
  call = ToolCall(id="t1", name="search", args={})
  project(ToolCallStarted(run_id="r1", timestamp=2.0, call=call), state)
  result = project(ToolCallCompleted(run_id="r1", timestamp=2.25, call=call, output="hit"), state)
  assert len(result.closed_spans) == 1
  span = result.closed_spans[0]
  assert span.kind == "tool"
  assert span.name == "search"
  assert span.duration_ms == 250.0
  assert span.status == "ok"
  assert span.metadata.get("output") == "hit"


def test_tool_call_failure_marks_err() -> None:
  state = _state()
  call = ToolCall(id="t2", name="fetch", args={})
  project(ToolCallStarted(run_id="r1", timestamp=3.0, call=call), state)
  result = project(ToolCallFailed(run_id="r1", timestamp=3.1, call=call, error="boom"), state)
  assert result.closed_spans[0].status == "err"
  assert result.closed_spans[0].metadata.get("error") == "boom"


def test_memory_access_zero_duration_span() -> None:
  state = _state()
  result = project(MemoryAccessed(run_id="r1", timestamp=4.0, op="read", key="ctx.md"), state)
  assert len(result.new_spans) == 1
  span = result.new_spans[0]
  assert span.kind == "memory"
  assert span.duration_ms == 0.0
  assert span.metadata.get("key") == "ctx.md"


def test_run_completed_closes_run() -> None:
  state = _state()
  result = project(RunCompleted(run_id="r1", timestamp=5.0, content="done", turns=2), state)
  assert result.closed_run is True
  assert state.run.status == "completed"
  assert state.run.output == "done"
  assert state.run.ended_at == 5.0


def test_run_errored_closes_run() -> None:
  state = _state()
  result = project(RunErrored(run_id="r1", timestamp=6.0, error="oops", turns=1), state)
  assert result.closed_run is True
  assert state.run.status == "errored"
  assert state.run.error == "oops"


def test_full_run_sequence() -> None:
  state = _state()
  events: list = [
    TurnStarted(run_id="r1", timestamp=1.0, snapshot=_snapshot(1)),
    ModelResponded(
      run_id="r1",
      timestamp=1.5,
      content=None,
      tool_calls=[ToolCall(id="t1", name="sum", args={"x": 1})],
      usage={"input_tokens": 80, "output_tokens": 10},
    ),
    ToolCallStarted(run_id="r1", timestamp=1.6, call=ToolCall(id="t1", name="sum", args={"x": 1})),
    ToolCallCompleted(run_id="r1", timestamp=1.7, call=ToolCall(id="t1", name="sum", args={"x": 1}), output=42),
    TurnStarted(run_id="r1", timestamp=1.8, snapshot=_snapshot(2)),
    ModelResponded(
      run_id="r1",
      timestamp=2.0,
      content="42",
      tool_calls=[],
      usage={"input_tokens": 30, "output_tokens": 5},
    ),
    RunCompleted(run_id="r1", timestamp=2.05, content="42", turns=2),
  ]
  new_spans = 0
  closed_spans = 0
  closed_run = False
  for e in events:
    r = project(e, state)
    new_spans += len(r.new_spans)
    closed_spans += len(r.closed_spans)
    closed_run = closed_run or r.closed_run
  # 2 llm opens + 1 tool open = 3 new; 2 llm closes + 1 tool close = 3 closed
  assert new_spans == 3
  assert closed_spans == 3
  assert closed_run is True
  assert state.run.total_input_tokens == 110
  assert state.run.total_output_tokens == 15
