from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.events import (
  ModelResponded,
  RunCompleted,
  ToolCall,
  ToolCallCompleted,
  ToolCallStarted,
  TurnStarted,
)
from definable.db import close_all
from definable.observability.store import TraceStore


@pytest.fixture(autouse=True)
async def _clean():
  await close_all()
  yield
  await close_all()


def _snapshot(turn: int) -> TurnSnapshot:
  return TurnSnapshot(turn=turn, context_tokens=0, messages=0, available_tools=[], memory_files=[])


async def _make_store() -> TraceStore:
  store = TraceStore(namespace=f"obs_test_{uuid4().hex[:8]}", db_path=":memory:")
  await store.aopen()
  return store


async def test_open_creates_schema() -> None:
  store = await _make_store()
  try:
    assert await store.list_runs() == []
    assert await store.list_agents() == []
  finally:
    await store.aclose()


async def test_records_run_with_spans_and_tokens() -> None:
  store = await _make_store()
  try:
    run_id = "run-1"
    name, model = "agent-x", "openai/gpt-4o"
    events = [
      TurnStarted(run_id=run_id, timestamp=1.0, snapshot=_snapshot(1)),
      ModelResponded(
        run_id=run_id,
        timestamp=1.5,
        content=None,
        tool_calls=[ToolCall(id="t1", name="lookup", args={})],
        usage={"input_tokens": 100, "output_tokens": 25},
      ),
      ToolCallStarted(run_id=run_id, timestamp=1.6, call=ToolCall(id="t1", name="lookup", args={})),
      ToolCallCompleted(run_id=run_id, timestamp=1.8, call=ToolCall(id="t1", name="lookup", args={}), output="ok"),
      TurnStarted(run_id=run_id, timestamp=1.9, snapshot=_snapshot(2)),
      ModelResponded(
        run_id=run_id,
        timestamp=2.0,
        content="hi",
        tool_calls=[],
        usage={"input_tokens": 50, "output_tokens": 10},
      ),
      RunCompleted(run_id=run_id, timestamp=2.05, content="hi", turns=2),
    ]
    for e in events:
      store.record_event(e, agent_name=name, agent_model=model)

    await store._queue.join()

    runs = await store.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.status == "completed"
    assert r.turns == 2
    assert r.total_input_tokens == 150
    assert r.total_output_tokens == 35
    assert r.total_cost_usd > 0
    spans = await store.get_run_spans(run_id)
    # 2 closed llm spans + 1 closed tool span
    assert len(spans) == 3
    kinds = sorted(s.kind for s in spans)
    assert kinds == ["llm", "llm", "tool"]
    events_persisted = await store.list_events(run_id)
    assert len(events_persisted) == len(events)
  finally:
    await store.aclose()


async def test_metrics_aggregate() -> None:
  import time as _t

  store = await _make_store()
  try:
    now = _t.time()
    for i in range(3):
      run_id = f"run-{i}"
      store.record_event(TurnStarted(run_id=run_id, timestamp=now, snapshot=_snapshot(1)), agent_name="a", agent_model="openai/gpt-4o")
      store.record_event(
        ModelResponded(
          run_id=run_id,
          timestamp=now + 0.5,
          content="x",
          tool_calls=[],
          usage={"input_tokens": 100, "output_tokens": 20},
        ),
        agent_name="a",
        agent_model="openai/gpt-4o",
      )
      store.record_event(RunCompleted(run_id=run_id, timestamp=now + 0.6, content="x", turns=1), agent_name="a", agent_model="openai/gpt-4o")
    await store._queue.join()

    m = await store.metrics("1h")
    assert m["runs"] == 3
    assert m["input_tokens"] == 300
    assert m["output_tokens"] == 60
    assert m["cost_usd"] > 0
    assert m["p50_ms"] >= 0
  finally:
    await store.aclose()


async def test_subscribe_broadcast_receives_events() -> None:
  store = await _make_store()
  received: list[dict] = []
  unsub = store.subscribe_broadcast(received.append)
  try:
    store.record_event(TurnStarted(run_id="r1", timestamp=1.0, snapshot=_snapshot(1)), agent_name="a", agent_model="openai/gpt-4o")
    await store._queue.join()
    # Allow a few loop turns for the broadcast to fire.
    await asyncio.sleep(0)
    assert len(received) >= 1
    assert received[0]["type"] == "TurnStarted"
  finally:
    unsub()
    await store.aclose()
