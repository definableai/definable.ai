"""Persistence + projection pipeline for harness events.

The store owns three pieces:

  1. The DB connection + Repo objects for ``agents``, ``runs``, ``events``,
     ``spans``.
  2. An in-memory dict of :class:`RunState` per ``run_id`` so projection
     can advance event-by-event.
  3. A bounded :class:`asyncio.Queue` + background drain task. Subscribers
     call :meth:`record_event` from the EventBus emit path (sync) and
     return immediately — the actual SQL happens off the agent's critical
     path. Overflow drops the oldest entries with a single warning.

Read-side helpers (``list_runs``, ``get_run``, ``metrics``) query the DB
directly with prepared statements.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import aiosqlite

from definable.agent.core.events import Event
from definable.db import connect, migrate
from definable.db.repo import Repo
from definable.observability.projection import RunState, project
from definable.observability.schema import AgentRow, EventRow, RunRow, SpanRow

log = logging.getLogger(__name__)

QUEUE_MAX = 10_000


class TraceStore:
  """Async DB-backed sink for harness Events plus an in-process broadcast fanout.

  ``broadcast`` callbacks (used by SSE) receive every persisted event in
  order so live UIs see exactly what landed on disk.
  """

  def __init__(self, *, namespace: str = "observability", db_path: str | None = None) -> None:
    self._namespace = namespace
    self._db_path = db_path
    self._conn: aiosqlite.Connection | None = None
    self._agents: Repo[AgentRow] | None = None
    self._runs: Repo[RunRow] | None = None
    self._events: Repo[EventRow] | None = None
    self._spans: Repo[SpanRow] | None = None

    self._states: dict[str, RunState] = {}
    self._known_agents: set[str] = set()
    self._queue: asyncio.Queue[tuple[Event, str, str, str | None]] = asyncio.Queue(maxsize=QUEUE_MAX)
    self._drain_task: asyncio.Task[None] | None = None
    self._opened = False
    self._closing = False
    self._overflow_warned = False
    self._subscribers: list[Callable[[dict[str, Any]], None]] = []

  # ---- lifecycle ---------------------------------------------------------

  async def aopen(self) -> None:
    if self._opened:
      return
    self._conn = await connect(self._namespace, db_path=self._db_path)
    # Schema lives under the "observability" migration namespace regardless
    # of the connection namespace (which controls the DB file location only).
    await migrate("observability", self._conn)
    self._agents = Repo(self._conn, table="agents", model=AgentRow)
    self._runs = Repo(self._conn, table="runs", model=RunRow)
    self._events = Repo(self._conn, table="events", model=EventRow)
    self._spans = Repo(self._conn, table="spans", model=SpanRow)
    self._opened = True
    self._drain_task = asyncio.create_task(self._drain())

  async def aclose(self) -> None:
    if not self._opened:
      return
    self._closing = True
    if self._drain_task is not None:
      # Wait for queue to fully drain.
      await self._queue.join()
      self._drain_task.cancel()
      with contextlib.suppress(asyncio.CancelledError, Exception):
        await self._drain_task
      self._drain_task = None
    self._opened = False

  # ---- write path --------------------------------------------------------

  def attach(self, *, agent_name: str, agent_model: str, instructions: str | None) -> Callable[[Event], None]:
    """Return a callback that the agent's EventBus should subscribe.

    The returned callable is sync (matches `EventBus.subscribe` contract);
    it pushes onto the bounded queue and returns immediately.
    """

    def _on_event(event: Event) -> None:
      self.record_event(event, agent_name=agent_name, agent_model=agent_model, instructions=instructions)

    return _on_event

  def record_event(self, event: Event, *, agent_name: str, agent_model: str, instructions: str | None = None) -> None:
    """Enqueue one event for background persistence.

    Sync — called from the EventBus emit path. Returns immediately. If the
    queue is full, the oldest entry is dropped to make room.
    """
    item = (event, agent_name, agent_model, instructions)
    try:
      self._queue.put_nowait(item)
    except asyncio.QueueFull:
      # Drop oldest so live tails stay responsive under load.
      with contextlib.suppress(asyncio.QueueEmpty):
        self._queue.get_nowait()
        self._queue.task_done()
      with contextlib.suppress(asyncio.QueueFull):
        self._queue.put_nowait(item)
      if not self._overflow_warned:
        self._overflow_warned = True
        log.warning("TraceStore queue overflowed (>=%d events); dropping oldest", QUEUE_MAX)

  def subscribe_broadcast(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Register a fanout consumer (SSE, etc). Receives serialized events."""
    self._subscribers.append(callback)

    def _unsub() -> None:
      with contextlib.suppress(ValueError):
        self._subscribers.remove(callback)

    return _unsub

  # ---- read path ---------------------------------------------------------

  async def list_runs(self, *, agent: str | None = None, limit: int = 50, offset: int = 0) -> list[RunRow]:
    assert self._runs is not None, "TraceStore not open"
    where = "agent_id = ?" if agent else None
    params: tuple[Any, ...] = (agent,) if agent else ()
    return await self._runs.select(where=where, params=params, order_by="started_at DESC", limit=limit, offset=offset)

  async def get_run(self, run_id: str) -> RunRow | None:
    assert self._runs is not None, "TraceStore not open"
    return await self._runs.get(run_id)

  async def get_run_spans(self, run_id: str) -> list[SpanRow]:
    assert self._spans is not None, "TraceStore not open"
    return await self._spans.select(where="run_id = ?", params=(run_id,), order_by="start_ts ASC")

  async def list_events(self, run_id: str, *, after_id: int | None = None, limit: int = 200) -> list[EventRow]:
    assert self._events is not None, "TraceStore not open"
    if after_id is not None:
      return await self._events.select(where="run_id = ? AND id > ?", params=(run_id, after_id), order_by="id ASC", limit=limit)
    return await self._events.select(where="run_id = ?", params=(run_id,), order_by="id ASC", limit=limit)

  async def list_agents(self) -> list[AgentRow]:
    assert self._agents is not None, "TraceStore not open"
    return await self._agents.select(order_by="registered_at ASC")

  async def metrics(self, range_: str = "1h") -> dict[str, Any]:
    """Aggregated metrics over a recent window.

    ``range_`` is one of ``"1h"``, ``"6h"``, ``"24h"``. Returns a flat dict
    suitable for direct JSON serialization to the dashboard.
    """
    assert self._conn is not None, "TraceStore not open"
    delta = {"1h": 3600, "6h": 21600, "24h": 86400}.get(range_, 3600)
    since = time.time() - delta
    async with self._conn.execute(
      "SELECT COUNT(*), "
      "SUM(total_input_tokens), SUM(total_output_tokens), SUM(total_cost_usd), "
      "SUM(CASE WHEN status='errored' THEN 1 ELSE 0 END) "
      "FROM runs WHERE started_at >= ?",
      (since,),
    ) as cur:
      row = await cur.fetchone()
    count = int(row[0] or 0) if row else 0
    in_tok = int(row[1] or 0) if row else 0
    out_tok = int(row[2] or 0) if row else 0
    cost = float(row[3] or 0.0) if row else 0.0
    errors = int(row[4] or 0) if row else 0
    # Per-run duration histogram for p50/p95.
    async with self._conn.execute(
      "SELECT (ended_at - started_at) * 1000.0 FROM runs WHERE started_at >= ? AND ended_at IS NOT NULL ORDER BY 1 ASC",
      (since,),
    ) as cur:
      durations = [float(r[0] or 0.0) for r in await cur.fetchall()]
    return {
      "range": range_,
      "since": since,
      "runs": count,
      "errors": errors,
      "input_tokens": in_tok,
      "output_tokens": out_tok,
      "cost_usd": cost,
      "p50_ms": _percentile(durations, 0.50),
      "p95_ms": _percentile(durations, 0.95),
      "rps": count / delta if delta else 0.0,
    }

  # ---- internals ---------------------------------------------------------

  async def _drain(self) -> None:
    assert self._agents is not None and self._runs is not None and self._events is not None and self._spans is not None
    while not self._closing or not self._queue.empty():
      try:
        event, agent_name, agent_model, instructions = await asyncio.wait_for(self._queue.get(), timeout=0.25)
      except TimeoutError:
        continue
      try:
        await self._handle(event, agent_name=agent_name, agent_model=agent_model, instructions=instructions)
      except Exception:
        log.exception("TraceStore drain failed for event %s", type(event).__name__)
      finally:
        self._queue.task_done()

  async def _handle(self, event: Event, *, agent_name: str, agent_model: str, instructions: str | None) -> None:
    # Ensure parent agent row exists (idempotent insert keyed on PK).
    if agent_name not in self._known_agents:
      assert self._agents is not None and self._conn is not None
      with contextlib.suppress(aiosqlite.IntegrityError):
        await self._agents.insert(AgentRow(id=agent_name, registered_at=time.time(), model=agent_model, instructions=instructions))
      self._known_agents.add(agent_name)

    # Ensure run row exists before any events/spans reference it.
    assert self._runs is not None
    state = self._states.get(event.run_id)
    if state is None:
      run = RunRow(id=event.run_id, agent_id=agent_name, started_at=event.timestamp, status="running", input=_extract_input(event))
      try:
        await self._runs.insert(run)
      except aiosqlite.IntegrityError:
        existing = await self._runs.get(event.run_id)
        if existing is not None:
          run = existing
      # Derive provider+model from agent_model — best-effort split on "/".
      provider, model = _split_provider_model(agent_model)
      state = RunState(run=run, provider=provider, model=model)
      self._states[event.run_id] = state

    # Persist the raw event for full-fidelity replay.
    assert self._events is not None
    payload = _event_payload(event)
    await self._events.insert(
      EventRow(run_id=event.run_id, timestamp=event.timestamp, type=event.__class__.__name__, payload=payload),
    )

    # Project into Run/Span patches and persist.
    result = project(event, state)
    assert self._spans is not None
    for span in result.new_spans:
      if span.end_ts is not None:
        # Single-row span (memory access) — insert as already-closed.
        await self._spans.insert(span)
    for span in result.closed_spans:
      if span in result.new_spans:
        continue  # already inserted above
      await self._spans.insert(span)
    await self._upsert_run(result.run)
    if result.closed_run:
      self._states.pop(event.run_id, None)

    # Fan out to live subscribers (SSE).
    self._broadcast({
      "type": event.__class__.__name__,
      "run_id": event.run_id,
      "agent": agent_name,
      "timestamp": event.timestamp,
      "payload": payload,
    })

  async def _upsert_run(self, row: RunRow) -> None:
    assert self._runs is not None
    fields = {f.name: getattr(row, f.name) for f in dataclasses.fields(row) if f.name != "id"}
    updated = await self._runs.update(row.id, **fields)
    if updated == 0:
      try:
        await self._runs.insert(row)
      except aiosqlite.IntegrityError:
        # Race: another writer inserted it; fall back to update.
        await self._runs.update(row.id, **fields)

  def _broadcast(self, payload: dict[str, Any]) -> None:
    for sub in list(self._subscribers):
      try:
        sub(payload)
      except Exception:
        log.exception("TraceStore broadcast subscriber raised; continuing")


# ---- helpers --------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
  if not sorted_vals:
    return 0.0
  idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
  return float(sorted_vals[idx])


def _event_payload(event: Event) -> dict[str, Any]:
  d = dataclasses.asdict(event)
  # Re-render via JSON round-trip to flatten any nested dataclasses (e.g. ToolCall).
  return json.loads(json.dumps(d, default=str))


def _extract_input(event: Event) -> str | None:
  # The harness does not currently emit an "input" event for the user prompt;
  # leave None and let later patches populate via agent metadata.
  return None


def _split_provider_model(s: str) -> tuple[str | None, str]:
  if "/" in s:
    provider, model = s.split("/", 1)
    return provider, model
  return None, s
