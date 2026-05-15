"""Project a stream of harness :class:`Event` objects into Run + Span rows.

Pure function design — no I/O, no async, no DB. The store layer feeds
events into ``project()`` and persists whatever patches come back.

A :class:`RunState` lives per ``run_id`` in the caller's memory:
trackers for the open turn's LLM span, currently-open tool spans, and
running token + cost totals. Closing the run drops the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from definable.agent.core.events import (
  Event,
  MemoryAccessed,
  ModelResponded,
  RunCompleted,
  RunErrored,
  ToolCallCompleted,
  ToolCallFailed,
  ToolCallStarted,
  TurnStarted,
)
from definable.observability.pricing import cost as compute_cost
from definable.observability.schema import RunRow, SpanRow


@dataclass
class RunState:
  """Mutable per-run accumulator. Owned by the caller (TraceStore)."""

  run: RunRow
  open_llm_span: SpanRow | None = None
  open_tool_spans: dict[str, SpanRow] = field(default_factory=dict)
  provider: str | None = None
  model: str | None = None


@dataclass(frozen=True)
class ProjectionResult:
  """Patch set produced by one event.

  - ``run``: latest authoritative ``RunRow`` (caller upserts).
  - ``new_spans``: ``SpanRow``s freshly opened (no ``end_ts``).
  - ``closed_spans``: ``SpanRow``s newly completed (have ``end_ts``).
  - ``closed_run``: True when this event terminated the run.
  """

  run: RunRow
  new_spans: list[SpanRow] = field(default_factory=list)
  closed_spans: list[SpanRow] = field(default_factory=list)
  closed_run: bool = False


def project(event: Event, state: RunState) -> ProjectionResult:
  """Advance ``state`` by one event; return patches the store must apply.

  ``state`` is mutated in place — caller owns lifetime.
  """
  ts = event.timestamp

  if isinstance(event, TurnStarted):
    span = SpanRow(run_id=state.run.id, kind="llm", name=state.model or "model", start_ts=ts)
    state.open_llm_span = span
    state.run = replace(state.run, turns=event.snapshot.turn)
    return ProjectionResult(run=state.run, new_spans=[span])

  if isinstance(event, ModelResponded):
    closed: list[SpanRow] = []
    if state.open_llm_span is not None:
      dur_ms = max(0.0, (ts - state.open_llm_span.start_ts) * 1000.0)
      closed_span = SpanRow(
        run_id=state.run.id,
        kind="llm",
        name=state.open_llm_span.name,
        start_ts=state.open_llm_span.start_ts,
        end_ts=ts,
        duration_ms=dur_ms,
        status="ok",
        metadata={"usage": event.usage or {}},
      )
      closed.append(closed_span)
      state.open_llm_span = None
    if event.usage:
      added_in = int(event.usage.get("input_tokens", 0))
      added_out = int(event.usage.get("output_tokens", 0))
      added_cached = int(event.usage.get("cache_read_tokens", 0))
      added_cost = compute_cost(state.provider, state.model, event.usage)
      state.run = replace(
        state.run,
        total_input_tokens=state.run.total_input_tokens + added_in,
        total_output_tokens=state.run.total_output_tokens + added_out,
        total_cached_tokens=state.run.total_cached_tokens + added_cached,
        total_cost_usd=state.run.total_cost_usd + added_cost,
      )
    return ProjectionResult(run=state.run, closed_spans=closed)

  if isinstance(event, ToolCallStarted):
    span = SpanRow(run_id=state.run.id, kind="tool", name=event.call.name, start_ts=ts)
    state.open_tool_spans[event.call.id] = span
    return ProjectionResult(run=state.run, new_spans=[span])

  if isinstance(event, ToolCallCompleted):
    return _close_tool_span(state, event.call.id, ts, status="ok", metadata={"output": _preview(event.output)})

  if isinstance(event, ToolCallFailed):
    return _close_tool_span(state, event.call.id, ts, status="err", metadata={"error": event.error})

  if isinstance(event, MemoryAccessed):
    span = SpanRow(
      run_id=state.run.id,
      kind="memory",
      name=event.op,
      start_ts=ts,
      end_ts=ts,
      duration_ms=0.0,
      status="ok",
      metadata={"key": event.key},
    )
    return ProjectionResult(run=state.run, new_spans=[span], closed_spans=[span])

  if isinstance(event, RunCompleted):
    state.run = replace(
      state.run,
      ended_at=ts,
      status="completed",
      output=event.content,
      exit_reason=event.exit_reason,
      turns=event.turns,
    )
    return ProjectionResult(run=state.run, closed_run=True)

  if isinstance(event, RunErrored):
    state.run = replace(
      state.run,
      ended_at=ts,
      status="errored",
      error=event.error,
      turns=event.turns,
    )
    return ProjectionResult(run=state.run, closed_run=True)

  # Unknown event (e.g. StreamChunkEvent) — no projection change.
  return ProjectionResult(run=state.run)


def _close_tool_span(state: RunState, call_id: str, ts: float, *, status: str, metadata: dict[str, Any]) -> ProjectionResult:
  open_span = state.open_tool_spans.pop(call_id, None)
  if open_span is None:
    return ProjectionResult(run=state.run)
  dur_ms = max(0.0, (ts - open_span.start_ts) * 1000.0)
  closed = SpanRow(
    run_id=state.run.id,
    kind="tool",
    name=open_span.name,
    start_ts=open_span.start_ts,
    end_ts=ts,
    duration_ms=dur_ms,
    status=status,
    metadata=metadata,
  )
  return ProjectionResult(run=state.run, closed_spans=[closed])


def _preview(value: Any, limit: int = 200) -> str:
  """Truncate a tool output for span metadata so single rows stay small."""
  s = str(value)
  if len(s) <= limit:
    return s
  return s[:limit] + "…"
