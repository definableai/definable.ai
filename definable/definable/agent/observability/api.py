"""Observability API router — all /obs/api/* endpoints."""

# NOTE: Do NOT use `from __future__ import annotations` here.
# FastAPI needs runtime access to the `Request` type annotation to inject it
# as a dependency rather than treating it as a query parameter.

import asyncio
import contextlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
  from definable.agent.observability.collector import ObservabilityExporter
  from definable.agent.observability.metrics import MetricsAggregator
  from definable.agent.observability.trace_browser import TraceBrowser


def create_observability_router(
  *,
  collector: "ObservabilityExporter",
  trace_browser: "TraceBrowser",
  metrics_aggregator: "MetricsAggregator",
) -> Any:
  """Create the FastAPI APIRouter for observability endpoints.

  Args:
    collector: The live event collector.
    trace_browser: The historical trace browser.
    metrics_aggregator: The metrics aggregator.

  Returns:
    A FastAPI APIRouter.
  """
  from fastapi import APIRouter, Query, Request
  from fastapi.responses import JSONResponse, StreamingResponse

  router = APIRouter()

  # --- Health ---
  @router.get("/health")
  async def obs_health() -> Dict[str, str]:
    return {
      "status": "ok",
      "service": "observability",
      "clients": str(collector.client_count),
      "buffered_events": str(len(collector.buffer)),
    }

  # --- SSE Live Events ---
  @router.get("/events")
  async def obs_events(
    request: Request,
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    run_id: Optional[str] = Query(None, description="Filter by run ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
  ) -> StreamingResponse:
    q = collector.subscribe()

    async def event_generator():
      try:
        # Send catchup from buffer
        for evt in collector.get_recent_events(limit=100):
          if _matches_filter(evt, event_type=event_type, run_id=run_id, session_id=session_id):
            yield f"data: {json.dumps(evt, default=str)}\n\n"

        # Stream live events
        while True:
          if await request.is_disconnected():
            break
          try:
            evt = await asyncio.wait_for(q.get(), timeout=30.0)  # type: ignore[arg-type]
          except asyncio.TimeoutError:
            # Send keepalive
            yield ": keepalive\n\n"
            continue

          if evt is None:
            break  # type: ignore[unreachable]

          if _matches_filter(evt, event_type=event_type, run_id=run_id, session_id=session_id):
            yield f"data: {json.dumps(evt, default=str)}\n\n"
      except (asyncio.CancelledError, GeneratorExit):
        pass
      finally:
        collector.unsubscribe(q)

    return StreamingResponse(
      event_generator(),
      media_type="text/event-stream",
      headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    )

  # --- Sessions ---
  @router.get("/sessions")
  async def obs_sessions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
  ) -> Dict[str, Any]:
    sessions = trace_browser.list_sessions()
    total = len(sessions)
    page = sessions[offset : offset + limit]
    return {
      "sessions": [asdict(s) for s in page],
      "total": total,
      "limit": limit,
      "offset": offset,
    }

  # --- Runs within a session ---
  @router.get("/sessions/{session_id}")
  async def obs_session_runs(session_id: str) -> Any:
    try:
      runs = trace_browser.list_runs(session_id)
      return {"session_id": session_id, "runs": [asdict(r) for r in runs]}
    except FileNotFoundError:
      return JSONResponse(
        status_code=404,
        content={"error": f"Session not found: {session_id}"},
      )

  # --- Run replay ---
  @router.get("/runs/{run_id}")
  async def obs_run_replay(run_id: str, session_id: Optional[str] = Query(None)) -> Any:
    # Try live buffer first
    live_events = collector.get_events_for_run(run_id)
    if live_events:
      return _replay_dict_from_events(live_events, run_id)

    # Fall back to trace files
    if session_id:
      try:
        replay = trace_browser.load_replay(session_id, run_id)
        return _replay_to_dict(replay)
      except FileNotFoundError:
        return JSONResponse(
          status_code=404,
          content={"error": f"Session not found: {session_id}"},
        )
      except ValueError as exc:
        return JSONResponse(
          status_code=404,
          content={"error": str(exc)},
        )

    # Search all sessions for the run_id
    for session in trace_browser.list_sessions():
      try:
        replay = trace_browser.load_replay(session.session_id, run_id)
        if replay.run_id == run_id:
          return _replay_to_dict(replay)
      except (FileNotFoundError, ValueError):
        continue

    return JSONResponse(
      status_code=404,
      content={"error": f"Run not found: {run_id}"},
    )

  # --- Compare ---
  @router.get("/compare")
  async def obs_compare(
    a: str = Query(..., description="Run ID A"),
    b: str = Query(..., description="Run ID B"),
    session_a: Optional[str] = Query(None, description="Session ID for run A"),
    session_b: Optional[str] = Query(None, description="Session ID for run B"),
  ) -> Any:
    from definable.agent.replay.compare import compare_runs

    try:
      replay_a = _load_replay(collector, trace_browser, a, session_a)
      replay_b = _load_replay(collector, trace_browser, b, session_b)
    except (FileNotFoundError, ValueError) as exc:
      return JSONResponse(
        status_code=404,
        content={"error": str(exc)},
      )

    if replay_a is None:
      return JSONResponse(status_code=404, content={"error": f"Run not found: {a}"})
    if replay_b is None:
      return JSONResponse(status_code=404, content={"error": f"Run not found: {b}"})

    comparison = compare_runs(replay_a, replay_b)
    return {
      "run_a": a,
      "run_b": b,
      "content_diff": comparison.content_diff,
      "cost_diff": comparison.cost_diff,
      "token_diff": comparison.token_diff,
      "duration_diff": comparison.duration_diff,
      "tool_calls_diff": {
        "added": [asdict(tc) for tc in comparison.tool_calls_diff.added],
        "removed": [asdict(tc) for tc in comparison.tool_calls_diff.removed],
        "common": comparison.tool_calls_diff.common,
      },
    }

  # --- Metrics ---
  @router.get("/metrics")
  async def obs_metrics() -> Dict[str, Any]:
    # Combine live buffer with historical traces
    all_events = list(collector.buffer)
    snap = metrics_aggregator.compute(all_events)
    return snap.to_dict()

  # --- Metrics timeline ---
  @router.get("/metrics/timeline")
  async def obs_metrics_timeline(
    bucket: str = Query("hour", pattern="^(hour|day)$"),
    range_param: str = Query("24h", alias="range", pattern="^(24h|7d|30d)$"),
  ) -> Dict[str, Any]:
    range_hours = {"24h": 24, "7d": 168, "30d": 720}.get(range_param, 24)
    all_events = list(collector.buffer)
    buckets = metrics_aggregator.compute_timeline(all_events, bucket=bucket, range_hours=range_hours)
    return {
      "bucket": bucket,
      "range": range_param,
      "data": [
        {
          "timestamp": b.timestamp,
          "label": b.label,
          "run_count": b.run_count,
          "error_count": b.error_count,
          "tokens": b.tokens,
          "cost": round(b.cost, 6),
        }
        for b in buckets
      ],
    }

  # --- Export session JSONL ---
  @router.get("/events/export/{session_id}")
  async def obs_export_session(session_id: str) -> Any:
    path = trace_browser.get_session_path(session_id)
    if not path.is_file():
      return JSONResponse(
        status_code=404,
        content={"error": f"Session not found: {session_id}"},
      )

    async def stream_file():
      with open(path, encoding="utf-8") as f:
        for line in f:
          yield line

    return StreamingResponse(
      stream_file(),
      media_type="application/x-ndjson",
      headers={
        "Content-Disposition": f'attachment; filename="{session_id}.jsonl"',
      },
    )

  return router


# --- Helpers ---


def _matches_filter(
  evt: Dict[str, Any],
  *,
  event_type: Optional[str] = None,
  run_id: Optional[str] = None,
  session_id: Optional[str] = None,
) -> bool:
  """Check if an event matches the given filters."""
  if event_type and evt.get("event") != event_type:
    return False
  if run_id and evt.get("run_id") != run_id:
    return False
  if session_id and evt.get("session_id") != session_id:
    return False
  return True


def _load_replay(
  collector: "ObservabilityExporter",
  trace_browser: "TraceBrowser",
  run_id: str,
  session_id: Optional[str],
) -> Any:
  """Load a Replay from live buffer or trace files."""
  from definable.agent.replay.replay import Replay
  from definable.agent.run.agent import run_output_event_from_dict

  # Try live buffer first
  live_events = collector.get_events_for_run(run_id)
  if live_events:
    typed = []
    for data in live_events:
      with contextlib.suppress(Exception):
        typed.append(run_output_event_from_dict(data))
    if typed:
      return Replay.from_events(typed, run_id=run_id)

  # Try trace files
  if session_id:
    return trace_browser.load_replay(session_id, run_id)

  # Search all sessions
  for session in trace_browser.list_sessions():
    try:
      replay = trace_browser.load_replay(session.session_id, run_id)
      if replay.run_id == run_id:
        return replay
    except (FileNotFoundError, ValueError):
      continue

  return None


def _replay_dict_from_events(events: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
  """Build a replay-like dict from raw event dicts (for live buffer data)."""
  from definable.agent.replay.replay import Replay
  from definable.agent.run.agent import run_output_event_from_dict

  typed = []
  for data in events:
    with contextlib.suppress(Exception):
      typed.append(run_output_event_from_dict(data))

  if typed:
    replay = Replay.from_events(typed, run_id=run_id)
    return _replay_to_dict(replay)

  return {"run_id": run_id, "events": events}


def _replay_to_dict(replay: Any) -> Dict[str, Any]:
  """Convert a Replay object to a serializable dict."""
  return {
    "run_id": replay.run_id,
    "session_id": replay.session_id,
    "agent_id": replay.agent_id,
    "agent_name": replay.agent_name,
    "model": replay.model,
    "model_provider": replay.model_provider,
    "status": replay.status,
    "error": replay.error,
    "input": str(replay.input) if replay.input else None,
    "content": str(replay.content) if replay.content else None,
    "tokens": {
      "input_tokens": replay.tokens.input_tokens,
      "output_tokens": replay.tokens.output_tokens,
      "total_tokens": replay.tokens.total_tokens,
      "reasoning_tokens": replay.tokens.reasoning_tokens,
    },
    "cost": replay.cost,
    "duration": replay.duration,
    "tool_calls": [
      {
        "tool_name": tc.tool_name,
        "tool_args": tc.tool_args,
        "result": tc.result,
        "error": tc.error,
        "started_at": tc.started_at,
        "completed_at": tc.completed_at,
        "duration_ms": tc.duration_ms,
      }
      for tc in replay.tool_calls
    ],
    "steps": [
      {
        "step_type": s.step_type,
        "name": s.name,
        "started_at": s.started_at,
        "completed_at": s.completed_at,
        "duration_ms": s.duration_ms,
      }
      for s in replay.steps
    ],
    "knowledge_retrievals": [
      {
        "query": kr.query,
        "documents_found": kr.documents_found,
        "documents_used": kr.documents_used,
        "duration_ms": kr.duration_ms,
      }
      for kr in replay.knowledge_retrievals
    ],
    "memory_recalls": [
      {
        "query": mr.query,
        "tokens_used": mr.tokens_used,
        "chunks_included": mr.chunks_included,
        "chunks_available": mr.chunks_available,
        "duration_ms": mr.duration_ms,
      }
      for mr in replay.memory_recalls
    ],
  }
