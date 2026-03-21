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
  agent: Any = None,
) -> Any:
  """Create the FastAPI APIRouter for observability endpoints.

  Args:
    collector: The live event collector.
    trace_browser: The historical trace browser.
    metrics_aggregator: The metrics aggregator.
    agent: Optional Agent instance for the chat endpoint.

  Returns:
    A FastAPI APIRouter.
  """
  from fastapi import APIRouter, Query, Request
  from fastapi.responses import JSONResponse, StreamingResponse

  router = APIRouter()

  # Chat conversation history keyed by session
  _chat_sessions: Dict[str, List[Any]] = {}

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
    _disconnected = False

    async def _watch_disconnect() -> None:
      """Poll for client disconnect and inject sentinel to unblock q.get()."""
      nonlocal _disconnected
      while not _disconnected:
        if await request.is_disconnected():
          _disconnected = True
          # Wake up the generator immediately
          with contextlib.suppress(asyncio.QueueFull):
            q.put_nowait(None)
          return
        await asyncio.sleep(1.0)

    async def event_generator():
      nonlocal _disconnected
      watcher = asyncio.create_task(_watch_disconnect())
      try:
        # Send catchup from buffer
        for evt in collector.get_recent_events(limit=100):
          if _disconnected:
            return
          if _matches_filter(evt, event_type=event_type, run_id=run_id, session_id=session_id):
            yield f"data: {json.dumps(evt, default=str)}\n\n"

        # Stream live events
        while not _disconnected:
          try:
            evt = await asyncio.wait_for(q.get(), timeout=15.0)  # type: ignore[arg-type]
          except asyncio.TimeoutError:
            if _disconnected:
              break  # type: ignore[unreachable]
            yield ": keepalive\n\n"
            continue

          if evt is None or _disconnected:
            break  # type: ignore[unreachable]

          if _matches_filter(evt, event_type=event_type, run_id=run_id, session_id=session_id):
            yield f"data: {json.dumps(evt, default=str)}\n\n"
      except (asyncio.CancelledError, GeneratorExit, ConnectionError, OSError):
        pass
      finally:
        _disconnected = True
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
          await watcher
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
      return {"run_id": run_id, "events": live_events}

    # Fall back to trace files
    if session_id:
      try:
        return trace_browser.load_replay(session_id, run_id)
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
        result = trace_browser.load_replay(session.session_id, run_id)
        if result.get("run_id") == run_id:
          return result
      except (FileNotFoundError, ValueError):
        continue

    return JSONResponse(
      status_code=404,
      content={"error": f"Run not found: {run_id}"},
    )

  # --- Compare (stub — replay module removed) ---
  @router.get("/compare")
  async def obs_compare(
    a: str = Query(..., description="Run ID A"),
    b: str = Query(..., description="Run ID B"),
    session_a: Optional[str] = Query(None, description="Session ID for run A"),
    session_b: Optional[str] = Query(None, description="Session ID for run B"),
  ) -> Any:
    return JSONResponse(
      status_code=501,
      content={"error": "Run comparison not available — replay module removed"},
    )

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
    bucket: str = Query("auto", pattern="^(auto|5min|30min|hour|day)$"),
    range_param: str = Query("auto", alias="range", pattern="^(auto|1h|6h|24h|7d|30d)$"),
  ) -> Dict[str, Any]:
    range_hours_map = {"auto": 0, "1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
    range_hours = range_hours_map.get(range_param, 0)
    all_events = list(collector.buffer)
    result = metrics_aggregator.compute_timeline(all_events, bucket=bucket, range_hours=range_hours)
    return {
      "bucket": result.resolved_bucket,
      "bucket_seconds": result.bucket_seconds,
      "range": range_param,
      "range_hours": round(result.range_hours, 2),
      "data": [
        {
          "timestamp": b.timestamp,
          "label": b.label,
          "run_count": b.run_count,
          "error_count": b.error_count,
          "tokens": b.tokens,
          "cost": round(b.cost, 6),
        }
        for b in result.buckets
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

  # --- Chat ---
  @router.post("/chat")
  async def obs_chat(request: Request) -> Any:
    if agent is None:
      return JSONResponse(status_code=501, content={"error": "No agent configured for chat"})

    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not message:
      return JSONResponse(status_code=400, content={"error": "Empty message"})

    # Retrieve or initialize conversation history
    prev_messages = _chat_sessions.get(session_id)

    try:
      result = await agent.arun(
        message,
        messages=prev_messages,
      )
      # Store updated conversation for multi-turn
      _chat_sessions[session_id] = result.messages

      return {
        "content": result.content,
        "run_id": result.run_id,
        "session_id": session_id,
        "tokens": {
          "input_tokens": result.metrics.input_tokens if result.metrics else 0,
          "output_tokens": result.metrics.output_tokens if result.metrics else 0,
          "total_tokens": result.metrics.total_tokens if result.metrics else 0,
        },
        "cost": result.metrics.cost if result.metrics else 0,
      }
    except Exception as exc:
      return JSONResponse(status_code=500, content={"error": str(exc)})

  @router.post("/chat/reset")
  async def obs_chat_reset(request: Request) -> Dict[str, str]:
    body = await request.json()
    session_id = body.get("session_id", "default")
    _chat_sessions.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}

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
