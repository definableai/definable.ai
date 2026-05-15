"""REST endpoints for the observability dashboard.

Read-only JSON over the :class:`TraceStore`. Mounted into the
singleton FastAPI app by :class:`ObservabilityServer`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from fastapi import FastAPI

  from definable.observability.store import TraceStore


def install_rest_routes(app: FastAPI, store: TraceStore) -> None:
  """Attach the JSON REST endpoints. Idempotent — call once per app."""
  from fastapi import HTTPException

  @app.get("/api/agents")
  async def get_agents() -> list[dict[str, Any]]:
    rows = await store.list_agents()
    return [dataclasses.asdict(r) for r in rows]

  @app.get("/api/runs")
  async def get_runs(agent: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    rows = await store.list_runs(agent=agent, limit=limit, offset=offset)
    return [dataclasses.asdict(r) for r in rows]

  @app.get("/api/runs/{run_id}")
  async def get_run(run_id: str) -> dict[str, Any]:
    row = await store.get_run(run_id)
    if row is None:
      raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    spans = await store.get_run_spans(run_id)
    events = await store.list_events(run_id, limit=200)
    return {
      "run": dataclasses.asdict(row),
      "spans": [dataclasses.asdict(s) for s in spans],
      "events": [dataclasses.asdict(e) for e in events],
    }

  @app.get("/api/runs/{run_id}/events")
  async def get_run_events(run_id: str, after: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    rows = await store.list_events(run_id, after_id=after, limit=limit)
    return [dataclasses.asdict(r) for r in rows]

  @app.get("/api/metrics")
  async def get_metrics(range: str = "1h") -> dict[str, Any]:  # noqa: A002 — query name matches frontend
    if range not in {"1h", "6h", "24h"}:
      raise HTTPException(status_code=422, detail="range must be one of 1h, 6h, 24h")
    return await store.metrics(range)
