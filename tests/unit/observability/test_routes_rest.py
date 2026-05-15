from __future__ import annotations

import time
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.events import ModelResponded, RunCompleted, TurnStarted
from definable.db import close_all
from definable.observability.routes import install_rest_routes
from definable.observability.store import TraceStore


@pytest.fixture(autouse=True)
async def _clean():
  await close_all()
  yield
  await close_all()


def _snapshot(turn: int) -> TurnSnapshot:
  return TurnSnapshot(turn=turn, context_tokens=0, messages=0, available_tools=[], memory_files=[])


async def _make_app_with_store() -> tuple:
  from fastapi import FastAPI

  store = TraceStore(namespace=f"rest_test_{uuid4().hex[:8]}", db_path=":memory:")
  await store.aopen()
  app = FastAPI()
  install_rest_routes(app, store)
  return app, store


async def _seed_one_run(store: TraceStore, *, run_id: str, agent: str, model: str) -> None:
  now = time.time()
  store.record_event(TurnStarted(run_id=run_id, timestamp=now, snapshot=_snapshot(1)), agent_name=agent, agent_model=model)
  store.record_event(
    ModelResponded(run_id=run_id, timestamp=now + 0.5, content="hi", tool_calls=[], usage={"input_tokens": 10, "output_tokens": 5}),
    agent_name=agent,
    agent_model=model,
  )
  store.record_event(RunCompleted(run_id=run_id, timestamp=now + 0.6, content="hi", turns=1), agent_name=agent, agent_model=model)
  await store._queue.join()


async def test_get_agents_empty() -> None:
  app, store = await _make_app_with_store()
  try:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/agents")
    assert r.status_code == 200
    assert r.json() == []
  finally:
    await store.aclose()


async def test_get_runs_returns_seeded() -> None:
  app, store = await _make_app_with_store()
  try:
    await _seed_one_run(store, run_id="r1", agent="a", model="openai/gpt-4o")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 1
    assert runs[0]["id"] == "r1"
    assert runs[0]["status"] == "completed"
  finally:
    await store.aclose()


async def test_get_run_detail_includes_spans_and_events() -> None:
  app, store = await _make_app_with_store()
  try:
    await _seed_one_run(store, run_id="r2", agent="a", model="openai/gpt-4o")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/runs/r2")
    assert r.status_code == 200
    body = r.json()
    assert body["run"]["id"] == "r2"
    assert len(body["spans"]) >= 1
    assert len(body["events"]) == 3
  finally:
    await store.aclose()


async def test_get_run_missing_returns_404() -> None:
  app, store = await _make_app_with_store()
  try:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/runs/nope")
    assert r.status_code == 404
  finally:
    await store.aclose()


async def test_get_metrics_known_range() -> None:
  app, store = await _make_app_with_store()
  try:
    await _seed_one_run(store, run_id="r3", agent="a", model="openai/gpt-4o")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/metrics?range=1h")
    assert r.status_code == 200
    body = r.json()
    assert body["runs"] == 1
    assert body["input_tokens"] == 10
  finally:
    await store.aclose()


async def test_get_metrics_invalid_range_422() -> None:
  app, store = await _make_app_with_store()
  try:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
      r = await ac.get("/api/metrics?range=7d")
    assert r.status_code == 422
  finally:
    await store.aclose()
