from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from definable.observability.playground import install_playground_routes
from definable.observability.registry import AgentRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
  AgentRegistry.get().clear()
  yield
  AgentRegistry.get().clear()


async def test_unknown_agent_returns_404() -> None:
  from fastapi import FastAPI

  app = FastAPI()
  install_playground_routes(app)
  async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
    r = await ac.post("/api/playground/run", json={"agent": "nope", "input": "hi"})
  assert r.status_code == 404


async def test_invalid_body_returns_422() -> None:
  from fastapi import FastAPI

  app = FastAPI()
  install_playground_routes(app)
  async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
    r = await ac.post("/api/playground/run", json={"input": "missing agent"})
  assert r.status_code == 422
