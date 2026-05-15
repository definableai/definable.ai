from __future__ import annotations

from uuid import uuid4

import pytest

from definable.db import close_all
from definable.observability.sse import install_sse_routes
from definable.observability.store import TraceStore


@pytest.fixture(autouse=True)
async def _clean():
  await close_all()
  yield
  await close_all()


async def test_sse_route_registered_on_app() -> None:
  from fastapi import FastAPI

  store = TraceStore(namespace=f"sse_test_{uuid4().hex[:8]}", db_path=":memory:")
  await store.aopen()
  try:
    app = FastAPI()
    install_sse_routes(app, store)
    routes = [getattr(r, "path", None) for r in app.router.routes]
    assert "/api/stream" in routes
  finally:
    await store.aclose()
