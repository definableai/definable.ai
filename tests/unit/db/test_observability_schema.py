from __future__ import annotations

import pytest

from definable.db import close_all
from definable.db.connection import connect
from definable.db.migrate import migrate


@pytest.fixture(autouse=True)
async def _clean_conns():
  await close_all()
  yield
  await close_all()


async def test_observability_init_creates_all_tables() -> None:
  conn = await connect("obs_schema", db_path=":memory:")
  applied = await migrate("observability", conn)
  assert applied == 1

  async with conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name") as cur:
    tables = {r[0] for r in await cur.fetchall()}
  assert {"agents", "runs", "events", "spans", "schema_migrations"}.issubset(tables)

  async with conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name") as cur:
    idx = {r[0] for r in await cur.fetchall()}
  assert "idx_runs_agent_started" in idx
  assert "idx_events_run_ts" in idx
  assert "idx_spans_run" in idx


async def test_observability_idempotent() -> None:
  conn = await connect("obs_idem", db_path=":memory:")
  assert await migrate("observability", conn) == 1
  assert await migrate("observability", conn) == 0


async def test_observability_insert_round_trip() -> None:
  conn = await connect("obs_insert", db_path=":memory:")
  await migrate("observability", conn)
  await conn.execute("INSERT INTO agents (id, registered_at, model) VALUES (?, ?, ?)", ("a1", 1.0, "gpt-4o"))
  await conn.execute(
    "INSERT INTO runs (id, agent_id, started_at, status, turns) VALUES (?, ?, ?, ?, ?)",
    ("r1", "a1", 1.0, "completed", 2),
  )
  await conn.execute(
    "INSERT INTO events (run_id, timestamp, type, payload) VALUES (?, ?, ?, ?)",
    ("r1", 1.1, "TurnStarted", "{}"),
  )
  await conn.execute(
    "INSERT INTO spans (run_id, kind, name, start_ts, status) VALUES (?, ?, ?, ?, ?)",
    ("r1", "llm", "gpt-4o", 1.2, "ok"),
  )
  await conn.commit()
  async with conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", ("r1",)) as cur:
    row = await cur.fetchone()
    assert row is not None and int(row[0]) == 1
  async with conn.execute("SELECT COUNT(*) FROM spans WHERE run_id=?", ("r1",)) as cur:
    row = await cur.fetchone()
    assert row is not None and int(row[0]) == 1
