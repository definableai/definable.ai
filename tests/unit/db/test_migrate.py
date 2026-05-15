from __future__ import annotations

from pathlib import Path

import pytest

from definable.db import close_all
from definable.db.connection import connect
from definable.db.migrate import migrate


@pytest.fixture(autouse=True)
async def _clean_conns():
  await close_all()
  yield
  await close_all()


@pytest.fixture
def fixtures_root(tmp_path: Path) -> Path:
  """Two-migration fixture namespace under a temp root."""
  ns = tmp_path / "test_ns"
  ns.mkdir(parents=True)
  (ns / "0001_a.sql").write_text("CREATE TABLE a (id INTEGER PRIMARY KEY);")
  (ns / "0002_b.sql").write_text("CREATE TABLE b (id INTEGER PRIMARY KEY, val TEXT);")
  return tmp_path


async def test_first_run_applies_all(fixtures_root: Path) -> None:
  conn = await connect("mig_first", db_path=":memory:")
  applied = await migrate("test_ns", conn, root=fixtures_root)
  assert applied == 2
  async with conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name") as cur:
    rows = await cur.fetchall()
  names = {r[0] for r in rows}
  assert {"a", "b", "schema_migrations"}.issubset(names)


async def test_second_run_is_noop(fixtures_root: Path) -> None:
  conn = await connect("mig_idem", db_path=":memory:")
  first = await migrate("test_ns", conn, root=fixtures_root)
  second = await migrate("test_ns", conn, root=fixtures_root)
  assert first == 2
  assert second == 0


async def test_recorded_versions(fixtures_root: Path) -> None:
  conn = await connect("mig_rec", db_path=":memory:")
  await migrate("test_ns", conn, root=fixtures_root)
  async with conn.execute("SELECT version, name FROM schema_migrations ORDER BY version") as cur:
    rows = await cur.fetchall()
  assert [(int(r[0]), r[1]) for r in rows] == [(1, "a"), (2, "b")]


async def test_malformed_sql_rolls_back(tmp_path: Path) -> None:
  ns = tmp_path / "bad_ns"
  ns.mkdir(parents=True)
  (ns / "0001_ok.sql").write_text("CREATE TABLE ok (id INTEGER);")
  (ns / "0002_bad.sql").write_text("CREATE TABLE x (id INTEGER); NOT_REAL_SQL;")
  conn = await connect("mig_bad", db_path=":memory:")
  with pytest.raises(Exception):  # noqa: B017 — driver-specific exception type varies
    await migrate("bad_ns", conn, root=tmp_path)
  # 0001 should have committed; 0002 was rolled back so x does not exist.
  async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
    rows = await cur.fetchall()
  names = {r[0] for r in rows}
  assert "ok" in names
  assert "x" not in names


async def test_unknown_namespace_is_zero(tmp_path: Path) -> None:
  conn = await connect("mig_unk", db_path=":memory:")
  applied = await migrate("nope", conn, root=tmp_path)
  assert applied == 0
