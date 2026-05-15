from __future__ import annotations

import pytest

from definable.db import connection as conn_mod
from definable.db.connection import close_all, connect


@pytest.fixture(autouse=True)
async def _clean_conns():
  """Ensure no leaked module-level connections across tests."""
  await close_all()
  yield
  await close_all()


async def test_connect_returns_same_instance_for_same_namespace() -> None:
  a = await connect("test_ns", db_path=":memory:")
  b = await connect("test_ns", db_path=":memory:")
  assert a is b


async def test_connect_separate_namespaces() -> None:
  a = await connect("ns_a", db_path=":memory:")
  b = await connect("ns_b", db_path=":memory:")
  assert a is not b


async def test_pragmas_applied() -> None:
  c = await connect("pragma_check", db_path=":memory:")
  async with c.execute("PRAGMA foreign_keys") as cur:
    row = await cur.fetchone()
    assert row is not None
    assert int(row[0]) == 1
  async with c.execute("PRAGMA synchronous") as cur:
    row = await cur.fetchone()
    assert row is not None
    # 1 == NORMAL
    assert int(row[0]) == 1


async def test_close_removes_from_cache() -> None:
  await connect("evict_me", db_path=":memory:")
  assert "evict_me" in conn_mod._CONNS
  await conn_mod.close("evict_me")
  assert "evict_me" not in conn_mod._CONNS
