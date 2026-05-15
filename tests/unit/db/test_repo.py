from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from definable.db import close_all
from definable.db.connection import connect
from definable.db.repo import Repo


@dataclass(frozen=True, kw_only=True)
class Row:
  id: str
  name: str
  payload: dict[str, Any]
  count: int = 0


@pytest.fixture(autouse=True)
async def _clean_conns():
  await close_all()
  yield
  await close_all()


@pytest.fixture
async def repo() -> Repo[Row]:
  conn = await connect("repo_tests", db_path=":memory:")
  await conn.execute("CREATE TABLE rows (id TEXT PRIMARY KEY, name TEXT, payload TEXT, count INTEGER)")
  await conn.commit()
  return Repo(conn, table="rows", model=Row)


async def test_insert_and_get(repo: Repo[Row]) -> None:
  row = Row(id="r1", name="alpha", payload={"k": "v"}, count=3)
  await repo.insert(row)
  back = await repo.get("r1")
  assert back is not None
  assert back.id == "r1"
  assert back.name == "alpha"
  assert back.payload == {"k": "v"}
  assert back.count == 3


async def test_get_missing_returns_none(repo: Repo[Row]) -> None:
  assert await repo.get("nope") is None


async def test_update(repo: Repo[Row]) -> None:
  await repo.insert(Row(id="r2", name="x", payload={}, count=0))
  changed = await repo.update("r2", name="y", count=10)
  assert changed == 1
  back = await repo.get("r2")
  assert back is not None
  assert back.name == "y" and back.count == 10


async def test_delete(repo: Repo[Row]) -> None:
  await repo.insert(Row(id="r3", name="x", payload={}, count=0))
  deleted = await repo.delete("r3")
  assert deleted == 1
  assert await repo.get("r3") is None


async def test_select_filter_and_order(repo: Repo[Row]) -> None:
  await repo.insert(Row(id="a", name="a", payload={}, count=1))
  await repo.insert(Row(id="b", name="b", payload={}, count=5))
  await repo.insert(Row(id="c", name="c", payload={}, count=3))
  rows = await repo.select(where="count > ?", params=(1,), order_by="count DESC")
  assert [r.id for r in rows] == ["b", "c"]


async def test_select_limit_offset(repo: Repo[Row]) -> None:
  for i in range(5):
    await repo.insert(Row(id=f"r{i}", name=str(i), payload={}, count=i))
  page = await repo.select(order_by="count ASC", limit=2, offset=2)
  assert [r.id for r in page] == ["r2", "r3"]


async def test_count(repo: Repo[Row]) -> None:
  for i in range(4):
    await repo.insert(Row(id=f"x{i}", name="x", payload={}, count=i))
  assert await repo.count() == 4
  assert await repo.count(where="count >= ?", params=(2,)) == 2


async def test_parameter_binding_safe_against_injection(repo: Repo[Row]) -> None:
  await repo.insert(Row(id="real", name="x", payload={}, count=0))
  # Quoting / injection inside the pk value must not be evaluated as SQL.
  result = await repo.get("real' OR '1'='1")
  assert result is None


def test_non_dataclass_raises() -> None:
  class NotADataclass:
    id: str = ""

  conn_stub: Any = object()
  with pytest.raises(TypeError):
    Repo(conn_stub, table="x", model=NotADataclass)
