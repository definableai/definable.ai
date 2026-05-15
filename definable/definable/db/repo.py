"""Typed CRUD over a frozen dataclass bound to a SQLite table.

Tiny and opinionated. Designed for record-keeping (events, runs, logs),
not relational graphs. Anything fancier — joins, aggregates, full-text
search — goes through ``conn.execute(...)`` directly.

Example::

    @dataclass(frozen=True, kw_only=True)
    class Run:
      id: str
      agent_id: str
      payload: dict[str, Any]
      started_at: float

    repo = Repo(conn, table="runs", model=Run)
    await repo.insert(Run(id="r1", agent_id="a", payload={"q": 1}, started_at=t))
    row = await repo.get("r1")
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Generic, Type, TypeVar

import aiosqlite

from definable.db.types import decode_value, encode_value

T = TypeVar("T")


class Repo(Generic[T]):
  """Typed CRUD over a frozen dataclass.

  All queries use parameter binding — never string-format user input into
  SQL. ``pk_col`` defaults to ``id``; override on construction if the
  primary key column is something else.
  """

  def __init__(self, conn: aiosqlite.Connection, *, table: str, model: Type[T], pk_col: str = "id") -> None:
    if not dataclasses.is_dataclass(model):
      raise TypeError(f"Repo model must be a dataclass, got {model!r}")
    self._conn = conn
    self._table = table
    self._model = model
    self._pk_col = pk_col
    self._fields: list[str] = [f.name for f in dataclasses.fields(model)]  # type: ignore[arg-type]
    self._field_types: dict[str, Any] = typing.get_type_hints(model)

  async def insert(self, row: T) -> None:
    cols = ", ".join(self._fields)
    placeholders = ", ".join("?" * len(self._fields))
    values = tuple(encode_value(getattr(row, f)) for f in self._fields)
    await self._conn.execute(f"INSERT INTO {self._table} ({cols}) VALUES ({placeholders})", values)
    await self._conn.commit()

  async def get(self, pk: Any) -> T | None:
    async with self._conn.execute(f"SELECT * FROM {self._table} WHERE {self._pk_col} = ?", (pk,)) as cur:
      row = await cur.fetchone()
      cols = [c[0] for c in cur.description] if cur.description else []
    return self._row_to_model(row, cols) if row else None

  async def update(self, pk: Any, **fields: Any) -> int:
    if not fields:
      return 0
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = tuple(encode_value(v) for v in fields.values()) + (pk,)
    cur = await self._conn.execute(f"UPDATE {self._table} SET {sets} WHERE {self._pk_col} = ?", values)
    await self._conn.commit()
    return cur.rowcount or 0

  async def delete(self, pk: Any) -> int:
    cur = await self._conn.execute(f"DELETE FROM {self._table} WHERE {self._pk_col} = ?", (pk,))
    await self._conn.commit()
    return cur.rowcount or 0

  async def select(
    self,
    *,
    where: str | None = None,
    params: tuple[Any, ...] = (),
    order_by: str | None = None,
    limit: int | None = None,
    offset: int = 0,
  ) -> list[T]:
    """Read rows. ``where`` and ``order_by`` are caller-provided SQL fragments.

    Caller is responsible for passing ``params`` matching ``?`` placeholders
    in ``where``. Never interpolate user values into ``where`` directly.
    """
    sql = f"SELECT * FROM {self._table}"
    if where:
      sql += f" WHERE {where}"
    if order_by:
      sql += f" ORDER BY {order_by}"
    if limit is not None:
      sql += " LIMIT ?"
      params = params + (int(limit),)
      if offset:
        sql += " OFFSET ?"
        params = params + (int(offset),)
    async with self._conn.execute(sql, params) as cur:
      rows = await cur.fetchall()
      cols = [c[0] for c in cur.description] if cur.description else []
    return [self._row_to_model(r, cols) for r in rows]

  async def count(self, *, where: str | None = None, params: tuple[Any, ...] = ()) -> int:
    sql = f"SELECT COUNT(*) FROM {self._table}"
    if where:
      sql += f" WHERE {where}"
    async with self._conn.execute(sql, params) as cur:
      row = await cur.fetchone()
    return int(row[0]) if row else 0

  def _row_to_model(self, row: Any, cols: list[str]) -> T:
    raw = dict(zip(cols, row, strict=False))
    kwargs = {f: decode_value(raw.get(f), self._field_types.get(f)) for f in self._fields}
    return self._model(**kwargs)  # type: ignore[call-arg]
