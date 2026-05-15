"""Per-namespace async SQLite connection manager.

Single connection per namespace per process. aiosqlite serializes
concurrent access on its background worker thread, so sharing one
connection across many tasks is the recommended SQLite path under WAL.

Pragmas applied on first connect:
  - ``journal_mode=WAL``        — readers don't block the writer
  - ``synchronous=NORMAL``       — safe under WAL, ~2× write throughput
  - ``foreign_keys=ON``          — referential integrity is opt-in in SQLite
  - ``busy_timeout=5000``        — 5 s lock wait before SQLITE_BUSY
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import aiosqlite

from definable.utils.workspace import workspace_path

# Module-level cache: one connection per namespace per process.
_CONNS: dict[str, aiosqlite.Connection] = {}
_LOCK = asyncio.Lock()

Connection = aiosqlite.Connection


async def connect(namespace: str, *, db_path: Path | str | None = None) -> aiosqlite.Connection:
  """Open (or reuse) the per-namespace connection.

  ``namespace`` is a stable short name (e.g. ``"observability"``). The DB
  file lives at ``.definable/<namespace>.db`` unless ``db_path`` overrides
  it explicitly (useful for tests with ``":memory:"``).
  """
  async with _LOCK:
    existing = _CONNS.get(namespace)
    if existing is not None:
      return existing

    if db_path is None:
      path: str = str(workspace_path(f"{namespace}.db"))
    elif db_path == ":memory:":
      path = ":memory:"
    else:
      path = str(db_path)

    conn = await aiosqlite.connect(path)
    await _apply_pragmas(conn, in_memory=(path == ":memory:"))
    _CONNS[namespace] = conn
    return conn


async def _apply_pragmas(conn: aiosqlite.Connection, *, in_memory: bool) -> None:
  # WAL is not meaningful for ``:memory:`` databases — skip to avoid noise.
  if not in_memory:
    await conn.execute("PRAGMA journal_mode=WAL")
  await conn.execute("PRAGMA synchronous=NORMAL")
  await conn.execute("PRAGMA foreign_keys=ON")
  await conn.execute("PRAGMA busy_timeout=5000")
  await conn.commit()


async def close(namespace: str) -> None:
  """Close one namespace's connection if open."""
  async with _LOCK:
    conn = _CONNS.pop(namespace, None)
  if conn is not None:
    await conn.close()


async def close_all() -> None:
  """Close every cached connection. Safe to call at process shutdown."""
  async with _LOCK:
    items = list(_CONNS.items())
    _CONNS.clear()
  for _, conn in items:
    with contextlib.suppress(Exception):
      await conn.close()
