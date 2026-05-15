"""Numbered SQL migration runner, one namespace at a time.

A migration file is named ``NNNN_<slug>.sql`` and lives under
``definable/db/migrations/<namespace>/``. The runner records each
applied version in a per-database ``schema_migrations`` table so reruns
are no-ops.

A single ``.sql`` file may contain multiple statements separated by
``;`` — they are executed as a single transaction; any failure rolls the
file back so half-applied migrations cannot wedge a DB.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

_MIGRATIONS_ROOT = Path(__file__).parent / "migrations"
_FILENAME_RE = re.compile(r"^(\d+)_([A-Za-z0-9_\-]+)\.sql$")


def _migrations_dir(namespace: str, *, root: Path | None = None) -> Path:
  return (root or _MIGRATIONS_ROOT) / namespace


def _discover(namespace: str, *, root: Path | None = None) -> list[tuple[int, str, Path]]:
  """Return ``(version, name, path)`` triples sorted by version."""
  d = _migrations_dir(namespace, root=root)
  if not d.exists():
    return []
  items: list[tuple[int, str, Path]] = []
  for p in d.iterdir():
    if not p.is_file():
      continue
    m = _FILENAME_RE.match(p.name)
    if not m:
      continue
    items.append((int(m.group(1)), m.group(2), p))
  items.sort(key=lambda t: t[0])
  return items


async def _ensure_table(conn: aiosqlite.Connection) -> None:
  await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (  version INTEGER PRIMARY KEY,  name TEXT NOT NULL,  applied_at REAL NOT NULL)")
  await conn.commit()


async def _applied_versions(conn: aiosqlite.Connection) -> set[int]:
  async with conn.execute("SELECT version FROM schema_migrations") as cur:
    rows = await cur.fetchall()
  return {int(r[0]) for r in rows}


async def migrate(namespace: str, conn: aiosqlite.Connection, *, root: Path | None = None) -> int:
  """Apply pending migrations for ``namespace``. Returns count applied."""
  await _ensure_table(conn)
  applied = await _applied_versions(conn)
  pending = [t for t in _discover(namespace, root=root) if t[0] not in applied]
  if not pending:
    return 0

  for version, name, path in pending:
    sql = path.read_text(encoding="utf-8")
    try:
      await conn.execute("BEGIN")
      for stmt in _split_statements(sql):
        await conn.execute(stmt)
      await conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (version, name, time.time()),
      )
      await conn.commit()
    except Exception:
      await conn.rollback()
      log.exception("Migration %s/%04d_%s failed; rolled back", namespace, version, name)
      raise
    log.info("Applied migration %s/%04d_%s", namespace, version, name)
  return len(pending)


def _split_statements(sql: str) -> list[str]:
  """Split a multi-statement SQL blob on top-level ``;`` boundaries.

  Tracks single-quoted strings so embedded semicolons inside string
  literals don't split. Migration files should avoid inline ``--``
  comments after statements; trailing-only comments are fine.
  """
  out: list[str] = []
  buf: list[str] = []
  in_string = False
  for ch in sql:
    if ch == "'":
      in_string = not in_string
    if ch == ";" and not in_string:
      stmt = "".join(buf).strip()
      if stmt:
        out.append(stmt)
      buf = []
      continue
    buf.append(ch)
  tail = "".join(buf).strip()
  if tail:
    out.append(tail)
  return out
