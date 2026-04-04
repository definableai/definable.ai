"""SQLite-backed memory store with FTS5 search, access tracking, and admin ops."""

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import aiosqlite

from definable.memory.v2.models import (
  IndexEntry,
  MemoryEntry,
  MemoryStats,
  WarmMemory,
  WorkingMemory,
  WorkingMemorySnapshot,
)
from definable.memory.v2.stores.base import MemoryStore

_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS working_memory (
  user_id TEXT PRIMARY KEY,
  content TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS warm_memory (
  user_id TEXT PRIMARY KEY,
  content TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS working_memory_history (
  user_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  content TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  session_id TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (user_id, version)
);

CREATE TABLE IF NOT EXISTS memory_index (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'conversation',
  tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  session_id TEXT NOT NULL DEFAULT '',
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  source TEXT NOT NULL DEFAULT 'user_stated'
);

CREATE INDEX IF NOT EXISTS idx_memory_index_user ON memory_index(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_index_category ON memory_index(user_id, category);
CREATE INDEX IF NOT EXISTS idx_memory_index_access ON memory_index(user_id, access_count);
CREATE INDEX IF NOT EXISTS idx_memory_index_created ON memory_index(user_id, created_at);

CREATE TABLE IF NOT EXISTS memory_entries (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'conversation',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_turn INTEGER,
  expires_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_entries_user ON memory_entries(user_id);

CREATE TABLE IF NOT EXISTS _schema_version (
  version INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  id UNINDEXED,
  user_id UNINDEXED,
  summary,
  tags,
  content_text
);
"""

_WM_HISTORY_LIMIT = 20  # Keep last N versions per user

# Strip FTS5 special characters to prevent syntax errors
_FTS_SPECIAL = re.compile(r'["\*\(\)\{\}\[\]:^~\-]')


def _sanitize_fts_word(word: str) -> str:
  """Remove FTS5 special characters from a search word."""
  return _FTS_SPECIAL.sub("", word).strip()


def _recency_weight(created_at_iso: str, half_life_days: float = 30.0) -> float:
  """Calculate time-decay weight. 1.0 for today, 0.5 at half_life_days ago."""
  try:
    created = datetime.fromisoformat(created_at_iso)
    now = datetime.now(timezone.utc)
    days_ago = max(0.0, (now - created).total_seconds() / 86400.0)
    return math.pow(0.5, days_ago / half_life_days)
  except (ValueError, TypeError):
    return 0.5


class SQLiteStore(MemoryStore):
  """SQLite memory store with FTS5 search, recency ranking, and enterprise features."""

  def __init__(self, db_path: str = "./memory.db", *, half_life_days: float = 30.0) -> None:
    self._db_path = db_path
    self._half_life_days = half_life_days
    self._db: Optional[aiosqlite.Connection] = None
    self._initialized = False

  async def _ensure_initialized(self) -> aiosqlite.Connection:
    if self._db is None:
      self._db = await aiosqlite.connect(self._db_path)
      self._db.row_factory = sqlite3.Row
    if not self._initialized:
      await self._db.executescript(_SCHEMA)
      await self._db.commit()
      await self._migrate_if_needed(self._db)
      self._initialized = True
    return self._db

  async def _migrate_if_needed(self, db: aiosqlite.Connection) -> None:
    """Check schema version and run migrations if needed."""
    cursor = await db.execute("SELECT version FROM _schema_version LIMIT 1")
    row = await cursor.fetchone()
    current_version = row["version"] if row else 0

    if current_version < _SCHEMA_VERSION:
      # Migration: rebuild FTS, add new columns if missing
      await db.execute("DROP TABLE IF EXISTS memory_fts")
      await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, user_id UNINDEXED, summary, tags, content_text)")
      # Add columns that may not exist (ALTER TABLE ADD COLUMN is idempotent-safe)
      import contextlib

      for col, typedef in [
        ("access_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_accessed_at", "TEXT"),
        ("confidence", "REAL NOT NULL DEFAULT 1.0"),
        ("source", "TEXT NOT NULL DEFAULT 'user_stated'"),
      ]:
        with contextlib.suppress(Exception):
          await db.execute(f"ALTER TABLE memory_index ADD COLUMN {col} {typedef}")
      for col, typedef in [("expires_at", "TEXT")]:
        with contextlib.suppress(Exception):
          await db.execute(f"ALTER TABLE memory_entries ADD COLUMN {col} {typedef}")
      # Create indexes that may not exist
      for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_memory_index_access ON memory_index(user_id, access_count)",
        "CREATE INDEX IF NOT EXISTS idx_memory_index_created ON memory_index(user_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entries_user ON memory_entries(user_id)",
      ]:
        await db.execute(idx_sql)
      # Re-populate FTS from existing data
      cursor = await db.execute(
        "SELECT mi.id, mi.user_id, mi.summary, mi.tags, COALESCE(me.content, '') as content "
        "FROM memory_index mi LEFT JOIN memory_entries me ON mi.id = me.id"
      )
      rows = await cursor.fetchall()
      for r in rows:
        await db.execute(
          "INSERT INTO memory_fts (id, user_id, summary, tags, content_text) VALUES (?, ?, ?, ?, ?)",
          (r["id"], r["user_id"], r["summary"], r["tags"], r["content"]),
        )
      # Upsert schema version
      if current_version == 0:
        await db.execute("INSERT INTO _schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
      else:
        await db.execute("UPDATE _schema_version SET version = ?", (_SCHEMA_VERSION,))
      await db.commit()

  # --- Core CRUD ---

  async def get_working_memory(self, user_id: str) -> Optional[WorkingMemory]:
    db = await self._ensure_initialized()
    cursor = await db.execute("SELECT content, updated_at, version FROM working_memory WHERE user_id = ?", (user_id,))
    row = await cursor.fetchone()
    if row is None:
      return None
    return WorkingMemory(
      user_id=user_id,
      content=row["content"],
      updated_at=datetime.fromisoformat(row["updated_at"]),
      version=row["version"],
    )

  async def set_working_memory(self, user_id: str, content: str, *, session_id: str = "") -> WorkingMemory:
    db = await self._ensure_initialized()
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
      """INSERT INTO working_memory (user_id, content, updated_at, version)
         VALUES (?, ?, ?, 1)
         ON CONFLICT(user_id) DO UPDATE SET
           content = excluded.content,
           updated_at = excluded.updated_at,
           version = version + 1""",
      (user_id, content, now),
    )
    cursor = await db.execute("SELECT version FROM working_memory WHERE user_id = ?", (user_id,))
    row = await cursor.fetchone()
    version = row["version"] if row else 1

    # Store snapshot in history (keep last N)
    await db.execute(
      "INSERT OR REPLACE INTO working_memory_history (user_id, version, content, updated_at, session_id) VALUES (?, ?, ?, ?, ?)",
      (user_id, version, content, now, session_id),
    )
    # Prune old history
    await db.execute(
      "DELETE FROM working_memory_history WHERE user_id = ? AND version <= ?",
      (user_id, version - _WM_HISTORY_LIMIT),
    )

    await db.commit()
    return WorkingMemory(user_id=user_id, content=content, updated_at=datetime.fromisoformat(now), version=version)

  async def add_entry(
    self,
    user_id: str,
    summary: str,
    content: str,
    category: str,
    tags: List[str],
    session_id: str,
    *,
    confidence: float = 1.0,
    source: str = "user_stated",
    expires_at: Optional[datetime] = None,
  ) -> IndexEntry:
    db = await self._ensure_initialized()
    entry_id = uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(tags)
    expires_str = expires_at.isoformat() if expires_at else None

    await db.execute(
      "INSERT INTO memory_index (id, user_id, summary, category, tags, created_at, session_id, confidence, source) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      (entry_id, user_id, summary, category, tags_json, now, session_id, confidence, source),
    )
    await db.execute(
      "INSERT INTO memory_fts (id, user_id, summary, tags, content_text) VALUES (?, ?, ?, ?, ?)",
      (entry_id, user_id, summary, tags_json, content),
    )
    await db.execute(
      "INSERT INTO memory_entries (id, user_id, content, category, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (entry_id, user_id, content, category, now, now, expires_str),
    )
    await db.commit()

    return IndexEntry(
      id=entry_id,
      user_id=user_id,
      summary=summary,
      category=category,
      tags=tags,
      created_at=datetime.fromisoformat(now),
      session_id=session_id,
      confidence=confidence,
      source=source,
    )

  async def search_index(
    self,
    user_id: str,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
    *,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
  ) -> List[IndexEntry]:
    db = await self._ensure_initialized()

    # Auto-prune expired entries on search
    await self._prune_expired_internal(db, user_id)

    if query:
      words = [_sanitize_fts_word(w) for w in query.split() if w.strip()]
      words = [w for w in words if w]
      if not words:
        return await self._search_like(db, user_id, query, category, limit, after, before)
      fts_query = " OR ".join(f"{w}*" for w in words)
      sql = """
        SELECT mi.id, mi.user_id, mi.summary, mi.category, mi.tags, mi.created_at,
               mi.session_id, mi.access_count, mi.last_accessed_at, mi.confidence, mi.source
        FROM memory_index mi
        JOIN memory_fts ON mi.id = memory_fts.id
        WHERE memory_fts MATCH ? AND mi.user_id = ?
      """
      params: list = [fts_query, user_id]
      if category:
        sql += " AND mi.category = ?"
        params.append(category)
      if after:
        sql += " AND mi.created_at >= ?"
        params.append(after.isoformat())
      if before:
        sql += " AND mi.created_at <= ?"
        params.append(before.isoformat())
      # Fetch more than limit, then rerank with recency weighting
      sql += " ORDER BY rank LIMIT ?"
      params.append(limit * 3)

      try:
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        if not rows:
          return await self._search_like(db, user_id, query, category, limit, after, before)
      except Exception:
        return await self._search_like(db, user_id, query, category, limit, after, before)
    else:
      sql = (
        "SELECT id, user_id, summary, category, tags, created_at, session_id, "
        "access_count, last_accessed_at, confidence, source "
        "FROM memory_index WHERE user_id = ?"
      )
      params = [user_id]
      if category:
        sql += " AND category = ?"
        params.append(category)
      if after:
        sql += " AND created_at >= ?"
        params.append(after.isoformat())
      if before:
        sql += " AND created_at <= ?"
        params.append(before.isoformat())
      sql += " ORDER BY created_at DESC LIMIT ?"
      params.append(limit)
      cursor = await db.execute(sql, params)
      rows = await cursor.fetchall()

    entries = [self._row_to_index_entry(row) for row in rows]

    # Re-rank with recency + access weighting if this was a query search
    if query and len(entries) > limit:
      for e in entries:
        recency = _recency_weight(e.created_at.isoformat(), self._half_life_days)
        access_boost = 1.0 + math.log1p(e.access_count)
        e._score = recency * access_boost * e.confidence  # type: ignore[attr-defined]
      entries.sort(key=lambda e: getattr(e, "_score", 0), reverse=True)
      entries = entries[:limit]

    return entries

  async def _search_like(
    self,
    db: aiosqlite.Connection,
    user_id: str,
    query: str,
    category: Optional[str],
    limit: int,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
  ) -> List[IndexEntry]:
    """Fallback search using case-insensitive LIKE on summary, tags, and content."""
    like_pat = f"%{query}%"

    sql = (
      "SELECT id, user_id, summary, category, tags, created_at, session_id, "
      "access_count, last_accessed_at, confidence, source "
      "FROM memory_index WHERE user_id = ? AND (LOWER(summary) LIKE LOWER(?) OR LOWER(tags) LIKE LOWER(?))"
    )
    params: list = [user_id, like_pat, like_pat]
    if category:
      sql += " AND category = ?"
      params.append(category)
    if after:
      sql += " AND created_at >= ?"
      params.append(after.isoformat())
    if before:
      sql += " AND created_at <= ?"
      params.append(before.isoformat())
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()

    if not rows:
      sql_content = (
        "SELECT mi.id, mi.user_id, mi.summary, mi.category, mi.tags, mi.created_at, mi.session_id, "
        "mi.access_count, mi.last_accessed_at, mi.confidence, mi.source "
        "FROM memory_entries me JOIN memory_index mi ON me.id = mi.id "
        "WHERE mi.user_id = ? AND LOWER(me.content) LIKE LOWER(?)"
      )
      params_c: list = [user_id, like_pat]
      if category:
        sql_content += " AND mi.category = ?"
        params_c.append(category)
      if after:
        sql_content += " AND mi.created_at >= ?"
        params_c.append(after.isoformat())
      if before:
        sql_content += " AND mi.created_at <= ?"
        params_c.append(before.isoformat())
      sql_content += " ORDER BY mi.created_at DESC LIMIT ?"
      params_c.append(limit)
      cursor = await db.execute(sql_content, params_c)
      rows = await cursor.fetchall()

    return [self._row_to_index_entry(row) for row in rows]

  async def get_entries(self, entry_ids: List[str]) -> List[MemoryEntry]:
    db = await self._ensure_initialized()
    if not entry_ids:
      return []
    placeholders = ",".join("?" for _ in entry_ids)
    cursor = await db.execute(
      f"SELECT id, user_id, content, category, created_at, updated_at, source_turn, expires_at FROM memory_entries WHERE id IN ({placeholders})",
      entry_ids,
    )
    rows = await cursor.fetchall()

    # Bump access tracking
    now = datetime.now(timezone.utc).isoformat()
    for eid in entry_ids:
      await db.execute(
        "UPDATE memory_index SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
        (now, eid),
      )
    await db.commit()

    return [
      MemoryEntry(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        category=row["category"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        source_turn=row["source_turn"],
        expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
      )
      for row in rows
    ]

  async def delete_entry(self, entry_id: str) -> bool:
    db = await self._ensure_initialized()
    await db.execute("DELETE FROM memory_fts WHERE id = ?", (entry_id,))
    cursor = await db.execute("DELETE FROM memory_index WHERE id = ?", (entry_id,))
    await db.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
    await db.commit()
    return cursor.rowcount > 0

  # --- Warm memory ---

  async def get_warm_memory(self, user_id: str) -> Optional[WarmMemory]:
    db = await self._ensure_initialized()
    cursor = await db.execute("SELECT content, updated_at FROM warm_memory WHERE user_id = ?", (user_id,))
    row = await cursor.fetchone()
    if row is None:
      return None
    return WarmMemory(user_id=user_id, content=row["content"], updated_at=datetime.fromisoformat(row["updated_at"]))

  async def set_warm_memory(self, user_id: str, content: str) -> WarmMemory:
    db = await self._ensure_initialized()
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
      "INSERT INTO warm_memory (user_id, content, updated_at) VALUES (?, ?, ?) "
      "ON CONFLICT(user_id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
      (user_id, content, now),
    )
    await db.commit()
    return WarmMemory(user_id=user_id, content=content, updated_at=datetime.fromisoformat(now))

  # --- WM history ---

  async def get_wm_history(self, user_id: str, limit: int = 20) -> List[WorkingMemorySnapshot]:
    db = await self._ensure_initialized()
    cursor = await db.execute(
      "SELECT version, content, updated_at, session_id FROM working_memory_history WHERE user_id = ? ORDER BY version DESC LIMIT ?",
      (user_id, limit),
    )
    rows = await cursor.fetchall()
    return [
      WorkingMemorySnapshot(
        user_id=user_id,
        version=row["version"],
        content=row["content"],
        updated_at=datetime.fromisoformat(row["updated_at"]),
        session_id=row["session_id"],
      )
      for row in rows
    ]

  async def rollback_working_memory(self, user_id: str, version: int) -> Optional[WorkingMemory]:
    db = await self._ensure_initialized()
    cursor = await db.execute(
      "SELECT content FROM working_memory_history WHERE user_id = ? AND version = ?",
      (user_id, version),
    )
    row = await cursor.fetchone()
    if row is None:
      return None
    return await self.set_working_memory(user_id, row["content"])

  # --- Admin / GDPR ---

  async def delete_user(self, user_id: str) -> int:
    db = await self._ensure_initialized()
    cursor = await db.execute("SELECT id FROM memory_index WHERE user_id = ?", (user_id,))
    ids = [row["id"] for row in await cursor.fetchall()]
    for eid in ids:
      await db.execute("DELETE FROM memory_fts WHERE id = ?", (eid,))
    await db.execute("DELETE FROM memory_index WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM memory_entries WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM working_memory WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM warm_memory WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM working_memory_history WHERE user_id = ?", (user_id,))
    await db.commit()
    return len(ids)

  async def export_user(self, user_id: str) -> dict:
    await self._ensure_initialized()
    wm = await self.get_working_memory(user_id)
    warm = await self.get_warm_memory(user_id)
    entries = await self.search_index(user_id, limit=10000)
    full = await self.get_entries([e.id for e in entries]) if entries else []
    return {
      "user_id": user_id,
      "working_memory": {"content": wm.content, "version": wm.version, "updated_at": wm.updated_at.isoformat()} if wm else None,
      "warm_memory": {"content": warm.content, "updated_at": warm.updated_at.isoformat()} if warm else None,
      "entries": [
        {
          "id": e.id,
          "content": e.content,
          "category": e.category,
          "created_at": e.created_at.isoformat(),
          "updated_at": e.updated_at.isoformat(),
          "expires_at": e.expires_at.isoformat() if e.expires_at else None,
        }
        for e in full
      ],
      "index": [
        {
          "id": e.id,
          "summary": e.summary,
          "category": e.category,
          "tags": e.tags,
          "confidence": e.confidence,
          "source": e.source,
          "access_count": e.access_count,
          "created_at": e.created_at.isoformat(),
        }
        for e in entries
      ],
    }

  async def count_entries(self, user_id: str) -> int:
    db = await self._ensure_initialized()
    cursor = await db.execute("SELECT COUNT(*) as cnt FROM memory_index WHERE user_id = ?", (user_id,))
    row = await cursor.fetchone()
    return row["cnt"] if row else 0

  async def get_stats(self, user_id: str) -> MemoryStats:
    db = await self._ensure_initialized()
    wm = await self.get_working_memory(user_id)
    warm = await self.get_warm_memory(user_id)

    cursor = await db.execute("SELECT COUNT(*) as cnt FROM memory_index WHERE user_id = ?", (user_id,))
    count_row = await cursor.fetchone()

    cursor = await db.execute(
      "SELECT SUM(LENGTH(content)) as total, MIN(created_at) as oldest, MAX(created_at) as newest FROM memory_entries WHERE user_id = ?",
      (user_id,),
    )
    agg = await cursor.fetchone()

    cursor = await db.execute("SELECT category, COUNT(*) as cnt FROM memory_index WHERE user_id = ? GROUP BY category", (user_id,))
    cats = {row["category"]: row["cnt"] for row in await cursor.fetchall()}

    return MemoryStats(
      user_id=user_id,
      wm_chars=len(wm.content) if wm else 0,
      wm_version=wm.version if wm else 0,
      warm_chars=len(warm.content) if warm else 0,
      entry_count=count_row["cnt"] if count_row else 0,
      total_content_chars=agg["total"] or 0 if agg else 0,
      oldest_entry=datetime.fromisoformat(agg["oldest"]) if agg and agg["oldest"] else None,
      newest_entry=datetime.fromisoformat(agg["newest"]) if agg and agg["newest"] else None,
      categories=cats,
    )

  # --- Maintenance ---

  async def prune_expired(self, user_id: str) -> int:
    db = await self._ensure_initialized()
    return await self._prune_expired_internal(db, user_id)

  async def _prune_expired_internal(self, db: aiosqlite.Connection, user_id: str) -> int:
    """Remove entries past their TTL."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = await db.execute(
      "SELECT id FROM memory_entries WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
      (user_id, now),
    )
    expired_ids = [row["id"] for row in await cursor.fetchall()]
    for eid in expired_ids:
      await db.execute("DELETE FROM memory_fts WHERE id = ?", (eid,))
      await db.execute("DELETE FROM memory_index WHERE id = ?", (eid,))
      await db.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
    if expired_ids:
      await db.commit()
    return len(expired_ids)

  async def enforce_limit(self, user_id: str, max_entries: int) -> int:
    db = await self._ensure_initialized()
    count = await self.count_entries(user_id)
    if count <= max_entries:
      return 0
    excess = count - max_entries
    # Delete lowest-priority: lowest access_count, oldest
    cursor = await db.execute(
      "SELECT id FROM memory_index WHERE user_id = ? ORDER BY access_count ASC, created_at ASC LIMIT ?",
      (user_id, excess),
    )
    to_delete = [row["id"] for row in await cursor.fetchall()]
    for eid in to_delete:
      await db.execute("DELETE FROM memory_fts WHERE id = ?", (eid,))
      await db.execute("DELETE FROM memory_index WHERE id = ?", (eid,))
      await db.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
    if to_delete:
      await db.commit()
    return len(to_delete)

  # --- Helpers ---

  def _row_to_index_entry(self, row: sqlite3.Row) -> IndexEntry:
    return IndexEntry(
      id=row["id"],
      user_id=row["user_id"],
      summary=row["summary"],
      category=row["category"],
      tags=json.loads(row["tags"]) if row["tags"] else [],
      created_at=datetime.fromisoformat(row["created_at"]),
      session_id=row["session_id"],
      access_count=row["access_count"] if "access_count" in row.keys() else 0,
      last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
      confidence=row["confidence"] if "confidence" in row.keys() else 1.0,
      source=row["source"] if "source" in row.keys() else "user_stated",
    )

  async def close(self) -> None:
    if self._db is not None:
      await self._db.close()
      self._db = None
      self._initialized = False
