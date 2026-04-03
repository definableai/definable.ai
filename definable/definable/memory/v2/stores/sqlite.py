"""SQLite-backed memory store with FTS5 search."""

import re
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import aiosqlite

from definable.memory.v2.models import IndexEntry, MemoryEntry, WorkingMemory
from definable.memory.v2.stores.base import MemoryStore

_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS working_memory (
  user_id TEXT PRIMARY KEY,
  content TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_index (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'conversation',
  tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  session_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memory_index_user ON memory_index(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_index_category ON memory_index(user_id, category);

CREATE TABLE IF NOT EXISTS memory_entries (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'conversation',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_turn INTEGER
);

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

# Strip FTS5 special characters to prevent syntax errors
_FTS_SPECIAL = re.compile(r'["\*\(\)\{\}\[\]:^~\-]')


def _sanitize_fts_word(word: str) -> str:
  """Remove FTS5 special characters from a search word."""
  return _FTS_SPECIAL.sub("", word).strip()


class SQLiteStore(MemoryStore):
  """SQLite memory store with FTS5 full-text search on summaries, tags, and content."""

  def __init__(self, db_path: str = "./memory.db") -> None:
    self._db_path = db_path
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
      # Migration to v2: rebuild FTS with content_text column
      await db.execute("DROP TABLE IF EXISTS memory_fts")
      await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, user_id UNINDEXED, summary, tags, content_text)")
      # Re-populate FTS from existing data
      cursor = await db.execute(
        "SELECT mi.id, mi.user_id, mi.summary, mi.tags, COALESCE(me.content, '') as content "
        "FROM memory_index mi LEFT JOIN memory_entries me ON mi.id = me.id"
      )
      rows = await cursor.fetchall()
      for row in rows:
        await db.execute(
          "INSERT INTO memory_fts (id, user_id, summary, tags, content_text) VALUES (?, ?, ?, ?, ?)",
          (row["id"], row["user_id"], row["summary"], row["tags"], row["content"]),
        )
      # Upsert schema version
      if current_version == 0:
        await db.execute("INSERT INTO _schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
      else:
        await db.execute("UPDATE _schema_version SET version = ?", (_SCHEMA_VERSION,))
      await db.commit()

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

  async def set_working_memory(self, user_id: str, content: str) -> WorkingMemory:
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
    await db.commit()
    cursor = await db.execute("SELECT version FROM working_memory WHERE user_id = ?", (user_id,))
    row = await cursor.fetchone()
    return WorkingMemory(
      user_id=user_id,
      content=content,
      updated_at=datetime.fromisoformat(now),
      version=row["version"] if row else 1,
    )

  async def add_entry(
    self,
    user_id: str,
    summary: str,
    content: str,
    category: str,
    tags: List[str],
    session_id: str,
  ) -> IndexEntry:
    db = await self._ensure_initialized()
    import json

    entry_id = uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    tags_json = json.dumps(tags)

    # Insert into index
    await db.execute(
      "INSERT INTO memory_index (id, user_id, summary, category, tags, created_at, session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (entry_id, user_id, summary, category, tags_json, now, session_id),
    )

    # Insert into FTS index (summary + tags + content for full-text search)
    await db.execute(
      "INSERT INTO memory_fts (id, user_id, summary, tags, content_text) VALUES (?, ?, ?, ?, ?)",
      (entry_id, user_id, summary, tags_json, content),
    )

    # Insert into entries
    await db.execute(
      "INSERT INTO memory_entries (id, user_id, content, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
      (entry_id, user_id, content, category, now, now),
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
    )

  async def search_index(
    self,
    user_id: str,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
  ) -> List[IndexEntry]:
    db = await self._ensure_initialized()
    import json

    if query:
      # FTS5 search scoped to user — tokenize, sanitize, prefix-match
      words = [_sanitize_fts_word(w) for w in query.split() if w.strip()]
      words = [w for w in words if w]
      if not words:
        return await self._search_like(db, user_id, query, category, limit)
      # Each word gets a * suffix for prefix matching (e.g., "Pyth" matches "Python")
      # Join with OR for broad recall (any word matches)
      fts_query = " OR ".join(f"{w}*" for w in words)
      sql = """
        SELECT mi.id, mi.user_id, mi.summary, mi.category, mi.tags, mi.created_at, mi.session_id
        FROM memory_index mi
        JOIN memory_fts ON mi.id = memory_fts.id
        WHERE memory_fts MATCH ? AND mi.user_id = ?
      """
      params: list = [fts_query, user_id]
      if category:
        sql += " AND mi.category = ?"
        params.append(category)
      sql += " ORDER BY rank LIMIT ?"
      params.append(limit)

      try:
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        # If FTS returned nothing, try LIKE fallback
        if not rows:
          return await self._search_like(db, user_id, query, category, limit)
      except Exception:
        # FTS match failed (bad syntax) — fall back to LIKE search
        return await self._search_like(db, user_id, query, category, limit)
    else:
      sql = "SELECT id, user_id, summary, category, tags, created_at, session_id FROM memory_index WHERE user_id = ?"
      params = [user_id]
      if category:
        sql += " AND category = ?"
        params.append(category)
      sql += " ORDER BY created_at DESC LIMIT ?"
      params.append(limit)
      cursor = await db.execute(sql, params)
      rows = await cursor.fetchall()
    return [
      IndexEntry(
        id=row["id"],
        user_id=row["user_id"],
        summary=row["summary"],
        category=row["category"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        created_at=datetime.fromisoformat(row["created_at"]),
        session_id=row["session_id"],
      )
      for row in rows
    ]

  async def _search_like(
    self,
    db: aiosqlite.Connection,
    user_id: str,
    query: str,
    category: Optional[str],
    limit: int,
  ) -> List[IndexEntry]:
    """Fallback search using case-insensitive LIKE on summary, tags, and content."""
    import json

    like_pat = f"%{query}%"

    # First: search summary + tags (fast, index-only)
    sql = (
      "SELECT id, user_id, summary, category, tags, created_at, session_id "
      "FROM memory_index WHERE user_id = ? AND (LOWER(summary) LIKE LOWER(?) OR LOWER(tags) LIKE LOWER(?))"
    )
    params: list = [user_id, like_pat, like_pat]
    if category:
      sql += " AND category = ?"
      params.append(category)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()

    # Second: if no summary/tag hits, search content (slower but catches more)
    if not rows:
      sql_content = (
        "SELECT mi.id, mi.user_id, mi.summary, mi.category, mi.tags, mi.created_at, mi.session_id "
        "FROM memory_entries me JOIN memory_index mi ON me.id = mi.id "
        "WHERE mi.user_id = ? AND LOWER(me.content) LIKE LOWER(?)"
      )
      params_content: list = [user_id, like_pat]
      if category:
        sql_content += " AND mi.category = ?"
        params_content.append(category)
      sql_content += " ORDER BY mi.created_at DESC LIMIT ?"
      params_content.append(limit)
      cursor = await db.execute(sql_content, params_content)
      rows = await cursor.fetchall()

    return [
      IndexEntry(
        id=row["id"],
        user_id=row["user_id"],
        summary=row["summary"],
        category=row["category"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        created_at=datetime.fromisoformat(row["created_at"]),
        session_id=row["session_id"],
      )
      for row in rows
    ]

  async def get_entries(self, entry_ids: List[str]) -> List[MemoryEntry]:
    db = await self._ensure_initialized()
    if not entry_ids:
      return []
    placeholders = ",".join("?" for _ in entry_ids)
    cursor = await db.execute(
      f"SELECT id, user_id, content, category, created_at, updated_at, source_turn FROM memory_entries WHERE id IN ({placeholders})",
      entry_ids,
    )
    rows = await cursor.fetchall()
    return [
      MemoryEntry(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        category=row["category"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        source_turn=row["source_turn"],
      )
      for row in rows
    ]

  async def delete_entry(self, entry_id: str) -> bool:
    db = await self._ensure_initialized()
    # Delete from FTS first (before index row is gone)
    await db.execute("DELETE FROM memory_fts WHERE id = ?", (entry_id,))
    cursor = await db.execute("DELETE FROM memory_index WHERE id = ?", (entry_id,))
    await db.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
    await db.commit()
    return cursor.rowcount > 0

  async def close(self) -> None:
    if self._db is not None:
      await self._db.close()
      self._db = None
      self._initialized = False
