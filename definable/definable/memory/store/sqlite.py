"""SQLite-backed session memory store using aiosqlite."""

import json
from typing import Any, List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


class SQLiteStore:
  """Async SQLite session memory store.

  Single table ``session_entries`` with JSON-encoded message_data.
  Tables are auto-created on first ``initialize()`` call.
  """

  def __init__(self, db_path: str = "./memory.db"):
    self.db_path = db_path
    self._db: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import aiosqlite
    except ImportError as e:
      raise ImportError("aiosqlite is required for SQLiteStore. Install it with: pip install aiosqlite") from e

    self._db = await aiosqlite.connect(self.db_path)
    self._db.row_factory = None
    await self._create_tables()
    self._initialized = True
    log_debug("SQLiteStore initialized", log_level=2)

  async def close(self) -> None:
    if self._db:
      await self._db.close()
      self._db = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_tables(self) -> None:
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS session_entries (
        memory_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT 'default',
        role TEXT NOT NULL DEFAULT 'user',
        content TEXT NOT NULL DEFAULT '',
        message_data TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_session_entries_session ON session_entries(session_id);
      CREATE INDEX IF NOT EXISTS idx_session_entries_user ON session_entries(user_id);
      CREATE INDEX IF NOT EXISTS idx_session_entries_created ON session_entries(created_at);
    """)
    await self._db.commit()

  async def add(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    msg_data = json.dumps(entry.message_data) if entry.message_data else None
    await self._db.execute(
      """INSERT INTO session_entries (memory_id, session_id, user_id, role, content, message_data, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
      (entry.memory_id, entry.session_id, entry.user_id, entry.role, entry.content, msg_data, entry.created_at, entry.updated_at),
    )
    await self._db.commit()

  async def get_entries(
    self,
    session_id: str,
    user_id: str = "default",
    limit: Optional[int] = None,
  ) -> List[MemoryEntry]:
    await self._ensure_initialized()
    query = "SELECT * FROM session_entries WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC"
    params: List[Any] = [session_id, user_id]
    if limit is not None:
      query += " LIMIT ?"
      params.append(limit)

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_entry(row) for row in rows]

  async def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
    await self._ensure_initialized()
    cursor = await self._db.execute("SELECT * FROM session_entries WHERE memory_id = ?", (memory_id,))
    row = await cursor.fetchone()
    return self._row_to_entry(row) if row else None

  async def update(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    msg_data = json.dumps(entry.message_data) if entry.message_data else None
    await self._db.execute(
      """UPDATE session_entries SET content = ?, role = ?, message_data = ?, updated_at = ?
         WHERE memory_id = ?""",
      (entry.content, entry.role, msg_data, entry.updated_at, entry.memory_id),
    )
    await self._db.commit()

  async def delete(self, memory_id: str) -> None:
    await self._ensure_initialized()
    await self._db.execute("DELETE FROM session_entries WHERE memory_id = ?", (memory_id,))
    await self._db.commit()

  async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    if user_id is not None:
      await self._db.execute("DELETE FROM session_entries WHERE session_id = ? AND user_id = ?", (session_id, user_id))
    else:
      await self._db.execute("DELETE FROM session_entries WHERE session_id = ?", (session_id,))
    await self._db.commit()

  async def count(self, session_id: str, user_id: str = "default") -> int:
    await self._ensure_initialized()
    cursor = await self._db.execute(
      "SELECT COUNT(*) FROM session_entries WHERE session_id = ? AND user_id = ?",
      (session_id, user_id),
    )
    row = await cursor.fetchone()
    return row[0] if row else 0

  def _row_to_entry(self, row: Any) -> MemoryEntry:
    return MemoryEntry(
      memory_id=row[0],
      session_id=row[1],
      user_id=row[2],
      role=row[3],
      content=row[4],
      message_data=json.loads(row[5]) if row[5] else None,
      created_at=row[6],
      updated_at=row[7],
    )

  async def __aenter__(self) -> "SQLiteStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
