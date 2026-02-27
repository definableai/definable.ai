"""SQLite-backed session memory store using aiosqlite."""

from __future__ import annotations

from types import TracebackType
import json
from typing import Any, List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


class SQLiteStore:
  """Async SQLite session memory store.

  Single table ``session_entries`` with JSON-encoded message_data.
  Tables are auto-created on first ``initialize()`` call.
  """

  def __init__(self, db_path: str | None = None):
    if db_path is None:
      from definable.utils.workspace import workspace_path

      db_path = str(workspace_path("memory.db"))
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
        updated_at REAL NOT NULL,
        entry_type TEXT NOT NULL DEFAULT 'message',
        semantic_data TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_session_entries_session ON session_entries(session_id);
      CREATE INDEX IF NOT EXISTS idx_session_entries_user ON session_entries(user_id);
      CREATE INDEX IF NOT EXISTS idx_session_entries_created ON session_entries(created_at);
      CREATE INDEX IF NOT EXISTS idx_session_entries_entry_type ON session_entries(entry_type);
    """)
    await self._db.commit()
    # Migrate existing databases that lack the new columns.
    await self._migrate_semantic_columns()

  async def _migrate_semantic_columns(self) -> None:
    """Add semantic columns to existing databases (no-op for fresh installs)."""
    cursor = await self._db.execute("PRAGMA table_info(session_entries)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "entry_type" not in columns:
      await self._db.execute("ALTER TABLE session_entries ADD COLUMN entry_type TEXT NOT NULL DEFAULT 'message'")
    if "semantic_data" not in columns:
      await self._db.execute("ALTER TABLE session_entries ADD COLUMN semantic_data TEXT")
    await self._db.commit()

  async def add(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    msg_data = json.dumps(entry.message_data) if entry.message_data else None
    semantic_data = self._pack_semantic_data(entry)
    await self._db.execute(
      """INSERT INTO session_entries
         (memory_id, session_id, user_id, role, content, message_data, created_at, updated_at, entry_type, semantic_data)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        entry.memory_id,
        entry.session_id,
        entry.user_id,
        entry.role,
        entry.content,
        msg_data,
        entry.created_at,
        entry.updated_at,
        entry.entry_type,
        semantic_data,
      ),
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
    params: list[object] = [session_id, user_id]
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
    semantic_data = self._pack_semantic_data(entry)
    await self._db.execute(
      """UPDATE session_entries SET content = ?, role = ?, message_data = ?, updated_at = ?,
         entry_type = ?, semantic_data = ?
         WHERE memory_id = ?""",
      (entry.content, entry.role, msg_data, entry.updated_at, entry.entry_type, semantic_data, entry.memory_id),
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

  @staticmethod
  def _pack_semantic_data(entry: MemoryEntry) -> Optional[str]:
    """Pack semantic fields into JSON. Returns None for plain messages."""
    if entry.entry_type == "message" and not entry.lossless_content:
      return None
    d: dict[str, Any] = {
      "lossless_content": entry.lossless_content,
      "keywords": entry.keywords or [],
      "entities": entry.entities or [],
      "persons": entry.persons or [],
      "topic": entry.topic,
      "importance": entry.importance,
      "superseded_by": entry.superseded_by,
    }
    if entry.vector is not None:
      d["vector"] = entry.vector
    return json.dumps(d)

  def _row_to_entry(self, row: tuple[Any, ...]) -> MemoryEntry:
    entry = MemoryEntry(
      memory_id=row[0],
      session_id=row[1],
      user_id=row[2],
      role=row[3],
      content=row[4],
      message_data=json.loads(row[5]) if row[5] else None,
      created_at=row[6],
      updated_at=row[7],
      entry_type=row[8] if len(row) > 8 and row[8] else "message",
    )
    # Unpack semantic_data JSON if present.
    if len(row) > 9 and row[9]:
      sd = json.loads(row[9])
      entry.lossless_content = sd.get("lossless_content")
      entry.keywords = sd.get("keywords", [])
      entry.entities = sd.get("entities", [])
      entry.persons = sd.get("persons", [])
      entry.topic = sd.get("topic")
      entry.importance = sd.get("importance", 0.5)
      entry.vector = sd.get("vector")
      entry.superseded_by = sd.get("superseded_by")
    return entry

  async def __aenter__(self) -> "SQLiteStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
    await self.close()
