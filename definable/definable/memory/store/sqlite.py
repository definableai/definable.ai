"""SQLite-backed memory store using aiosqlite."""

import json
import time
from typing import Any, List, Optional

from definable.memory.types import UserMemory
from definable.utils.log import log_debug


class SQLiteStore:
  """Async SQLite memory store.

  Single table ``memories`` with JSON-encoded topics.
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
      CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        memory TEXT NOT NULL,
        topics TEXT DEFAULT '[]',
        user_id TEXT,
        agent_id TEXT,
        input TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
      CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);
      CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
    """)
    await self._db.commit()

  async def get_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[UserMemory]:
    await self._ensure_initialized()
    query = "SELECT * FROM memories WHERE memory_id = ?"
    params: List[Any] = [memory_id]
    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)

    cursor = await self._db.execute(query, params)
    row = await cursor.fetchone()
    return self._row_to_memory(row) if row else None

  async def get_user_memories(
    self,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    topics: Optional[List[str]] = None,
    limit: Optional[int] = None,
  ) -> List[UserMemory]:
    await self._ensure_initialized()
    query = "SELECT * FROM memories WHERE 1=1"
    params: List[Any] = []

    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    if agent_id is not None:
      query += " AND agent_id = ?"
      params.append(agent_id)

    query += " ORDER BY updated_at DESC"

    if limit is not None:
      query += " LIMIT ?"
      params.append(limit)

    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()

    memories = [self._row_to_memory(row) for row in rows]

    # Filter by topics in Python (JSON column)
    if topics:
      topic_set = set(topics)
      memories = [m for m in memories if set(m.topics or []).intersection(topic_set)]

    return memories

  async def upsert_user_memory(self, memory: UserMemory) -> None:
    await self._ensure_initialized()
    memory.updated_at = time.time()
    topics_json = json.dumps(memory.topics or [])

    await self._db.execute(
      """INSERT INTO memories (memory_id, memory, topics, user_id, agent_id, input, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(memory_id) DO UPDATE SET
           memory = excluded.memory,
           topics = excluded.topics,
           updated_at = excluded.updated_at""",
      (
        memory.memory_id,
        memory.memory,
        topics_json,
        memory.user_id,
        memory.agent_id,
        memory.input,
        memory.created_at,
        memory.updated_at,
      ),
    )
    await self._db.commit()

  async def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    query = "DELETE FROM memories WHERE memory_id = ?"
    params: List[Any] = [memory_id]
    if user_id is not None:
      query += " AND user_id = ?"
      params.append(user_id)
    await self._db.execute(query, params)
    await self._db.commit()

  async def clear_user_memories(self, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    if user_id is None:
      await self._db.execute("DELETE FROM memories")
    else:
      await self._db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
    await self._db.commit()

  def _row_to_memory(self, row: Any) -> UserMemory:
    return UserMemory(
      memory_id=row[0],
      memory=row[1],
      topics=json.loads(row[2]) if row[2] else [],
      user_id=row[3],
      agent_id=row[4],
      input=row[5],
      created_at=row[6],
      updated_at=row[7],
    )

  async def __aenter__(self) -> "SQLiteStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
