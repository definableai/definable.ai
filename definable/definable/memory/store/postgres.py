"""PostgreSQL-backed memory store using asyncpg."""

import json
import os
import time
from typing import Any, List, Optional

from definable.memory.types import UserMemory
from definable.utils.log import log_debug


class PostgresStore:
  """Async PostgreSQL memory store.

  Single table ``memories``. No pgvector dependency — plain SQL.
  Tables are auto-created on first ``initialize()`` call.
  """

  def __init__(self, db_url: str = "", pool_size: int = 5, table_prefix: str = "memory_"):
    self._db_url = db_url or os.environ.get("MEMORY_POSTGRES_URL", "")
    self._pool_size = pool_size
    self._prefix = table_prefix
    self._pool: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return

    try:
      import asyncpg  # noqa: F401
    except ImportError as e:
      raise ImportError("asyncpg is required for PostgresStore. Install it with: pip install asyncpg") from e

    if not self._db_url:
      raise ValueError("PostgreSQL connection URL is required. Set db_url or MEMORY_POSTGRES_URL environment variable.")

    import asyncpg

    self._pool = await asyncpg.create_pool(self._db_url, min_size=1, max_size=self._pool_size)
    await self._create_tables()
    self._initialized = True
    log_debug("PostgresStore initialized", log_level=2)

  async def close(self) -> None:
    if self._pool:
      await self._pool.close()
      self._pool = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_tables(self) -> None:
    p = self._prefix
    async with self._pool.acquire() as conn:
      await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {p}memories (
          memory_id TEXT PRIMARY KEY,
          memory TEXT NOT NULL,
          topics JSONB DEFAULT '[]'::jsonb,
          user_id TEXT,
          agent_id TEXT,
          input TEXT,
          created_at DOUBLE PRECISION NOT NULL,
          updated_at DOUBLE PRECISION NOT NULL
        )
      """)
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}memories_user_id ON {p}memories(user_id)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}memories_agent_id ON {p}memories(agent_id)")
      await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{p}memories_updated ON {p}memories(updated_at)")

  async def get_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[UserMemory]:
    await self._ensure_initialized()
    query = f"SELECT * FROM {self._prefix}memories WHERE memory_id = $1"
    params: List[Any] = [memory_id]
    idx = 1

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    row = await self._pool.fetchrow(query, *params)
    return self._row_to_memory(row) if row else None

  async def get_user_memories(
    self,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    topics: Optional[List[str]] = None,
    limit: Optional[int] = None,
  ) -> List[UserMemory]:
    await self._ensure_initialized()
    query = f"SELECT * FROM {self._prefix}memories WHERE TRUE"
    params: List[Any] = []
    idx = 0

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)
    if agent_id is not None:
      idx += 1
      query += f" AND agent_id = ${idx}"
      params.append(agent_id)
    if topics:
      # Match any topic using JSONB array overlap
      idx += 1
      query += f" AND topics ?| ${idx}"
      params.append(topics)

    query += " ORDER BY updated_at DESC"

    if limit is not None:
      idx += 1
      query += f" LIMIT ${idx}"
      params.append(limit)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_memory(row) for row in rows]

  async def upsert_user_memory(self, memory: UserMemory) -> None:
    await self._ensure_initialized()
    memory.updated_at = time.time()
    topics_json = json.dumps(memory.topics or [])

    await self._pool.execute(
      f"""INSERT INTO {self._prefix}memories
         (memory_id, memory, topics, user_id, agent_id, input, created_at, updated_at)
         VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8)
         ON CONFLICT (memory_id) DO UPDATE SET
           memory = EXCLUDED.memory,
           topics = EXCLUDED.topics,
           updated_at = EXCLUDED.updated_at""",
      memory.memory_id,
      memory.memory,
      topics_json,
      memory.user_id,
      memory.agent_id,
      memory.input,
      memory.created_at,
      memory.updated_at,
    )

  async def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    query = f"DELETE FROM {self._prefix}memories WHERE memory_id = $1"
    params: List[Any] = [memory_id]
    idx = 1

    if user_id is not None:
      idx += 1
      query += f" AND user_id = ${idx}"
      params.append(user_id)

    await self._pool.execute(query, *params)

  async def clear_user_memories(self, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    if user_id is None:
      await self._pool.execute(f"DELETE FROM {self._prefix}memories")
    else:
      await self._pool.execute(f"DELETE FROM {self._prefix}memories WHERE user_id = $1", user_id)

  def _row_to_memory(self, row: Any) -> UserMemory:
    return UserMemory(
      memory_id=row["memory_id"],
      memory=row["memory"],
      topics=row["topics"] if isinstance(row["topics"], list) else json.loads(row["topics"] or "[]"),
      user_id=row["user_id"],
      agent_id=row["agent_id"],
      input=row["input"],
      created_at=row["created_at"],
      updated_at=row["updated_at"],
    )

  async def __aenter__(self) -> "PostgresStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
