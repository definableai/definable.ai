"""CortexStore — unified SQLite-backed storage for Cortex memory records.

All Cortex data lives in a single SQLite database (cortex.db) with separate
tables for records, scratchpad, and index data (signatures, graph edges, tags).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from definable.memory.cortex.record.scratchpad import Scratchpad
from definable.memory.cortex.record.types import MemoryRecord
from definable.utils.log import log_debug


class CortexStore:
  """Async SQLite store for Cortex memory records and scratchpad.

  Tables:
    cortex_records: Main memory records with JSON-encoded multi-representation data.
    cortex_scratchpad: Per-session/user belief state (single row each).
  """

  def __init__(self, db_path: Optional[str] = None):
    if db_path is None:
      from definable.utils.workspace import workspace_path

      db_path = str(workspace_path("cortex.db"))
    self.db_path = db_path
    self._db: Any = None
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return
    try:
      import aiosqlite
    except ImportError as e:
      raise ImportError("aiosqlite is required for CortexStore. Install with: pip install aiosqlite") from e

    self._db = await aiosqlite.connect(self.db_path)
    self._db.row_factory = None
    await self._create_tables()
    self._initialized = True
    log_debug("CortexStore initialized", log_level=2)

  async def close(self) -> None:
    if self._db:
      await self._db.close()
      self._db = None
      self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def _create_tables(self) -> None:
    assert self._db is not None
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS cortex_records (
        record_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT 'default',
        source TEXT NOT NULL DEFAULT 'conversation',
        raw_content TEXT NOT NULL DEFAULT '',
        role TEXT NOT NULL DEFAULT 'user',
        turn_index INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        staleness REAL NOT NULL DEFAULT 0.0,
        superseded_by TEXT,
        record_data TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_cortex_records_session ON cortex_records(session_id);
      CREATE INDEX IF NOT EXISTS idx_cortex_records_user ON cortex_records(user_id);
      CREATE INDEX IF NOT EXISTS idx_cortex_records_created ON cortex_records(created_at);
      CREATE INDEX IF NOT EXISTS idx_cortex_records_active ON cortex_records(superseded_by);

      CREATE TABLE IF NOT EXISTS cortex_scratchpad (
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT 'default',
        data TEXT NOT NULL DEFAULT '{}',
        updated_at REAL NOT NULL,
        PRIMARY KEY (session_id, user_id)
      );
    """)
    await self._db.commit()

  # --- Record CRUD ---

  async def add_record(self, record: MemoryRecord) -> None:
    """Insert a new memory record."""
    await self._ensure_initialized()
    assert self._db is not None
    record_data = json.dumps(self._pack_record_data(record))
    await self._db.execute(
      """INSERT OR REPLACE INTO cortex_records
         (record_id, session_id, user_id, source, raw_content, role, turn_index,
          created_at, updated_at, staleness, superseded_by, record_data)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
        record.record_id,
        record.session_id,
        record.user_id,
        record.source.value,
        record.raw_content,
        record.role,
        record.turn_index,
        record.created_at,
        record.updated_at,
        record.staleness,
        record.superseded_by,
        record_data,
      ),
    )
    await self._db.commit()

  async def get_record(self, record_id: str) -> Optional[MemoryRecord]:
    """Get a single record by ID."""
    await self._ensure_initialized()
    assert self._db is not None
    cursor = await self._db.execute("SELECT * FROM cortex_records WHERE record_id = ?", (record_id,))
    row = await cursor.fetchone()
    return self._row_to_record(row) if row else None

  async def get_records(
    self,
    session_id: str,
    user_id: str = "default",
    active_only: bool = True,
    limit: Optional[int] = None,
  ) -> List[MemoryRecord]:
    """Get records for a session, ordered by created_at ascending."""
    await self._ensure_initialized()
    assert self._db is not None
    query = "SELECT * FROM cortex_records WHERE session_id = ? AND user_id = ?"
    params: list[Any] = [session_id, user_id]
    if active_only:
      query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at ASC"
    if limit is not None:
      query += " LIMIT ?"
      params.append(limit)
    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_record(row) for row in rows]

  async def get_all_records(
    self,
    user_id: str = "default",
    active_only: bool = True,
    limit: Optional[int] = None,
  ) -> List[MemoryRecord]:
    """Get all records for a user across all sessions."""
    await self._ensure_initialized()
    assert self._db is not None
    query = "SELECT * FROM cortex_records WHERE user_id = ?"
    params: list[Any] = [user_id]
    if active_only:
      query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at ASC"
    if limit is not None:
      query += " LIMIT ?"
      params.append(limit)
    cursor = await self._db.execute(query, params)
    rows = await cursor.fetchall()
    return [self._row_to_record(row) for row in rows]

  async def update_record(self, record: MemoryRecord) -> None:
    """Update an existing record."""
    await self._ensure_initialized()
    assert self._db is not None
    record_data = json.dumps(self._pack_record_data(record))
    await self._db.execute(
      """UPDATE cortex_records SET raw_content = ?, updated_at = ?, staleness = ?,
         superseded_by = ?, record_data = ?
         WHERE record_id = ?""",
      (record.raw_content, record.updated_at, record.staleness, record.superseded_by, record_data, record.record_id),
    )
    await self._db.commit()

  async def delete_record(self, record_id: str) -> None:
    """Hard-delete a record (prefer soft-delete via superseded_by)."""
    await self._ensure_initialized()
    assert self._db is not None
    await self._db.execute("DELETE FROM cortex_records WHERE record_id = ?", (record_id,))
    await self._db.commit()

  async def count_records(self, session_id: str, user_id: str = "default", active_only: bool = True) -> int:
    """Count records for a session/user."""
    await self._ensure_initialized()
    assert self._db is not None
    query = "SELECT COUNT(*) FROM cortex_records WHERE session_id = ? AND user_id = ?"
    params: list[Any] = [session_id, user_id]
    if active_only:
      query += " AND superseded_by IS NULL"
    cursor = await self._db.execute(query, params)
    row = await cursor.fetchone()
    return row[0] if row else 0

  # --- Scratchpad ---

  async def get_scratchpad(self, session_id: str, user_id: str = "default") -> Scratchpad:
    """Get the scratchpad for a session/user. Returns empty if not found."""
    await self._ensure_initialized()
    assert self._db is not None
    cursor = await self._db.execute(
      "SELECT data, updated_at FROM cortex_scratchpad WHERE session_id = ? AND user_id = ?",
      (session_id, user_id),
    )
    row = await cursor.fetchone()
    if row:
      data = json.loads(row[0])
      data["session_id"] = session_id
      data["user_id"] = user_id
      data["updated_at"] = row[1]
      return Scratchpad.from_dict(data)
    return Scratchpad(session_id=session_id, user_id=user_id)

  async def save_scratchpad(self, scratchpad: Scratchpad) -> None:
    """Upsert the scratchpad for a session/user."""
    await self._ensure_initialized()
    assert self._db is not None
    data = json.dumps({
      "beliefs": scratchpad.beliefs,
      "active_topics": scratchpad.active_topics,
      "pending_tasks": scratchpad.pending_tasks,
    })
    await self._db.execute(
      """INSERT OR REPLACE INTO cortex_scratchpad (session_id, user_id, data, updated_at)
         VALUES (?, ?, ?, ?)""",
      (scratchpad.session_id, scratchpad.user_id, data, scratchpad.updated_at),
    )
    await self._db.commit()

  # --- Bulk operations ---

  async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
    """Delete all records and scratchpad for a session."""
    await self._ensure_initialized()
    assert self._db is not None
    if user_id is not None:
      await self._db.execute("DELETE FROM cortex_records WHERE session_id = ? AND user_id = ?", (session_id, user_id))
      await self._db.execute("DELETE FROM cortex_scratchpad WHERE session_id = ? AND user_id = ?", (session_id, user_id))
    else:
      await self._db.execute("DELETE FROM cortex_records WHERE session_id = ?", (session_id,))
      await self._db.execute("DELETE FROM cortex_scratchpad WHERE session_id = ?", (session_id,))
    await self._db.commit()

  # --- Serialization helpers ---

  @staticmethod
  def _pack_record_data(record: MemoryRecord) -> Dict[str, Any]:
    """Pack multi-representation fields into a JSON-serializable dict."""
    d: Dict[str, Any] = {}
    if record.narrative:
      d["narrative"] = record.narrative.to_dict()
    if record.facts:
      d["facts"] = [f.to_dict() for f in record.facts]
    if record.tags:
      d["tags"] = record.tags
    if record.signature is not None:
      import base64

      d["signature"] = base64.b64encode(record.signature).decode("ascii")
    if record.embedding is not None:
      d["embedding"] = record.embedding
    return d

  def _row_to_record(self, row: tuple[Any, ...]) -> MemoryRecord:
    """Convert a database row to a MemoryRecord."""
    from definable.memory.cortex.record.types import MemorySource, NarrativeEpisode, Fact

    record = MemoryRecord(
      record_id=row[0],
      session_id=row[1],
      user_id=row[2],
      source=MemorySource(row[3]) if row[3] else MemorySource.CONVERSATION,
      raw_content=row[4] or "",
      role=row[5] or "user",
      turn_index=row[6] or 0,
      created_at=row[7],
      updated_at=row[8],
      staleness=row[9] or 0.0,
      superseded_by=row[10],
    )
    # Unpack record_data JSON
    if row[11]:
      rd = json.loads(row[11])
      if "narrative" in rd and rd["narrative"]:
        record.narrative = NarrativeEpisode.from_dict(rd["narrative"])
      if "facts" in rd:
        record.facts = [Fact.from_dict(f) for f in rd["facts"]]
      if "tags" in rd:
        record.tags = rd["tags"]
      if "signature" in rd and rd["signature"]:
        import base64

        record.signature = base64.b64decode(rd["signature"])
      if "embedding" in rd:
        record.embedding = rd["embedding"]
    return record

  # --- Lifecycle ---

  async def __aenter__(self) -> "CortexStore":
    await self.initialize()
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.close()
