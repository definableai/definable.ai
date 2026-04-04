"""Per-user SQLite store — one database file per user for tenant isolation.

Eliminates write contention and provides physical data isolation.
GDPR deletion is trivial: delete the file.

Usage:
    store = PerUserSQLiteStore(base_dir="./memory_data")
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from definable.memory.v2.models import (
  IndexEntry,
  MemoryEntry,
  MemoryStats,
  WarmMemory,
  WorkingMemory,
  WorkingMemorySnapshot,
)
from definable.memory.v2.stores.base import MemoryStore
from definable.memory.v2.stores.sqlite import SQLiteStore


def _safe_filename(user_id: str) -> str:
  """Convert user_id to a safe filename."""
  # Replace any non-alphanumeric chars with underscore
  return "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)


class PerUserSQLiteStore(MemoryStore):
  """One SQLite file per user — zero write contention, physical tenant isolation.

  Each user gets their own database file at {base_dir}/{user_id}.db.
  GDPR deletion is file deletion.
  """

  def __init__(self, base_dir: str = "./memory_data", *, half_life_days: float = 30.0) -> None:
    self._base_dir = Path(base_dir)
    self._base_dir.mkdir(parents=True, exist_ok=True)
    self._half_life_days = half_life_days
    self._stores: dict[str, SQLiteStore] = {}

  def _get_store(self, user_id: str) -> SQLiteStore:
    if user_id not in self._stores:
      db_path = self._base_dir / f"{_safe_filename(user_id)}.db"
      self._stores[user_id] = SQLiteStore(str(db_path), half_life_days=self._half_life_days)
    return self._stores[user_id]

  # --- Core CRUD (delegate to per-user store) ---

  async def get_working_memory(self, user_id: str) -> Optional[WorkingMemory]:
    return await self._get_store(user_id).get_working_memory(user_id)

  async def set_working_memory(self, user_id: str, content: str, *, session_id: str = "") -> WorkingMemory:
    return await self._get_store(user_id).set_working_memory(user_id, content, session_id=session_id)

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
    return await self._get_store(user_id).add_entry(
      user_id,
      summary,
      content,
      category,
      tags,
      session_id,
      confidence=confidence,
      source=source,
      expires_at=expires_at,
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
    return await self._get_store(user_id).search_index(user_id, query, category, limit, after=after, before=before)

  async def get_entries(self, entry_ids: List[str]) -> List[MemoryEntry]:
    # Entry IDs are globally unique, but we need the user_id to route.
    # Since entries are fetched after search (which is user-scoped),
    # we check all open stores. In practice, only one store matters per request.
    for store in self._stores.values():
      result = await store.get_entries(entry_ids)
      if result:
        return result
    return []

  async def delete_entry(self, entry_id: str) -> bool:
    for store in self._stores.values():
      if await store.delete_entry(entry_id):
        return True
    return False

  # --- Warm memory ---

  async def get_warm_memory(self, user_id: str) -> Optional[WarmMemory]:
    return await self._get_store(user_id).get_warm_memory(user_id)

  async def set_warm_memory(self, user_id: str, content: str) -> WarmMemory:
    return await self._get_store(user_id).set_warm_memory(user_id, content)

  # --- WM history ---

  async def get_wm_history(self, user_id: str, limit: int = 20) -> List[WorkingMemorySnapshot]:
    return await self._get_store(user_id).get_wm_history(user_id, limit)

  async def rollback_working_memory(self, user_id: str, version: int) -> Optional[WorkingMemory]:
    return await self._get_store(user_id).rollback_working_memory(user_id, version)

  # --- Admin / GDPR ---

  async def delete_user(self, user_id: str) -> int:
    store = self._get_store(user_id)
    count = await store.delete_user(user_id)
    await store.close()
    self._stores.pop(user_id, None)
    # Delete the file for complete erasure
    db_path = self._base_dir / f"{_safe_filename(user_id)}.db"
    for ext in ["", "-shm", "-wal"]:
      p = Path(str(db_path) + ext)
      if p.exists():
        p.unlink()
    return count

  async def export_user(self, user_id: str) -> dict:
    return await self._get_store(user_id).export_user(user_id)

  async def count_entries(self, user_id: str) -> int:
    return await self._get_store(user_id).count_entries(user_id)

  async def get_stats(self, user_id: str) -> MemoryStats:
    return await self._get_store(user_id).get_stats(user_id)

  # --- Maintenance ---

  async def prune_expired(self, user_id: str) -> int:
    return await self._get_store(user_id).prune_expired(user_id)

  async def enforce_limit(self, user_id: str, max_entries: int) -> int:
    return await self._get_store(user_id).enforce_limit(user_id, max_entries)

  async def close(self) -> None:
    for store in self._stores.values():
      await store.close()
    self._stores.clear()
