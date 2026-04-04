"""Abstract base for memory stores."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from definable.memory.v2.models import (
  IndexEntry,
  MemoryEntry,
  MemoryStats,
  WarmMemory,
  WorkingMemory,
  WorkingMemorySnapshot,
)


class MemoryStore(ABC):
  """Pluggable backend for memory persistence.

  Subclass this to implement custom storage backends (SQLite, Postgres, etc.).
  """

  # --- Core CRUD ---

  @abstractmethod
  async def get_working_memory(self, user_id: str) -> Optional[WorkingMemory]:
    """Load the working memory scratchpad for a user."""
    ...

  @abstractmethod
  async def set_working_memory(self, user_id: str, content: str, *, session_id: str = "") -> WorkingMemory:
    """Replace the working memory content. Returns the updated object.

    Also stores a snapshot in WM history for rollback.
    """
    ...

  @abstractmethod
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
    """Archive a new memory entry. Returns the index entry."""
    ...

  @abstractmethod
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
    """Search the memory index with recency-weighted ranking.

    Args:
      after: Only return entries created after this date.
      before: Only return entries created before this date.
    """
    ...

  @abstractmethod
  async def get_entries(self, entry_ids: List[str]) -> List[MemoryEntry]:
    """Load full content for specific entry IDs. Bumps access_count."""
    ...

  @abstractmethod
  async def delete_entry(self, entry_id: str) -> bool:
    """Delete a memory entry. Returns True if found and deleted."""
    ...

  # --- Warm memory (extended tier) ---

  async def get_warm_memory(self, user_id: str) -> Optional[WarmMemory]:
    """Load the warm (extended) working memory tier."""
    return None

  async def set_warm_memory(self, user_id: str, content: str) -> WarmMemory:
    """Replace the warm memory content."""
    return WarmMemory(user_id=user_id, content=content)

  # --- WM history (rollback) ---

  async def get_wm_history(self, user_id: str, limit: int = 20) -> List[WorkingMemorySnapshot]:
    """Get recent working memory snapshots for rollback."""
    return []

  async def rollback_working_memory(self, user_id: str, version: int) -> Optional[WorkingMemory]:
    """Restore working memory to a previous version."""
    return None

  # --- Admin / GDPR ---

  async def delete_user(self, user_id: str) -> int:
    """Delete ALL data for a user (GDPR). Returns count of entries deleted."""
    return 0

  async def export_user(self, user_id: str) -> dict:
    """Export all user data as a dict (GDPR data portability)."""
    return {}

  async def count_entries(self, user_id: str) -> int:
    """Count total archived entries for a user."""
    return 0

  async def get_stats(self, user_id: str) -> MemoryStats:
    """Return usage statistics for a user's memory."""
    return MemoryStats(user_id=user_id)

  # --- Maintenance ---

  async def prune_expired(self, user_id: str) -> int:
    """Delete entries past their expires_at. Returns count removed."""
    return 0

  async def enforce_limit(self, user_id: str, max_entries: int) -> int:
    """Enforce entry limit by pruning lowest-priority entries. Returns count removed."""
    return 0

  async def close(self) -> None:
    """Clean up resources. Override if store needs cleanup."""
    pass
