"""Abstract base for memory stores."""

from abc import ABC, abstractmethod
from typing import List, Optional

from definable.memory.v2.models import IndexEntry, MemoryEntry, WorkingMemory


class MemoryStore(ABC):
  """Pluggable backend for memory persistence."""

  @abstractmethod
  async def get_working_memory(self, user_id: str) -> Optional[WorkingMemory]:
    """Load the working memory scratchpad for a user."""
    ...

  @abstractmethod
  async def set_working_memory(self, user_id: str, content: str) -> WorkingMemory:
    """Replace the working memory content. Returns the updated object."""
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
  ) -> List[IndexEntry]:
    """Search the memory index. Returns matching summaries."""
    ...

  @abstractmethod
  async def get_entries(self, entry_ids: List[str]) -> List[MemoryEntry]:
    """Load full content for specific entry IDs."""
    ...

  @abstractmethod
  async def delete_entry(self, entry_id: str) -> bool:
    """Delete a memory entry. Returns True if found and deleted."""
    ...

  async def close(self) -> None:
    """Clean up resources. Override if store needs cleanup."""
    pass
