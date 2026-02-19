"""Base protocol for memory stores."""

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.memory.types import MemoryEntry


@runtime_checkable
class MemoryStore(Protocol):
  """Protocol for session memory storage backends.

  All methods are async. Stores must implement all methods.
  Lifecycle: call ``initialize()`` before use, ``close()`` when done.
  """

  async def initialize(self) -> None:
    """Prepare the store (create tables, open connections, etc.)."""
    ...

  async def close(self) -> None:
    """Release resources (close connections, flush buffers, etc.)."""
    ...

  async def add(self, entry: "MemoryEntry") -> None:
    """Add a new entry to the store."""
    ...

  async def get_entries(
    self,
    session_id: str,
    user_id: str = "default",
    limit: Optional[int] = None,
  ) -> List["MemoryEntry"]:
    """Retrieve entries for a session, ordered by created_at ascending."""
    ...

  async def get_entry(self, memory_id: str) -> Optional["MemoryEntry"]:
    """Retrieve a single entry by ID."""
    ...

  async def update(self, entry: "MemoryEntry") -> None:
    """Update an existing entry (matched by memory_id)."""
    ...

  async def delete(self, memory_id: str) -> None:
    """Delete a single entry by ID."""
    ...

  async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
    """Delete all entries for a session (optionally scoped to a user)."""
    ...

  async def count(self, session_id: str, user_id: str = "default") -> int:
    """Count entries for a session + user."""
    ...
