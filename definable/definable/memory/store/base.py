"""Base protocol for memory stores."""

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.memory.types import UserMemory


@runtime_checkable
class MemoryStore(Protocol):
  """Protocol for memory storage backends.

  All methods are async. Stores must implement all 7 methods.
  Lifecycle: call ``initialize()`` before use, ``close()`` when done.
  """

  async def initialize(self) -> None:
    """Prepare the store (create tables, open connections, etc.)."""
    ...

  async def close(self) -> None:
    """Release resources (close connections, flush buffers, etc.)."""
    ...

  async def get_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional["UserMemory"]:
    """Retrieve a single memory by ID.

    Args:
      memory_id: The UUID of the memory to retrieve.
      user_id: Optional user scope. If provided, the memory must belong to this user.

    Returns:
      The matching UserMemory, or None if not found.
    """
    ...

  async def get_user_memories(
    self,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    topics: Optional[List[str]] = None,
    limit: Optional[int] = None,
  ) -> List["UserMemory"]:
    """Retrieve memories with optional filters.

    Args:
      user_id: Filter by user.
      agent_id: Filter by agent.
      topics: Filter by topics (any match).
      limit: Maximum number of memories to return.

    Returns:
      List of matching UserMemory objects, ordered by updated_at descending.
    """
    ...

  async def upsert_user_memory(self, memory: "UserMemory") -> None:
    """Insert or update a memory.

    If a memory with the same ``memory_id`` exists, it is replaced.
    Otherwise, a new record is inserted.
    """
    ...

  async def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
    """Delete a single memory by ID.

    Args:
      memory_id: The UUID of the memory to delete.
      user_id: Optional user scope for safety.
    """
    ...

  async def clear_user_memories(self, user_id: Optional[str] = None) -> None:
    """Delete all memories for a user.

    Args:
      user_id: The user whose memories should be cleared.
        If None, deletes ALL memories (use with caution).
    """
    ...
