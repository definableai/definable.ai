"""Pure-Python in-memory store implementing the MemoryStore protocol."""

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from definable.memory.types import UserMemory
from definable.utils.log import log_debug


class InMemoryStore:
  """In-memory memory store backed by a plain dict.

  Useful for testing, development, and short-lived processes that do not
  require persistence. All data is lost when the process exits.
  """

  def __init__(self) -> None:
    self._memories: Dict[str, UserMemory] = {}
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return
    self._initialized = True
    log_debug("InMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    self._memories.clear()
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def get_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> Optional[UserMemory]:
    await self._ensure_initialized()
    mem = self._memories.get(memory_id)
    if mem is None:
      return None
    if user_id is not None and mem.user_id != user_id:
      return None
    return deepcopy(mem)

  async def get_user_memories(
    self,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    topics: Optional[List[str]] = None,
    limit: Optional[int] = None,
  ) -> List[UserMemory]:
    await self._ensure_initialized()
    results: List[UserMemory] = []

    for mem in self._memories.values():
      if user_id is not None and mem.user_id != user_id:
        continue
      if agent_id is not None and mem.agent_id != agent_id:
        continue
      if topics:
        mem_topics = set(mem.topics or [])
        if not mem_topics.intersection(topics):
          continue
      results.append(deepcopy(mem))

    results.sort(key=lambda m: m.updated_at or 0.0, reverse=True)
    if limit is not None:
      results = results[:limit]
    return results

  async def upsert_user_memory(self, memory: UserMemory) -> None:
    await self._ensure_initialized()
    memory.updated_at = time.time()
    self._memories[memory.memory_id] = deepcopy(memory)  # type: ignore[arg-type, index]

  async def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    mem = self._memories.get(memory_id)
    if mem is None:
      return
    if user_id is not None and mem.user_id != user_id:
      return
    del self._memories[memory_id]

  async def clear_user_memories(self, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    if user_id is None:
      self._memories.clear()
    else:
      self._memories = {mid: m for mid, m in self._memories.items() if m.user_id != user_id}

  async def __aenter__(self) -> "InMemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
