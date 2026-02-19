"""Pure-Python in-memory store implementing the MemoryStore protocol."""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


class InMemoryStore:
  """In-memory session store backed by a dict of lists.

  Useful for testing and short-lived processes.
  All data is lost when the process exits.
  """

  def __init__(self) -> None:
    self._entries: Dict[str, List[MemoryEntry]] = {}  # key: "{session_id}:{user_id}"
    self._by_id: Dict[str, MemoryEntry] = {}
    self._initialized = False

  def _key(self, session_id: str, user_id: str) -> str:
    return f"{session_id}:{user_id}"

  async def initialize(self) -> None:
    if self._initialized:
      return
    self._initialized = True
    log_debug("InMemoryStore initialized", log_level=2)

  async def close(self) -> None:
    self._entries.clear()
    self._by_id.clear()
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  async def add(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    key = self._key(entry.session_id, entry.user_id)
    if key not in self._entries:
      self._entries[key] = []
    copy = deepcopy(entry)
    self._entries[key].append(copy)
    self._by_id[copy.memory_id] = copy  # type: ignore[index]

  async def get_entries(
    self,
    session_id: str,
    user_id: str = "default",
    limit: Optional[int] = None,
  ) -> List[MemoryEntry]:
    await self._ensure_initialized()
    key = self._key(session_id, user_id)
    entries = list(self._entries.get(key, []))
    entries.sort(key=lambda e: e.created_at or 0.0)
    if limit is not None:
      entries = entries[:limit]
    return [deepcopy(e) for e in entries]

  async def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
    await self._ensure_initialized()
    entry = self._by_id.get(memory_id)
    return deepcopy(entry) if entry else None

  async def update(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    key = self._key(entry.session_id, entry.user_id)
    entries = self._entries.get(key, [])
    for i, e in enumerate(entries):
      if e.memory_id == entry.memory_id:
        copy = deepcopy(entry)
        entries[i] = copy
        self._by_id[entry.memory_id] = copy  # type: ignore[index]
        return

  async def delete(self, memory_id: str) -> None:
    await self._ensure_initialized()
    entry = self._by_id.pop(memory_id, None)
    if entry is None:
      return
    key = self._key(entry.session_id, entry.user_id)
    entries = self._entries.get(key, [])
    self._entries[key] = [e for e in entries if e.memory_id != memory_id]

  async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    keys_to_remove = []
    for key in list(self._entries.keys()):
      sid, uid = key.split(":", 1)
      if sid == session_id and (user_id is None or uid == user_id):
        for e in self._entries[key]:
          self._by_id.pop(e.memory_id, None)  # type: ignore[arg-type]
        keys_to_remove.append(key)
    for key in keys_to_remove:
      del self._entries[key]

  async def count(self, session_id: str, user_id: str = "default") -> int:
    await self._ensure_initialized()
    key = self._key(session_id, user_id)
    return len(self._entries.get(key, []))

  async def __aenter__(self) -> "InMemoryStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
