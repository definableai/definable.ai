"""JSONL file-based session memory store.

Human-readable storage for debugging and inspection.
Each session gets its own directory, each user gets a JSONL file.

Directory structure:
  <base_dir>/
    <session_id>/
      <user_id>.jsonl

Each line in the .jsonl file is one JSON-serialized MemoryEntry.
"""

import json
import shutil
from pathlib import Path
from typing import Any, List, Optional

from definable.memory.types import MemoryEntry
from definable.utils.log import log_debug


class FileStore:
  """JSONL file-based session memory store.

  Args:
    base_dir: Root directory for memory files. Defaults to ".memory".
  """

  def __init__(self, base_dir: str = ".memory") -> None:
    self.base_dir = Path(base_dir)
    self._initialized = False

  async def initialize(self) -> None:
    if self._initialized:
      return
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self._initialized = True
    log_debug("FileStore initialized", log_level=2)

  async def close(self) -> None:
    self._initialized = False

  async def _ensure_initialized(self) -> None:
    if not self._initialized:
      await self.initialize()

  def _get_file_path(self, session_id: str, user_id: str) -> Path:
    """Get the JSONL file path for a session + user."""
    return self.base_dir / session_id / f"{user_id}.jsonl"

  def _read_all_entries(self, path: Path) -> List[MemoryEntry]:
    """Read and parse all entries from a JSONL file."""
    if not path.exists():
      return []
    entries: List[MemoryEntry] = []
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          data = json.loads(line)
          entries.append(MemoryEntry.from_dict(data))
        except (json.JSONDecodeError, KeyError):
          continue
    return entries

  def _write_all_entries(self, path: Path, entries: List[MemoryEntry]) -> None:
    """Write all entries to a JSONL file (full rewrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      for entry in entries:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

  async def add(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    path = self._get_file_path(entry.session_id, entry.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
      f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

  async def get_entries(
    self,
    session_id: str,
    user_id: str = "default",
    limit: Optional[int] = None,
  ) -> List[MemoryEntry]:
    await self._ensure_initialized()
    path = self._get_file_path(session_id, user_id)
    entries = self._read_all_entries(path)
    entries.sort(key=lambda e: e.created_at or 0.0)
    if limit is not None:
      entries = entries[:limit]
    return entries

  async def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
    await self._ensure_initialized()
    # Scan all session directories
    if not self.base_dir.exists():
      return None
    for session_dir in self.base_dir.iterdir():
      if not session_dir.is_dir():
        continue
      for jsonl_file in session_dir.glob("*.jsonl"):
        for entry in self._read_all_entries(jsonl_file):
          if entry.memory_id == memory_id:
            return entry
    return None

  async def update(self, entry: MemoryEntry) -> None:
    await self._ensure_initialized()
    path = self._get_file_path(entry.session_id, entry.user_id)
    entries = self._read_all_entries(path)
    updated = False
    for i, e in enumerate(entries):
      if e.memory_id == entry.memory_id:
        entries[i] = entry
        updated = True
        break
    if updated:
      self._write_all_entries(path, entries)

  async def delete(self, memory_id: str) -> None:
    await self._ensure_initialized()
    if not self.base_dir.exists():
      return
    for session_dir in self.base_dir.iterdir():
      if not session_dir.is_dir():
        continue
      for jsonl_file in session_dir.glob("*.jsonl"):
        entries = self._read_all_entries(jsonl_file)
        filtered = [e for e in entries if e.memory_id != memory_id]
        if len(filtered) < len(entries):
          self._write_all_entries(jsonl_file, filtered)
          return

  async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
    await self._ensure_initialized()
    session_dir = self.base_dir / session_id
    if not session_dir.exists():
      return
    if user_id is not None:
      file_path = session_dir / f"{user_id}.jsonl"
      if file_path.exists():
        file_path.unlink()
      # Clean up empty directory
      if session_dir.exists() and not any(session_dir.iterdir()):
        session_dir.rmdir()
    else:
      shutil.rmtree(session_dir, ignore_errors=True)

  async def count(self, session_id: str, user_id: str = "default") -> int:
    await self._ensure_initialized()
    path = self._get_file_path(session_id, user_id)
    if not path.exists():
      return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          count += 1
    return count

  async def __aenter__(self) -> "FileStore":
    await self.initialize()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()
