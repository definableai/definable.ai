"""
Smoke-test all MemoryStore backends.

Runs a minimal round-trip (add, get_entries, update, delete) against each backend.
Backends whose dependencies are not installed or whose services are not
reachable are skipped gracefully.

Usage:
    python definable/examples/memory/03_store_backends.py
"""

import asyncio
import os
from typing import Any, List, Tuple

from definable.memory import InMemoryStore
from definable.memory.types import MemoryEntry


async def test_store(name: str, store: Any) -> str:
  """Run a minimal round-trip against *store*. Returns a status string."""

  await store.initialize()

  # --- Add an entry ---
  entry = MemoryEntry(
    session_id="test-session",
    user_id="example-user",
    role="user",
    content="Hello from the smoke test!",
  )
  await store.add(entry)

  # --- Retrieve ---
  entries = await store.get_entries("test-session", user_id="example-user")
  assert len(entries) >= 1, f"{name}: expected at least 1 entry"
  assert entries[0].content == "Hello from the smoke test!"

  # --- Get single ---
  single = await store.get_entry(entry.memory_id)
  assert single is not None, f"{name}: get_entry returned None"

  # --- Update ---
  fetched = await store.get_entry(entry.memory_id)
  assert fetched is not None
  fetched.content = "Updated content!"
  await store.update(fetched)
  updated = await store.get_entry(entry.memory_id)
  assert updated is not None and updated.content == "Updated content!"

  # --- Delete ---
  await store.delete(entry.memory_id)
  after_delete = await store.get_entries("test-session", user_id="example-user")
  assert len(after_delete) == 0, f"{name}: expected 0 after delete, got {len(after_delete)}"

  # --- Cleanup ---
  await store.close()

  return "PASS"


def _build_backends() -> List[Tuple[str, Any]]:
  """Instantiate each backend, returning (name, store) pairs.

  Backends whose optional dependencies are missing are silently skipped.
  """
  backends: List[Tuple[str, Any]] = []

  # 1. InMemoryStore — always available
  backends.append(("InMemoryStore", InMemoryStore()))

  # 2. SQLiteStore
  try:
    from definable.memory import SQLiteStore

    backends.append(("SQLiteStore", SQLiteStore("./test_example.db")))
  except ImportError:
    pass

  # 3. FileStore
  try:
    from definable.memory import FileStore

    backends.append(("FileStore", FileStore("./test_file_store")))
  except ImportError:
    pass

  return backends


async def main():
  backends = _build_backends()
  results: List[Tuple[str, str]] = []

  print(f"Testing {len(backends)} backend(s)...\n")

  for name, store in backends:
    try:
      status = await test_store(name, store)
    except Exception as exc:
      status = f"FAIL ({type(exc).__name__}: {exc})"
    results.append((name, status))
    symbol = "+" if status == "PASS" else "x"
    print(f"  [{symbol}] {name}: {status}")

  # Cleanup test files
  for path in ("./test_example.db", "./test_file_store"):
    if os.path.exists(path):
      if os.path.isdir(path):
        import shutil

        shutil.rmtree(path)
      else:
        os.remove(path)

  # Summary table
  passed = sum(1 for _, s in results if s == "PASS")
  print(f"\n{'=' * 50}")
  print(f"Results: {passed}/{len(results)} backends passed")
  print(f"{'=' * 50}")
  for name, status in results:
    print(f"  {name:<25s} {status}")


if __name__ == "__main__":
  asyncio.run(main())
