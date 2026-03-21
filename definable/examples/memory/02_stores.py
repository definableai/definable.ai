"""
MemoryStore protocol and backends.

This example shows how to:
- Use the MemoryStore protocol (add, get, update, delete, count)
- Test all available backends (InMemory, SQLite, File)

No API keys required.

Usage:
    python definable/examples/memory/02_stores.py
"""

import asyncio
import os
from typing import Any, List, Tuple

from definable.memory import InMemoryStore
from definable.memory.types import MemoryEntry


async def protocol_walkthrough():
  """Exercise all MemoryStore protocol methods with InMemoryStore."""
  print("MemoryStore Protocol Walkthrough")
  print("=" * 60)

  store = InMemoryStore()
  await store.initialize()

  # --- Add ---
  entry1 = MemoryEntry(session_id="s1", user_id="alice", role="user", content="Hello, I'm Alice from San Francisco.")
  entry2 = MemoryEntry(session_id="s1", user_id="alice", role="assistant", content="Nice to meet you, Alice!")
  entry3 = MemoryEntry(session_id="s1", user_id="bob", role="user", content="Bob prefers dark mode.")

  for e in [entry1, entry2, entry3]:
    await store.add(e)
    print(f"  add: [{e.role}] {e.content}")

  # --- Get entries ---
  alice_entries = await store.get_entries("s1", user_id="alice")
  assert len(alice_entries) == 2
  print(f"\n  get_entries(alice): {len(alice_entries)} entries")

  # --- Get single ---
  single = await store.get_entry(entry1.memory_id)  # type: ignore[arg-type]
  assert single is not None
  print(f"  get_entry: {single.content!r}")

  # --- Count ---
  count = await store.count("s1", user_id="alice")
  assert count == 2
  print(f"  count(alice): {count}")

  # --- Update ---
  single.content = "Hello, I'm Alice. I moved to New York!"
  await store.update(single)
  updated = await store.get_entry(entry1.memory_id)  # type: ignore[arg-type]
  assert updated is not None and "New York" in updated.content
  print(f"  update: {updated.content!r}")

  # --- Delete ---
  await store.delete(entry2.memory_id)  # type: ignore[arg-type]
  assert len(await store.get_entries("s1", user_id="alice")) == 1
  print("  delete: removed entry2")

  # --- Delete session ---
  await store.delete_session("s1", user_id="alice")
  assert len(await store.get_entries("s1", user_id="alice")) == 0
  bob_entries = await store.get_entries("s1", user_id="bob")
  assert len(bob_entries) == 1
  print(f"  delete_session(alice): bob still has {len(bob_entries)} entry")

  await store.close()
  print("\nAll protocol methods verified!\n")


async def test_store(name: str, store: Any) -> str:
  """Run a minimal round-trip against a store backend."""
  await store.initialize()

  entry = MemoryEntry(session_id="test", user_id="user", role="user", content="Hello from smoke test!")
  await store.add(entry)

  entries = await store.get_entries("test", user_id="user")
  assert len(entries) >= 1 and entries[0].content == "Hello from smoke test!"

  fetched = await store.get_entry(entry.memory_id)
  assert fetched is not None
  fetched.content = "Updated!"
  await store.update(fetched)
  updated = await store.get_entry(entry.memory_id)
  assert updated is not None and updated.content == "Updated!"

  await store.delete(entry.memory_id)
  assert len(await store.get_entries("test", user_id="user")) == 0

  await store.close()
  return "PASS"


async def backend_smoke_test():
  """Smoke-test all available MemoryStore backends."""
  print("Backend Smoke Test")
  print("=" * 60)

  backends: List[Tuple[str, Any]] = [("InMemoryStore", InMemoryStore())]

  try:
    from definable.memory import SQLiteStore

    backends.append(("SQLiteStore", SQLiteStore("./test_example.db")))
  except ImportError:
    pass

  try:
    from definable.memory import FileStore

    backends.append(("FileStore", FileStore("./test_file_store")))
  except ImportError:
    pass

  results: List[Tuple[str, str]] = []

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

  passed = sum(1 for _, s in results if s == "PASS")
  print(f"\n  {passed}/{len(results)} backends passed")


async def main():
  await protocol_walkthrough()
  await backend_smoke_test()


if __name__ == "__main__":
  asyncio.run(main())
