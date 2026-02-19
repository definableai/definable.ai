"""
Smoke-test all MemoryStore backends.

Runs a minimal round-trip (upsert, retrieve, delete) against each backend.
Backends whose dependencies are not installed or whose services are not
reachable are skipped gracefully.

Environment variables for optional backends:
    MEMORY_POSTGRES_URL   — e.g. postgresql://user:pass@localhost/dbname

Usage:
    python definable/examples/memory/03_store_backends.py
"""

import asyncio
import os
from typing import Any, List, Tuple

from definable.memory import InMemoryStore
from definable.memory.types import UserMemory


async def test_store(name: str, store: Any) -> str:
  """Run a minimal round-trip against *store*. Returns a status string."""

  await store.initialize()

  # --- Upsert a memory ---
  mem = UserMemory(
    memory="Hello from the smoke test!",
    topics=["test"],
    user_id="example-user",
  )
  await store.upsert_user_memory(mem)

  # --- Retrieve ---
  memories = await store.get_user_memories(user_id="example-user")
  assert len(memories) >= 1, f"{name}: expected at least 1 memory"
  assert memories[0].memory == "Hello from the smoke test!"

  # --- Get single ---
  single = await store.get_user_memory(mem.memory_id, user_id="example-user")
  assert single is not None, f"{name}: get_user_memory returned None"

  # --- Delete ---
  await store.delete_user_memory(mem.memory_id, user_id="example-user")
  after_delete = await store.get_user_memories(user_id="example-user")
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

  # 3. PostgresStore
  pg_url = os.environ.get("MEMORY_POSTGRES_URL")
  if pg_url:
    try:
      from definable.memory import PostgresStore

      backends.append(("PostgresStore", PostgresStore(db_url=pg_url)))
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

  # Cleanup SQLite test file
  if os.path.exists("./test_example.db"):
    os.remove("./test_example.db")

  # Summary table
  passed = sum(1 for _, s in results if s == "PASS")
  print(f"\n{'=' * 50}")
  print(f"Results: {passed}/{len(results)} backends passed")
  print(f"{'=' * 50}")
  for name, status in results:
    print(f"  {name:<25s} {status}")


if __name__ == "__main__":
  asyncio.run(main())
