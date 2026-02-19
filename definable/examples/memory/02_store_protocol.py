"""
MemoryStore protocol walkthrough using InMemoryStore.

Exercises all protocol methods (initialize, upsert, get, delete, clear, close)
with assertions and printed output.
No external dependencies or API keys required.

Usage:
    python definable/examples/memory/02_store_protocol.py
"""

import asyncio

from definable.memory import InMemoryStore
from definable.memory.types import UserMemory


async def main():
  store = InMemoryStore()
  await store.initialize()
  print("InMemoryStore initialized\n")

  # ------------------------------------------------------------------
  # 1. Upsert Memories
  # ------------------------------------------------------------------
  print("=" * 60)
  print("UPSERT MEMORIES")
  print("=" * 60)

  mem1 = UserMemory(
    memory="Alice lives in San Francisco",
    topics=["location", "personal"],
    user_id="alice",
  )
  await store.upsert_user_memory(mem1)
  print(f"  upserted memory: {mem1.memory_id!r} — {mem1.memory!r}")

  mem2 = UserMemory(
    memory="Alice works as a Python developer",
    topics=["work", "personal"],
    user_id="alice",
  )
  await store.upsert_user_memory(mem2)
  print(f"  upserted memory: {mem2.memory_id!r} — {mem2.memory!r}")

  mem3 = UserMemory(
    memory="Bob prefers dark mode",
    topics=["preferences"],
    user_id="bob",
  )
  await store.upsert_user_memory(mem3)
  print(f"  upserted memory: {mem3.memory_id!r} — {mem3.memory!r}")

  print()

  # ------------------------------------------------------------------
  # 2. Get Memories
  # ------------------------------------------------------------------
  print("=" * 60)
  print("GET MEMORIES")
  print("=" * 60)

  # Get all memories for alice
  alice_mems = await store.get_user_memories(user_id="alice")
  assert len(alice_mems) == 2, f"Expected 2 alice memories, got {len(alice_mems)}"
  print(f"  alice memories: {len(alice_mems)}")
  for m in alice_mems:
    print(f"    [{m.memory_id}] {m.memory} (topics: {m.topics})")

  # Get single memory by ID
  single = await store.get_user_memory(mem1.memory_id, user_id="alice")  # type: ignore[arg-type]
  assert single is not None
  assert single.memory == "Alice lives in San Francisco"
  print(f"\n  get_user_memory({mem1.memory_id!r}): {single.memory!r}")

  # Get all memories for bob
  bob_mems = await store.get_user_memories(user_id="bob")
  assert len(bob_mems) == 1
  print(f"  bob memories: {len(bob_mems)}")

  print()

  # ------------------------------------------------------------------
  # 3. Update (Upsert with existing ID)
  # ------------------------------------------------------------------
  print("=" * 60)
  print("UPDATE MEMORY")
  print("=" * 60)

  # Update mem1 content
  mem1.memory = "Alice lives in New York (moved from San Francisco)"
  mem1.topics = ["location", "personal", "recent"]
  await store.upsert_user_memory(mem1)

  updated = await store.get_user_memory(mem1.memory_id, user_id="alice")  # type: ignore[arg-type]
  assert updated is not None
  assert "New York" in updated.memory
  print(f"  updated: {updated.memory!r}")
  print(f"  topics: {updated.topics}")

  print()

  # ------------------------------------------------------------------
  # 4. Delete Memory
  # ------------------------------------------------------------------
  print("=" * 60)
  print("DELETE MEMORY")
  print("=" * 60)

  await store.delete_user_memory(mem2.memory_id, user_id="alice")  # type: ignore[arg-type]
  alice_after_delete = await store.get_user_memories(user_id="alice")
  assert len(alice_after_delete) == 1, f"Expected 1, got {len(alice_after_delete)}"
  print(f"  deleted {mem2.memory_id!r}")
  print(f"  alice memories remaining: {len(alice_after_delete)}")

  print()

  # ------------------------------------------------------------------
  # 5. Clear All Memories
  # ------------------------------------------------------------------
  print("=" * 60)
  print("CLEAR MEMORIES")
  print("=" * 60)

  await store.clear_user_memories(user_id="alice")
  alice_after_clear = await store.get_user_memories(user_id="alice")
  assert len(alice_after_clear) == 0
  print(f"  cleared alice's memories: {len(alice_after_clear)} remaining")

  # Bob's memories should be untouched
  bob_after = await store.get_user_memories(user_id="bob")
  assert len(bob_after) == 1
  print(f"  bob's memories intact: {len(bob_after)}")

  print()

  # ------------------------------------------------------------------
  # 6. Close
  # ------------------------------------------------------------------
  await store.close()
  print("Store closed. All protocol methods verified successfully!")


if __name__ == "__main__":
  asyncio.run(main())
