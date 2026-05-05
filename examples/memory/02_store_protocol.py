"""
MemoryStore protocol walkthrough using InMemoryStore.

Exercises all protocol methods (initialize, add, get_entries, get_entry,
update, delete, delete_session, count, close) with assertions and printed output.
No external dependencies or API keys required.

Usage:
    python definable/examples/memory/02_store_protocol.py
"""

import asyncio

from definable.memory import InMemoryStore
from definable.memory.types import MemoryEntry


async def main():
  store = InMemoryStore()
  await store.initialize()
  print("InMemoryStore initialized\n")

  # ------------------------------------------------------------------
  # 1. Add Entries
  # ------------------------------------------------------------------
  print("=" * 60)
  print("ADD ENTRIES")
  print("=" * 60)

  entry1 = MemoryEntry(
    session_id="s1",
    user_id="alice",
    role="user",
    content="Hello, I'm Alice from San Francisco.",
  )
  await store.add(entry1)
  print(f"  added entry: {entry1.memory_id!r} — role={entry1.role!r} content={entry1.content!r}")

  entry2 = MemoryEntry(
    session_id="s1",
    user_id="alice",
    role="assistant",
    content="Nice to meet you, Alice! How can I help?",
  )
  await store.add(entry2)
  print(f"  added entry: {entry2.memory_id!r} — role={entry2.role!r} content={entry2.content!r}")

  entry3 = MemoryEntry(
    session_id="s1",
    user_id="bob",
    role="user",
    content="Bob prefers dark mode.",
  )
  await store.add(entry3)
  print(f"  added entry: {entry3.memory_id!r} — role={entry3.role!r} content={entry3.content!r}")

  print()

  # ------------------------------------------------------------------
  # 2. Get Entries
  # ------------------------------------------------------------------
  print("=" * 60)
  print("GET ENTRIES")
  print("=" * 60)

  # Get all entries for alice in session s1
  alice_entries = await store.get_entries("s1", user_id="alice")
  assert len(alice_entries) == 2, f"Expected 2 alice entries, got {len(alice_entries)}"
  print(f"  alice entries (s1): {len(alice_entries)}")
  for e in alice_entries:
    print(f"    [{e.memory_id}] {e.role}: {e.content}")

  # Get single entry by ID
  single = await store.get_entry(entry1.memory_id)  # type: ignore[arg-type]
  assert single is not None
  assert single.content == "Hello, I'm Alice from San Francisco."
  print(f"\n  get_entry({entry1.memory_id!r}): {single.content!r}")

  # Get all entries for bob
  bob_entries = await store.get_entries("s1", user_id="bob")
  assert len(bob_entries) == 1
  print(f"  bob entries (s1): {len(bob_entries)}")

  print()

  # ------------------------------------------------------------------
  # 3. Count
  # ------------------------------------------------------------------
  print("=" * 60)
  print("COUNT ENTRIES")
  print("=" * 60)

  alice_count = await store.count("s1", user_id="alice")
  assert alice_count == 2
  print(f"  alice count (s1): {alice_count}")

  bob_count = await store.count("s1", user_id="bob")
  assert bob_count == 1
  print(f"  bob count (s1): {bob_count}")

  print()

  # ------------------------------------------------------------------
  # 4. Update Entry
  # ------------------------------------------------------------------
  print("=" * 60)
  print("UPDATE ENTRY")
  print("=" * 60)

  entry1_fetched = await store.get_entry(entry1.memory_id)  # type: ignore[arg-type]
  assert entry1_fetched is not None
  entry1_fetched.content = "Hello, I'm Alice. I moved to New York!"
  await store.update(entry1_fetched)
  updated = await store.get_entry(entry1.memory_id)  # type: ignore[arg-type]
  assert updated is not None
  assert "New York" in updated.content
  print(f"  updated: {updated.content!r}")

  print()

  # ------------------------------------------------------------------
  # 5. Delete Entry
  # ------------------------------------------------------------------
  print("=" * 60)
  print("DELETE ENTRY")
  print("=" * 60)

  await store.delete(entry2.memory_id)  # type: ignore[arg-type]
  alice_after_delete = await store.get_entries("s1", user_id="alice")
  assert len(alice_after_delete) == 1, f"Expected 1, got {len(alice_after_delete)}"
  print(f"  deleted {entry2.memory_id!r}")
  print(f"  alice entries remaining: {len(alice_after_delete)}")

  print()

  # ------------------------------------------------------------------
  # 6. Delete Session
  # ------------------------------------------------------------------
  print("=" * 60)
  print("DELETE SESSION")
  print("=" * 60)

  await store.delete_session("s1", user_id="alice")
  alice_after_clear = await store.get_entries("s1", user_id="alice")
  assert len(alice_after_clear) == 0
  print(f"  cleared alice's session s1: {len(alice_after_clear)} remaining")

  # Bob's entries should be untouched
  bob_after = await store.get_entries("s1", user_id="bob")
  assert len(bob_after) == 1
  print(f"  bob's entries intact: {len(bob_after)}")

  print()

  # ------------------------------------------------------------------
  # 7. Close
  # ------------------------------------------------------------------
  await store.close()
  print("Store closed. All protocol methods verified successfully!")


if __name__ == "__main__":
  asyncio.run(main())
