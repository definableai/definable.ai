"""
Contract tests for the new MemoryStore protocol (7 methods).

Tests are parameterized across all store implementations:
  - InMemoryStore (always runs)
  - SQLiteStore (requires aiosqlite)

Every store must satisfy these contracts to be a valid MemoryStore backend.
"""

import os
import tempfile

import pytest

from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_memory(user_id: str = "u1", memory: str = "Test fact", topics: list = None, agent_id: str = "a1") -> UserMemory:  # type: ignore[assignment]
  return UserMemory(
    memory=memory,
    user_id=user_id,
    agent_id=agent_id,
    topics=topics or [],
  )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def in_memory_store():
  from definable.memory.store.in_memory import InMemoryStore

  store = InMemoryStore()
  await store.initialize()
  yield store
  await store.close()


@pytest.fixture
async def sqlite_store():
  from definable.memory.store.sqlite import SQLiteStore

  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name
  store = SQLiteStore(db_path=db_path)
  await store.initialize()
  yield store
  await store.close()
  os.unlink(db_path)


@pytest.fixture(params=["in_memory", "sqlite"])
async def store(request, in_memory_store, sqlite_store):
  if request.param == "in_memory":
    return in_memory_store
  elif request.param == "sqlite":
    return sqlite_store


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMemoryStoreContracts:
  """Every MemoryStore implementation must pass all of these."""

  @pytest.mark.asyncio
  async def test_upsert_and_get(self, store):
    """Round-trip: upsert a memory then retrieve it by ID."""
    mem = make_memory(memory="User prefers dark mode")
    await store.upsert_user_memory(mem)

    retrieved = await store.get_user_memory(mem.memory_id)
    assert retrieved is not None
    assert retrieved.memory == "User prefers dark mode"
    assert retrieved.memory_id == mem.memory_id
    assert retrieved.user_id == "u1"

  @pytest.mark.asyncio
  async def test_upsert_updates_existing(self, store):
    """Upserting with same ID updates the memory."""
    mem = make_memory(memory="Original fact", user_id="u1")
    await store.upsert_user_memory(mem)

    mem.memory = "Updated fact"
    await store.upsert_user_memory(mem)

    retrieved = await store.get_user_memory(mem.memory_id)
    assert retrieved is not None
    assert retrieved.memory == "Updated fact"

  @pytest.mark.asyncio
  async def test_get_nonexistent(self, store):
    """Getting a nonexistent memory returns None."""
    result = await store.get_user_memory("nonexistent-id")
    assert result is None

  @pytest.mark.asyncio
  async def test_get_by_user_id(self, store):
    """get_user_memories filters by user_id."""
    await store.upsert_user_memory(make_memory(user_id="alice", memory="Alice fact"))
    await store.upsert_user_memory(make_memory(user_id="bob", memory="Bob fact"))

    alice_mems = await store.get_user_memories(user_id="alice")
    assert len(alice_mems) == 1
    assert alice_mems[0].memory == "Alice fact"

    bob_mems = await store.get_user_memories(user_id="bob")
    assert len(bob_mems) == 1
    assert bob_mems[0].memory == "Bob fact"

  @pytest.mark.asyncio
  async def test_get_by_agent_id(self, store):
    """get_user_memories filters by agent_id."""
    await store.upsert_user_memory(make_memory(user_id="u1", agent_id="agent-a", memory="Fact from A"))
    await store.upsert_user_memory(make_memory(user_id="u1", agent_id="agent-b", memory="Fact from B"))

    a_mems = await store.get_user_memories(user_id="u1", agent_id="agent-a")
    assert len(a_mems) == 1
    assert a_mems[0].memory == "Fact from A"

  @pytest.mark.asyncio
  async def test_get_by_topics(self, store):
    """get_user_memories filters by topics (any match)."""
    await store.upsert_user_memory(make_memory(memory="Work fact", topics=["work"]))
    await store.upsert_user_memory(make_memory(memory="Hobby fact", topics=["hobbies"]))
    await store.upsert_user_memory(make_memory(memory="Both", topics=["work", "hobbies"]))

    work_mems = await store.get_user_memories(user_id="u1", topics=["work"])
    assert len(work_mems) >= 2  # "Work fact" + "Both"
    work_texts = {m.memory for m in work_mems}
    assert "Work fact" in work_texts
    assert "Both" in work_texts

  @pytest.mark.asyncio
  async def test_get_with_limit(self, store):
    """get_user_memories respects limit parameter."""
    for i in range(5):
      await store.upsert_user_memory(make_memory(memory=f"Fact {i}"))

    mems = await store.get_user_memories(user_id="u1", limit=3)
    assert len(mems) == 3

  @pytest.mark.asyncio
  async def test_get_ordered_by_updated_at(self, store):
    """get_user_memories returns results ordered by updated_at descending."""
    import time

    m1 = make_memory(memory="Oldest")
    m1.updated_at = 1000.0
    await store.upsert_user_memory(m1)

    time.sleep(0.01)

    m2 = make_memory(memory="Newest")
    m2.updated_at = 2000.0
    await store.upsert_user_memory(m2)

    mems = await store.get_user_memories(user_id="u1")
    assert len(mems) >= 2
    assert mems[0].memory == "Newest"

  @pytest.mark.asyncio
  async def test_delete(self, store):
    """delete_user_memory removes a single memory."""
    mem = make_memory(memory="To be deleted")
    await store.upsert_user_memory(mem)
    assert await store.get_user_memory(mem.memory_id) is not None

    await store.delete_user_memory(mem.memory_id)
    assert await store.get_user_memory(mem.memory_id) is None

  @pytest.mark.asyncio
  async def test_delete_with_user_scope(self, store):
    """delete_user_memory with user_id only deletes if user matches."""
    mem = make_memory(user_id="alice", memory="Alice secret")
    await store.upsert_user_memory(mem)

    # Bob can't delete Alice's memory
    await store.delete_user_memory(mem.memory_id, user_id="bob")
    assert await store.get_user_memory(mem.memory_id) is not None

    # Alice can delete her own memory
    await store.delete_user_memory(mem.memory_id, user_id="alice")
    assert await store.get_user_memory(mem.memory_id) is None

  @pytest.mark.asyncio
  async def test_clear(self, store):
    """clear_user_memories removes all memories for a user."""
    await store.upsert_user_memory(make_memory(user_id="alice", memory="Fact 1"))
    await store.upsert_user_memory(make_memory(user_id="alice", memory="Fact 2"))
    await store.upsert_user_memory(make_memory(user_id="bob", memory="Bob fact"))

    await store.clear_user_memories(user_id="alice")

    alice_mems = await store.get_user_memories(user_id="alice")
    assert len(alice_mems) == 0

    bob_mems = await store.get_user_memories(user_id="bob")
    assert len(bob_mems) == 1

  @pytest.mark.asyncio
  async def test_clear_all(self, store):
    """clear_user_memories with no user_id removes everything."""
    await store.upsert_user_memory(make_memory(user_id="alice", memory="A"))
    await store.upsert_user_memory(make_memory(user_id="bob", memory="B"))

    await store.clear_user_memories()

    all_mems = await store.get_user_memories()
    assert len(all_mems) == 0

  @pytest.mark.asyncio
  async def test_get_with_user_scope(self, store):
    """get_user_memory with user_id returns None if user doesn't match."""
    mem = make_memory(user_id="alice", memory="Alice only")
    await store.upsert_user_memory(mem)

    assert await store.get_user_memory(mem.memory_id, user_id="alice") is not None
    assert await store.get_user_memory(mem.memory_id, user_id="bob") is None

  @pytest.mark.asyncio
  async def test_initialize_close_lifecycle(self, store):
    """Stores can be initialized and closed without error."""
    # Already initialized by fixture
    await store.close()
    # Re-initialize should work
    await store.initialize()
