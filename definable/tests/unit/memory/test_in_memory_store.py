"""
Unit tests for InMemoryStore memory store.

Tests CRUD operations with a pure in-memory backend. No API calls.
No external dependencies beyond the standard library.

Covers:
  - InMemoryStore creation
  - upsert_user_memory stores memory
  - get_user_memories returns stored memories
  - get_user_memory retrieves single memory by ID
  - delete_user_memory removes memory
  - clear_user_memories removes all
  - Multi-user isolation (user A can't see user B's memories)
  - Context manager protocol (__aenter__ / __aexit__)
  - Returned memories are deep copies (mutations don't leak)
"""

import pytest

from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import UserMemory


@pytest.mark.unit
class TestInMemoryStoreCreation:
  """InMemoryStore creation and initialization."""

  @pytest.mark.asyncio
  async def test_create(self):
    """InMemoryStore can be created."""
    store = InMemoryStore()
    assert store._initialized is False

  @pytest.mark.asyncio
  async def test_initialize(self):
    """initialize() sets _initialized."""
    store = InMemoryStore()
    await store.initialize()
    assert store._initialized is True
    await store.close()

  @pytest.mark.asyncio
  async def test_double_initialize_is_idempotent(self):
    """Calling initialize() twice does not error."""
    store = InMemoryStore()
    await store.initialize()
    await store.initialize()
    assert store._initialized is True
    await store.close()

  @pytest.mark.asyncio
  async def test_close_clears_data(self):
    """close() clears all data and resets _initialized."""
    store = InMemoryStore()
    await store.initialize()
    mem = UserMemory(memory="test", user_id="u1")
    await store.upsert_user_memory(mem)
    await store.close()
    assert store._initialized is False
    assert len(store._memories) == 0

  @pytest.mark.asyncio
  async def test_context_manager(self):
    """InMemoryStore works as an async context manager."""
    async with InMemoryStore() as store:
      assert store._initialized is True
    assert store._initialized is False


@pytest.mark.unit
class TestInMemoryStoreUpsert:
  """InMemoryStore.upsert_user_memory stores memories."""

  @pytest.mark.asyncio
  async def test_upsert_stores_memory(self):
    """upsert_user_memory stores a memory that can be retrieved."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="User likes dark mode", user_id="u1")
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 1
      assert results[0].memory == "User likes dark mode"

  @pytest.mark.asyncio
  async def test_upsert_updates_existing(self):
    """upsert_user_memory updates an existing memory with the same ID."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="Original", memory_id="m1", user_id="u1")
      await store.upsert_user_memory(mem)
      mem.memory = "Updated"
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 1
      assert results[0].memory == "Updated"

  @pytest.mark.asyncio
  async def test_upsert_preserves_topics(self):
    """upsert_user_memory preserves topics."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="test", user_id="u1", topics=["work", "coding"])
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert results[0].topics == ["work", "coding"]

  @pytest.mark.asyncio
  async def test_upsert_stores_deep_copy(self):
    """Stored memory is a deep copy — external mutation doesn't affect store."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="original", user_id="u1")
      await store.upsert_user_memory(mem)
      mem.memory = "mutated externally"
      results = await store.get_user_memories(user_id="u1")
      assert results[0].memory == "original"


@pytest.mark.unit
class TestInMemoryStoreGet:
  """InMemoryStore.get_user_memories and get_user_memory retrieval."""

  @pytest.mark.asyncio
  async def test_get_user_memories_returns_all_for_user(self):
    """get_user_memories returns all memories for a given user."""
    async with InMemoryStore() as store:
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 3

  @pytest.mark.asyncio
  async def test_get_user_memories_empty_for_unknown_user(self):
    """get_user_memories returns empty list for unknown user."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="test", user_id="u1")
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u999")
      assert results == []

  @pytest.mark.asyncio
  async def test_get_user_memory_by_id(self):
    """get_user_memory retrieves a single memory by ID."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="specific", memory_id="target-id", user_id="u1")
      await store.upsert_user_memory(mem)
      result = await store.get_user_memory("target-id")
      assert result is not None
      assert result.memory == "specific"

  @pytest.mark.asyncio
  async def test_get_user_memory_not_found(self):
    """get_user_memory returns None for non-existent ID."""
    async with InMemoryStore() as store:
      result = await store.get_user_memory("nonexistent")
      assert result is None

  @pytest.mark.asyncio
  async def test_get_user_memory_respects_user_scope(self):
    """get_user_memory with user_id returns None if user doesn't match."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="owned by A", memory_id="m1", user_id="userA")
      await store.upsert_user_memory(mem)
      result = await store.get_user_memory("m1", user_id="userB")
      assert result is None

  @pytest.mark.asyncio
  async def test_get_user_memories_returns_deep_copies(self):
    """Returned memories are deep copies — mutation doesn't affect store."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="immutable in store", user_id="u1")
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      results[0].memory = "mutated"
      results2 = await store.get_user_memories(user_id="u1")
      assert results2[0].memory == "immutable in store"

  @pytest.mark.asyncio
  async def test_get_user_memories_ordered_by_updated_at_desc(self):
    """Memories are returned ordered by updated_at descending."""
    async with InMemoryStore() as store:
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1", created_at=float(i), updated_at=float(i))
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      # Most recently updated should come first
      assert results[0].memory == "Fact 2"

  @pytest.mark.asyncio
  async def test_get_user_memories_with_limit(self):
    """get_user_memories respects limit parameter."""
    async with InMemoryStore() as store:
      for i in range(5):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1", limit=2)
      assert len(results) == 2

  @pytest.mark.asyncio
  async def test_get_user_memories_filter_by_topics(self):
    """get_user_memories filters by topics."""
    async with InMemoryStore() as store:
      mem1 = UserMemory(memory="coding fact", user_id="u1", topics=["coding"])
      mem2 = UserMemory(memory="food fact", user_id="u1", topics=["food"])
      mem3 = UserMemory(memory="both", user_id="u1", topics=["coding", "food"])
      await store.upsert_user_memory(mem1)
      await store.upsert_user_memory(mem2)
      await store.upsert_user_memory(mem3)
      results = await store.get_user_memories(user_id="u1", topics=["coding"])
      memories_text = {r.memory for r in results}
      assert "coding fact" in memories_text
      assert "both" in memories_text
      assert "food fact" not in memories_text

  @pytest.mark.asyncio
  async def test_get_user_memories_filter_by_agent_id(self):
    """get_user_memories filters by agent_id."""
    async with InMemoryStore() as store:
      mem1 = UserMemory(memory="from agent A", user_id="u1", agent_id="agentA")
      mem2 = UserMemory(memory="from agent B", user_id="u1", agent_id="agentB")
      await store.upsert_user_memory(mem1)
      await store.upsert_user_memory(mem2)
      results = await store.get_user_memories(user_id="u1", agent_id="agentA")
      assert len(results) == 1
      assert results[0].memory == "from agent A"


@pytest.mark.unit
class TestInMemoryStoreDelete:
  """InMemoryStore.delete_user_memory and clear_user_memories."""

  @pytest.mark.asyncio
  async def test_delete_user_memory(self):
    """delete_user_memory removes a specific memory."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="to delete", memory_id="del-id", user_id="u1")
      await store.upsert_user_memory(mem)
      await store.delete_user_memory("del-id")
      result = await store.get_user_memory("del-id")
      assert result is None

  @pytest.mark.asyncio
  async def test_delete_nonexistent_does_not_error(self):
    """delete_user_memory with non-existent ID does not raise."""
    async with InMemoryStore() as store:
      await store.delete_user_memory("nonexistent")  # Should not raise

  @pytest.mark.asyncio
  async def test_clear_user_memories_removes_all_for_user(self):
    """clear_user_memories removes all memories for a specific user."""
    async with InMemoryStore() as store:
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      await store.clear_user_memories(user_id="u1")
      results = await store.get_user_memories(user_id="u1")
      assert results == []

  @pytest.mark.asyncio
  async def test_clear_all_memories(self):
    """clear_user_memories with no user_id removes all memories."""
    async with InMemoryStore() as store:
      for uid in ["u1", "u2", "u3"]:
        mem = UserMemory(memory=f"Fact for {uid}", user_id=uid)
        await store.upsert_user_memory(mem)
      await store.clear_user_memories()
      for uid in ["u1", "u2", "u3"]:
        results = await store.get_user_memories(user_id=uid)
        assert results == []


@pytest.mark.unit
class TestInMemoryStoreMultiUserIsolation:
  """Multi-user isolation: user A cannot see user B's memories."""

  @pytest.mark.asyncio
  async def test_user_isolation(self):
    """Memories from user A are not visible to user B."""
    async with InMemoryStore() as store:
      mem_a = UserMemory(memory="A's secret", user_id="userA")
      mem_b = UserMemory(memory="B's secret", user_id="userB")
      await store.upsert_user_memory(mem_a)
      await store.upsert_user_memory(mem_b)

      results_a = await store.get_user_memories(user_id="userA")
      results_b = await store.get_user_memories(user_id="userB")

      assert len(results_a) == 1
      assert results_a[0].memory == "A's secret"
      assert len(results_b) == 1
      assert results_b[0].memory == "B's secret"

  @pytest.mark.asyncio
  async def test_clear_one_user_does_not_affect_other(self):
    """Clearing user A's memories leaves user B's intact."""
    async with InMemoryStore() as store:
      mem_a = UserMemory(memory="A's fact", user_id="userA")
      mem_b = UserMemory(memory="B's fact", user_id="userB")
      await store.upsert_user_memory(mem_a)
      await store.upsert_user_memory(mem_b)

      await store.clear_user_memories(user_id="userA")

      results_a = await store.get_user_memories(user_id="userA")
      results_b = await store.get_user_memories(user_id="userB")
      assert results_a == []
      assert len(results_b) == 1

  @pytest.mark.asyncio
  async def test_delete_with_user_scope(self):
    """delete_user_memory with user_id scope only deletes if user matches."""
    async with InMemoryStore() as store:
      mem = UserMemory(memory="owned by A", memory_id="shared-id", user_id="userA")
      await store.upsert_user_memory(mem)

      # Try deleting with wrong user_id
      await store.delete_user_memory("shared-id", user_id="userB")
      result = await store.get_user_memory("shared-id")
      assert result is not None  # Still exists

      # Delete with correct user_id
      await store.delete_user_memory("shared-id", user_id="userA")
      result = await store.get_user_memory("shared-id")
      assert result is None
