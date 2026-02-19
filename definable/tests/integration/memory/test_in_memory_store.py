"""
Integration tests for InMemoryStore.

Rules:
  - NO MOCKS — tests the real InMemoryStore implementation
  - No external deps required (pure Python, always runs)
  - Tests verify the full MemoryStore protocol with real data flow

Covers:
  - upsert_user_memory() / get_user_memories() round-trip
  - get_user_memory() single retrieval by ID
  - User isolation (memories from different users don't mix)
  - Upsert with same memory_id replaces content
  - delete_user_memory() removes a specific memory
  - clear_user_memories() for a user / for all users
  - Limit parameter on get_user_memories()
  - Ordering by updated_at descending
  - Topic filtering
  - Agent ID filtering
  - Context manager (async with) usage
  - Auto-initialize on first operation
  - Close clears all data
"""

import time
import uuid

import pytest

from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_memory(
  user_id: str,
  content: str,
  topics: list | None = None,
  agent_id: str | None = None,
) -> UserMemory:
  return UserMemory(
    memory=content,
    user_id=user_id,
    topics=topics or [],
    agent_id=agent_id,
  )


# ---------------------------------------------------------------------------
# InMemoryStore integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInMemoryStore:
  """InMemoryStore tests run without any API keys or external services."""

  @pytest.fixture
  def store(self):
    return InMemoryStore()

  @pytest.fixture
  def user_id(self):
    return f"test-user-{uuid.uuid4().hex[:8]}"

  # --- Lifecycle ---

  @pytest.mark.asyncio
  async def test_lifecycle_initialize_and_close(self, store):
    """Store can be initialized and closed without error."""
    await store.initialize()
    await store.close()

  @pytest.mark.asyncio
  async def test_close_is_idempotent(self, store):
    """Closing a store twice should not raise."""
    await store.initialize()
    await store.close()
    await store.close()

  @pytest.mark.asyncio
  async def test_context_manager(self, user_id):
    """async with should initialize and close properly."""
    async with InMemoryStore() as store:
      mem = make_memory(user_id, "Context manager test")
      await store.upsert_user_memory(mem)
      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 1

  # --- Upsert and retrieve ---

  @pytest.mark.asyncio
  async def test_upsert_and_retrieve_memory(self, store, user_id):
    """upsert_user_memory() persists and get_user_memories() retrieves."""
    await store.initialize()
    try:
      mem = make_memory(user_id, "User prefers dark mode")
      await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 1
      assert memories[0].memory == "User prefers dark mode"
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_get_user_memory_by_id(self, store, user_id):
    """get_user_memory() retrieves a specific memory by ID."""
    await store.initialize()
    try:
      mem = make_memory(user_id, "User likes TypeScript")
      await store.upsert_user_memory(mem)

      retrieved = await store.get_user_memory(mem.memory_id, user_id=user_id)
      assert retrieved is not None
      assert retrieved.memory == "User likes TypeScript"
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_get_user_memory_nonexistent_returns_none(self, store, user_id):
    await store.initialize()
    try:
      result = await store.get_user_memory("nonexistent-id-xyz", user_id=user_id)
      assert result is None
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_get_user_memories_empty_user_returns_empty(self, store):
    """Querying a non-existent user returns empty list."""
    await store.initialize()
    try:
      memories = await store.get_user_memories(user_id=f"nonexistent-{uuid.uuid4().hex}")
      assert isinstance(memories, list)
      assert len(memories) == 0
    finally:
      await store.close()

  # --- User isolation ---

  @pytest.mark.asyncio
  async def test_user_isolation(self, store):
    """Memories from user A must not appear in user B queries."""
    await store.initialize()
    try:
      user_a = f"user-a-{uuid.uuid4().hex[:8]}"
      user_b = f"user-b-{uuid.uuid4().hex[:8]}"

      mem_a = make_memory(user_a, "Secret info for user A")
      await store.upsert_user_memory(mem_a)

      memories_b = await store.get_user_memories(user_id=user_b)
      assert all(m.memory != "Secret info for user A" for m in memories_b)
    finally:
      await store.close()

  # --- Multiple memories ---

  @pytest.mark.asyncio
  async def test_multiple_memories_stored_and_retrieved(self, store, user_id):
    """Multiple memories for the same user are all retrievable."""
    await store.initialize()
    try:
      for i in range(3):
        mem = make_memory(user_id, f"Memory number {i}")
        await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 3
    finally:
      await store.close()

  # --- Upsert replaces ---

  @pytest.mark.asyncio
  async def test_upsert_updates_existing_memory(self, store, user_id):
    """Upsert with same memory_id should update, not duplicate."""
    await store.initialize()
    try:
      mem = make_memory(user_id, "Original content")
      await store.upsert_user_memory(mem)

      mem.memory = "Updated content"
      await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 1
      assert memories[0].memory == "Updated content"
    finally:
      await store.close()

  # --- Delete ---

  @pytest.mark.asyncio
  async def test_delete_user_memory_removes_it(self, store, user_id):
    await store.initialize()
    try:
      mem = make_memory(user_id, "To be deleted")
      await store.upsert_user_memory(mem)
      assert len(await store.get_user_memories(user_id=user_id)) == 1

      await store.delete_user_memory(mem.memory_id, user_id=user_id)
      assert len(await store.get_user_memories(user_id=user_id)) == 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_clear_user_memories_removes_all_for_user(self, store, user_id):
    """clear_user_memories() removes all memories for a specific user."""
    await store.initialize()
    try:
      for i in range(3):
        await store.upsert_user_memory(make_memory(user_id, f"Mem {i}"))
      assert len(await store.get_user_memories(user_id=user_id)) == 3

      await store.clear_user_memories(user_id=user_id)
      assert len(await store.get_user_memories(user_id=user_id)) == 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_clear_preserves_other_users(self, store):
    """clear_user_memories(user_id) should not affect other users."""
    await store.initialize()
    try:
      user_a = f"user-a-{uuid.uuid4().hex[:8]}"
      user_b = f"user-b-{uuid.uuid4().hex[:8]}"

      await store.upsert_user_memory(make_memory(user_a, "User A memory"))
      await store.upsert_user_memory(make_memory(user_b, "User B memory"))

      await store.clear_user_memories(user_id=user_a)

      assert len(await store.get_user_memories(user_id=user_a)) == 0
      assert len(await store.get_user_memories(user_id=user_b)) == 1
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_clear_all_memories(self, store):
    """clear_user_memories(user_id=None) should remove everything."""
    await store.initialize()
    try:
      await store.upsert_user_memory(make_memory("user-1", "Mem 1"))
      await store.upsert_user_memory(make_memory("user-2", "Mem 2"))

      await store.clear_user_memories(user_id=None)

      assert len(await store.get_user_memories(user_id="user-1")) == 0
      assert len(await store.get_user_memories(user_id="user-2")) == 0
    finally:
      await store.close()

  # --- Ordering ---

  @pytest.mark.asyncio
  async def test_memories_ordered_by_updated_at_descending(self, store, user_id):
    """get_user_memories() returns most recently updated first."""
    await store.initialize()
    try:
      for i in range(3):
        mem = make_memory(user_id, f"Memory {i}")
        mem.created_at = time.time() + i
        mem.updated_at = time.time() + i
        await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      timestamps = [m.updated_at for m in memories]
      assert timestamps == sorted(timestamps, reverse=True)
    finally:
      await store.close()

  # --- Limit ---

  @pytest.mark.asyncio
  async def test_get_user_memories_respects_limit(self, store, user_id):
    await store.initialize()
    try:
      for i in range(5):
        await store.upsert_user_memory(make_memory(user_id, f"Mem {i}"))

      memories = await store.get_user_memories(user_id=user_id, limit=2)
      assert len(memories) <= 2
    finally:
      await store.close()

  # --- Topic filtering ---

  @pytest.mark.asyncio
  async def test_topic_filtering(self, store, user_id):
    """get_user_memories(topics=[...]) filters by topic tags."""
    await store.initialize()
    try:
      mem_python = make_memory(user_id, "Loves Python", topics=["programming", "python"])
      mem_cooking = make_memory(user_id, "Enjoys cooking", topics=["hobby", "cooking"])
      await store.upsert_user_memory(mem_python)
      await store.upsert_user_memory(mem_cooking)

      results = await store.get_user_memories(user_id=user_id, topics=["programming"])
      assert len(results) >= 1
      assert all(set(m.topics or []).intersection({"programming", "python"}) for m in results)
    finally:
      await store.close()

  # --- Agent ID filtering ---

  @pytest.mark.asyncio
  async def test_agent_id_filtering(self, store, user_id):
    """get_user_memories(agent_id=...) filters by agent."""
    await store.initialize()
    try:
      mem_a = make_memory(user_id, "From agent A", agent_id="agent-a")
      mem_b = make_memory(user_id, "From agent B", agent_id="agent-b")
      await store.upsert_user_memory(mem_a)
      await store.upsert_user_memory(mem_b)

      results = await store.get_user_memories(user_id=user_id, agent_id="agent-a")
      assert len(results) == 1
      assert results[0].memory == "From agent A"
    finally:
      await store.close()

  # --- Auto-initialize ---

  @pytest.mark.asyncio
  async def test_auto_initializes_on_first_operation(self, store, user_id):
    """Store should auto-initialize when used without explicit initialize()."""
    mem = make_memory(user_id, "Auto-init test")
    await store.upsert_user_memory(mem)
    memories = await store.get_user_memories(user_id=user_id)
    assert len(memories) == 1
    await store.close()

  # --- Close clears data ---

  @pytest.mark.asyncio
  async def test_close_clears_data(self, store, user_id):
    """InMemoryStore.close() clears all in-memory data."""
    await store.initialize()
    await store.upsert_user_memory(make_memory(user_id, "Temporary"))
    assert len(await store.get_user_memories(user_id=user_id)) == 1

    await store.close()

    # After close, re-init shows empty store
    await store.initialize()
    assert len(await store.get_user_memories(user_id=user_id)) == 0
    await store.close()

  # --- Returned memory is a copy ---

  @pytest.mark.asyncio
  async def test_returned_memory_is_independent_copy(self, store, user_id):
    """Modifying a returned memory should not affect the stored version."""
    await store.initialize()
    try:
      mem = make_memory(user_id, "Original")
      await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      memories[0].memory = "Mutated externally"

      # Stored version should be unchanged
      fresh = await store.get_user_memories(user_id=user_id)
      assert fresh[0].memory == "Original"
    finally:
      await store.close()

  # --- Memory fields preserved ---

  @pytest.mark.asyncio
  async def test_all_memory_fields_preserved(self, store, user_id):
    """All UserMemory fields should survive the round-trip."""
    await store.initialize()
    try:
      mem = UserMemory(
        memory="Full field test",
        user_id=user_id,
        agent_id="agent-xyz",
        topics=["test", "fields"],
        input="What is testing?",
      )
      await store.upsert_user_memory(mem)

      retrieved = await store.get_user_memory(mem.memory_id)
      assert retrieved is not None
      assert retrieved.memory == "Full field test"
      assert retrieved.user_id == user_id
      assert retrieved.agent_id == "agent-xyz"
      assert retrieved.topics == ["test", "fields"]
      assert retrieved.input == "What is testing?"
      assert retrieved.created_at is not None
      assert retrieved.updated_at is not None
    finally:
      await store.close()
