"""
Integration tests for SQLiteStore with real SQLite database.

Rules:
  - NO MOCKS — tests real SQLite operations via aiosqlite
  - Each test gets a fresh temp DB file, cleaned up after
  - Tests verify the full MemoryStore protocol with real persistence

Covers:
  - initialize() creates tables and close() releases connection
  - upsert_user_memory() / get_user_memories() round-trip
  - get_user_memory() retrieves a single memory by ID
  - User isolation (memories from different users don't mix)
  - Upsert with same memory_id replaces content
  - delete_user_memory() removes a specific memory
  - clear_user_memories() removes all for a user
  - Limit parameter on get_user_memories()
  - Ordering by updated_at descending
  - Topic filtering
  - Context manager (async with) usage
"""

import time
import uuid

import pytest

from definable.memory.store.sqlite import SQLiteStore
from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_memory(user_id: str, content: str, topics: list | None = None) -> UserMemory:
  return UserMemory(
    memory=content,
    user_id=user_id,
    topics=topics or [],
  )


# ---------------------------------------------------------------------------
# SQLiteStore integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSQLiteStore:
  """SQLiteStore tests run against real SQLite files via aiosqlite."""

  @pytest.fixture
  def store(self, temp_memory_db_file):
    return SQLiteStore(db_path=temp_memory_db_file)

  @pytest.fixture
  def user_id(self):
    return f"test-user-{uuid.uuid4().hex[:8]}"

  # --- Lifecycle ---

  @pytest.mark.asyncio
  async def test_initialize_creates_tables(self, store):
    """Store can be initialized and tables are created without error."""
    await store.initialize()
    await store.close()

  @pytest.mark.asyncio
  async def test_close_is_idempotent(self, store):
    """Closing a store twice should not raise."""
    await store.initialize()
    await store.close()
    await store.close()

  @pytest.mark.asyncio
  async def test_context_manager(self, temp_memory_db_file, user_id):
    """async with should initialize and close properly."""
    async with SQLiteStore(db_path=temp_memory_db_file) as store:
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
      mem = make_memory(user_id, "User works at Acme")
      await store.upsert_user_memory(mem)

      retrieved = await store.get_user_memory(mem.memory_id, user_id=user_id)
      assert retrieved is not None
      assert retrieved.memory == "User works at Acme"
      assert retrieved.user_id == user_id
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
      assert all("python" in (m.topics or []) or "programming" in (m.topics or []) for m in results)
    finally:
      await store.close()

  # --- Persistence across re-open ---

  @pytest.mark.asyncio
  async def test_data_persists_across_reopen(self, temp_memory_db_file, user_id):
    """Data written and closed should survive re-opening the same DB file."""
    store1 = SQLiteStore(db_path=temp_memory_db_file)
    await store1.initialize()
    mem = make_memory(user_id, "Persistent memory")
    await store1.upsert_user_memory(mem)
    await store1.close()

    store2 = SQLiteStore(db_path=temp_memory_db_file)
    await store2.initialize()
    try:
      memories = await store2.get_user_memories(user_id=user_id)
      assert len(memories) == 1
      assert memories[0].memory == "Persistent memory"
    finally:
      await store2.close()

  # --- Auto-initialize ---

  @pytest.mark.asyncio
  async def test_auto_initializes_on_first_operation(self, store, user_id):
    """Store should auto-initialize when used without explicit initialize()."""
    mem = make_memory(user_id, "Auto-init test")
    await store.upsert_user_memory(mem)
    memories = await store.get_user_memories(user_id=user_id)
    assert len(memories) == 1
    await store.close()

  # --- Memory fields preserved ---

  @pytest.mark.asyncio
  async def test_all_memory_fields_preserved(self, store, user_id):
    """All UserMemory fields should survive the round-trip through SQLite."""
    await store.initialize()
    try:
      mem = UserMemory(
        memory="Full field test",
        user_id=user_id,
        agent_id="agent-123",
        topics=["test", "fields"],
        input="What is testing?",
      )
      await store.upsert_user_memory(mem)

      retrieved = await store.get_user_memory(mem.memory_id)
      assert retrieved is not None
      assert retrieved.memory == "Full field test"
      assert retrieved.user_id == user_id
      assert retrieved.agent_id == "agent-123"
      assert retrieved.topics == ["test", "fields"]
      assert retrieved.input == "What is testing?"
      assert retrieved.created_at is not None
      assert retrieved.updated_at is not None
    finally:
      await store.close()
