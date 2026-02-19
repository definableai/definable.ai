"""
Unit tests for SQLiteStore memory store.

Tests CRUD operations using a temp file. No external API calls.
Uses aiosqlite (a lightweight async SQLite driver).

Covers:
  - SQLiteStore creation with db_path
  - upsert_user_memory stores memory
  - get_user_memories returns stored memories
  - get_user_memory retrieves single memory by ID
  - delete_user_memory removes memory
  - clear_user_memories removes all
  - Multi-user isolation (user A can't see user B's memories)
  - Context manager protocol (__aenter__ / __aexit__)
"""

import pytest

from definable.memory.store.sqlite import SQLiteStore
from definable.memory.types import UserMemory


@pytest.fixture
def tmp_db_path(tmp_path):
  """Return a temp SQLite DB path."""
  return str(tmp_path / "test_memory.db")


@pytest.mark.unit
class TestSQLiteStoreCreation:
  """SQLiteStore creation and initialization."""

  @pytest.mark.asyncio
  async def test_create_with_db_path(self, tmp_db_path):
    """SQLiteStore can be created with a db_path."""
    store = SQLiteStore(db_path=tmp_db_path)
    assert store.db_path == tmp_db_path
    assert store._initialized is False

  @pytest.mark.asyncio
  async def test_initialize_creates_table(self, tmp_db_path):
    """initialize() creates the memories table and sets _initialized."""
    store = SQLiteStore(db_path=tmp_db_path)
    await store.initialize()
    assert store._initialized is True
    await store.close()

  @pytest.mark.asyncio
  async def test_double_initialize_is_idempotent(self, tmp_db_path):
    """Calling initialize() twice does not error."""
    store = SQLiteStore(db_path=tmp_db_path)
    await store.initialize()
    await store.initialize()
    assert store._initialized is True
    await store.close()

  @pytest.mark.asyncio
  async def test_close_resets_state(self, tmp_db_path):
    """close() sets _initialized to False."""
    store = SQLiteStore(db_path=tmp_db_path)
    await store.initialize()
    await store.close()
    assert store._initialized is False

  @pytest.mark.asyncio
  async def test_context_manager(self, tmp_db_path):
    """SQLiteStore works as an async context manager."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      assert store._initialized is True
    assert store._initialized is False


@pytest.mark.unit
class TestSQLiteStoreUpsert:
  """SQLiteStore.upsert_user_memory stores memories."""

  @pytest.mark.asyncio
  async def test_upsert_stores_memory(self, tmp_db_path):
    """upsert_user_memory stores a memory that can be retrieved."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="User likes dark mode", user_id="u1")
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 1
      assert results[0].memory == "User likes dark mode"

  @pytest.mark.asyncio
  async def test_upsert_updates_existing(self, tmp_db_path):
    """upsert_user_memory updates an existing memory with the same ID."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="Original text", memory_id="m1", user_id="u1")
      await store.upsert_user_memory(mem)
      mem.memory = "Updated text"
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 1
      assert results[0].memory == "Updated text"

  @pytest.mark.asyncio
  async def test_upsert_preserves_topics(self, tmp_db_path):
    """upsert_user_memory preserves topics as JSON."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="test", user_id="u1", topics=["work", "coding"])
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert results[0].topics == ["work", "coding"]


@pytest.mark.unit
class TestSQLiteStoreGet:
  """SQLiteStore.get_user_memories and get_user_memory retrieval."""

  @pytest.mark.asyncio
  async def test_get_user_memories_returns_all_for_user(self, tmp_db_path):
    """get_user_memories returns all memories for a given user."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      assert len(results) == 3

  @pytest.mark.asyncio
  async def test_get_user_memories_empty_for_unknown_user(self, tmp_db_path):
    """get_user_memories returns empty list for unknown user."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="test", user_id="u1")
      await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u999")
      assert results == []

  @pytest.mark.asyncio
  async def test_get_user_memory_by_id(self, tmp_db_path):
    """get_user_memory retrieves a single memory by ID."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="specific", memory_id="target-id", user_id="u1")
      await store.upsert_user_memory(mem)
      result = await store.get_user_memory("target-id")
      assert result is not None
      assert result.memory == "specific"

  @pytest.mark.asyncio
  async def test_get_user_memory_not_found(self, tmp_db_path):
    """get_user_memory returns None for non-existent ID."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      result = await store.get_user_memory("nonexistent")
      assert result is None

  @pytest.mark.asyncio
  async def test_get_user_memories_ordered_by_updated_at_desc(self, tmp_db_path):
    """Memories are returned ordered by updated_at descending."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      # Insert in order: old first
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1", created_at=float(i), updated_at=float(i))
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1")
      # Most recently updated should come first
      assert results[0].memory == "Fact 2"

  @pytest.mark.asyncio
  async def test_get_user_memories_with_limit(self, tmp_db_path):
    """get_user_memories respects limit parameter."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      for i in range(5):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      results = await store.get_user_memories(user_id="u1", limit=2)
      assert len(results) == 2


@pytest.mark.unit
class TestSQLiteStoreDelete:
  """SQLiteStore.delete_user_memory and clear_user_memories."""

  @pytest.mark.asyncio
  async def test_delete_user_memory(self, tmp_db_path):
    """delete_user_memory removes a specific memory."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      mem = UserMemory(memory="to delete", memory_id="del-id", user_id="u1")
      await store.upsert_user_memory(mem)
      await store.delete_user_memory("del-id")
      result = await store.get_user_memory("del-id")
      assert result is None

  @pytest.mark.asyncio
  async def test_delete_nonexistent_does_not_error(self, tmp_db_path):
    """delete_user_memory with non-existent ID does not raise."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      await store.delete_user_memory("nonexistent")  # Should not raise

  @pytest.mark.asyncio
  async def test_clear_user_memories_removes_all_for_user(self, tmp_db_path):
    """clear_user_memories removes all memories for a specific user."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      for i in range(3):
        mem = UserMemory(memory=f"Fact {i}", user_id="u1")
        await store.upsert_user_memory(mem)
      await store.clear_user_memories(user_id="u1")
      results = await store.get_user_memories(user_id="u1")
      assert results == []

  @pytest.mark.asyncio
  async def test_clear_all_memories(self, tmp_db_path):
    """clear_user_memories with no user_id removes all memories."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
      for uid in ["u1", "u2", "u3"]:
        mem = UserMemory(memory=f"Fact for {uid}", user_id=uid)
        await store.upsert_user_memory(mem)
      await store.clear_user_memories()
      for uid in ["u1", "u2", "u3"]:
        results = await store.get_user_memories(user_id=uid)
        assert results == []


@pytest.mark.unit
class TestSQLiteStoreMultiUserIsolation:
  """Multi-user isolation: user A cannot see user B's memories."""

  @pytest.mark.asyncio
  async def test_user_isolation(self, tmp_db_path):
    """Memories from user A are not visible to user B."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
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
  async def test_clear_one_user_does_not_affect_other(self, tmp_db_path):
    """Clearing user A's memories leaves user B's intact."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
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
  async def test_delete_with_user_scope(self, tmp_db_path):
    """delete_user_memory with user_id scope only deletes if user matches."""
    async with SQLiteStore(db_path=tmp_db_path) as store:
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
