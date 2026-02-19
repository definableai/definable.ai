"""
Contract tests: Every MemoryStore implementation must satisfy these.

The MemoryStore Protocol (definable.memory.store.base) defines the contract:
  - initialize() / close() lifecycle
  - upsert_user_memory(memory) -> None
  - get_user_memories(user_id, agent_id, topics, limit) -> List[UserMemory]
  - get_user_memory(memory_id, user_id) -> Optional[UserMemory]
  - delete_user_memory(memory_id, user_id) -> None
  - clear_user_memories(user_id) -> None

UserMemory (from definable.memory.types):
  memory, memory_id, topics, user_id, agent_id, input, created_at, updated_at

InMemoryStore always runs (no external deps).
External stores gated by service availability marks.

To add a new MemoryStore: inherit this class and provide a `store` fixture.
"""

import uuid

import pytest

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
# Contract definition
# ---------------------------------------------------------------------------


class MemoryStoreContractTests:
  """
  Abstract contract test suite for all MemoryStore implementations.

  Every concrete MemoryStore must pass ALL tests in this class.
  """

  @pytest.fixture
  def store(self):
    raise NotImplementedError("Subclass must provide a store fixture")

  @pytest.fixture
  def user_id(self) -> str:
    return f"contract-user-{uuid.uuid4().hex[:8]}"

  @pytest.fixture
  def agent_id(self) -> str:
    return f"contract-agent-{uuid.uuid4().hex[:8]}"

  # --- Contract: lifecycle ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_initialize_does_not_raise(self, store):
    await store.initialize()
    await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_close_is_safe_after_initialize(self, store):
    await store.initialize()
    await store.close()  # Must not raise

  # --- Contract: upsert and get ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_upsert_user_memory_does_not_raise(self, store, user_id):
    await store.initialize()
    try:
      mem = make_memory(user_id, "User prefers dark mode")
      await store.upsert_user_memory(mem)  # Should not raise
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_user_memories_returns_list(self, store, user_id):
    await store.initialize()
    try:
      result = await store.get_user_memories(user_id=user_id)
      assert isinstance(result, list)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_stored_memory_is_retrievable(self, store, user_id):
    await store.initialize()
    try:
      mem = make_memory(user_id, "User works at Acme Corp")
      await store.upsert_user_memory(mem)
      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) >= 1
      contents = [m.memory for m in memories]
      assert "User works at Acme Corp" in contents
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_user_memories_empty_user_returns_empty(self, store):
    await store.initialize()
    try:
      result = await store.get_user_memories(user_id=f"nonexistent-{uuid.uuid4().hex}")
      assert isinstance(result, list)
      assert len(result) == 0
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_user_memory_by_id(self, store, user_id):
    await store.initialize()
    try:
      mem = make_memory(user_id, "User likes Python")
      await store.upsert_user_memory(mem)
      retrieved = await store.get_user_memory(mem.memory_id, user_id=user_id)
      assert retrieved is not None
      assert retrieved.memory == "User likes Python"
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_user_memory_nonexistent_returns_none(self, store, user_id):
    await store.initialize()
    try:
      result = await store.get_user_memory("nonexistent-id-xyz", user_id=user_id)
      assert result is None
    finally:
      await store.close()

  # --- Contract: user isolation ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_user_isolation(self, store):
    await store.initialize()
    try:
      user_a = f"iso-user-a-{uuid.uuid4().hex[:6]}"
      user_b = f"iso-user-b-{uuid.uuid4().hex[:6]}"

      mem_a = make_memory(user_a, "Private to user A")
      await store.upsert_user_memory(mem_a)

      memories_b = await store.get_user_memories(user_id=user_b)
      assert not any(m.memory == "Private to user A" for m in memories_b)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_multiple_memories_stored_and_retrieved(self, store, user_id):
    await store.initialize()
    try:
      for i in range(3):
        mem = make_memory(user_id, f"Memory number {i}")
        await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 3
    finally:
      await store.close()

  # --- Contract: upsert replaces existing ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_upsert_updates_existing_memory(self, store, user_id):
    await store.initialize()
    try:
      mem = make_memory(user_id, "Original content")
      await store.upsert_user_memory(mem)

      # Update with same memory_id
      mem.memory = "Updated content"
      await store.upsert_user_memory(mem)

      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) == 1
      assert memories[0].memory == "Updated content"
    finally:
      await store.close()

  # --- Contract: delete ---

  @pytest.mark.contract
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

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_clear_user_memories_removes_all_for_user(self, store, user_id):
    await store.initialize()
    try:
      for i in range(3):
        await store.upsert_user_memory(make_memory(user_id, f"Mem {i}"))
      assert len(await store.get_user_memories(user_id=user_id)) == 3

      await store.clear_user_memories(user_id=user_id)
      assert len(await store.get_user_memories(user_id=user_id)) == 0
    finally:
      await store.close()

  # --- Contract: memory has required fields ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_memory_has_required_fields(self, store, user_id):
    """Retrieved UserMemory must have all expected fields."""
    await store.initialize()
    try:
      mem = make_memory(user_id, "Field check")
      await store.upsert_user_memory(mem)
      memories = await store.get_user_memories(user_id=user_id)
      assert len(memories) >= 1
      retrieved = memories[0]
      assert hasattr(retrieved, "memory_id")
      assert hasattr(retrieved, "memory")
      assert hasattr(retrieved, "user_id")
      assert hasattr(retrieved, "topics")
      assert hasattr(retrieved, "created_at")
      assert hasattr(retrieved, "updated_at")
    finally:
      await store.close()

  # --- Contract: limit ---

  @pytest.mark.contract
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


# ---------------------------------------------------------------------------
# InMemoryStore — always runs, no external deps
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestInMemoryStoreContract(MemoryStoreContractTests):
  """InMemoryStore satisfies the MemoryStore contract."""

  @pytest.fixture
  def store(self):
    from definable.memory.store.in_memory import InMemoryStore

    return InMemoryStore()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_close_without_initialize_does_not_raise(self, store):
    """InMemoryStore should be resilient to close() without initialize()."""
    await store.close()  # Must not raise

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_close_clears_data(self, store):
    """InMemoryStore.close() clears all data."""
    await store.initialize()
    mem = make_memory("user-x", "Temporary data")
    await store.upsert_user_memory(mem)
    assert len(await store.get_user_memories(user_id="user-x")) == 1

    await store.close()
    # After close, re-init should show empty store
    await store.initialize()
    assert len(await store.get_user_memories(user_id="user-x")) == 0
    await store.close()
