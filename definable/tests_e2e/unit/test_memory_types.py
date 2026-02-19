"""Unit tests for the new memory module data types and MemoryManager."""

import time

import pytest

from definable.memory.manager import MemoryManager
from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# UserMemory dataclass
# ---------------------------------------------------------------------------


class TestUserMemory:
  def test_auto_id(self):
    """UserMemory auto-generates a UUID if memory_id not provided."""
    mem = UserMemory(memory="User likes dark mode")
    assert mem.memory_id is not None
    assert len(mem.memory_id) == 36  # UUID format

  def test_explicit_id(self):
    """UserMemory preserves explicitly provided memory_id."""
    mem = UserMemory(memory="test", memory_id="custom-id")
    assert mem.memory_id == "custom-id"

  def test_auto_timestamps(self):
    """UserMemory auto-sets created_at and updated_at."""
    before = time.time()
    mem = UserMemory(memory="test")
    after = time.time()
    assert before <= mem.created_at <= after  # type: ignore[operator]
    assert before <= mem.updated_at <= after  # type: ignore[operator]

  def test_explicit_timestamps(self):
    """UserMemory preserves explicit timestamps."""
    mem = UserMemory(memory="test", created_at=100.0, updated_at=200.0)
    assert mem.created_at == 100.0
    assert mem.updated_at == 200.0

  def test_topics_default_empty(self):
    """UserMemory defaults topics to empty list."""
    mem = UserMemory(memory="test")
    assert mem.topics == []

  def test_topics_none_becomes_empty(self):
    """UserMemory converts None topics to empty list."""
    mem = UserMemory(memory="test", topics=None)
    assert mem.topics == []

  def test_to_dict(self):
    """UserMemory.to_dict() produces correct structure."""
    mem = UserMemory(
      memory="User prefers Python",
      memory_id="test-id",
      topics=["preferences"],
      user_id="u1",
      agent_id="a1",
      input="I prefer Python",
      created_at=1000.0,
      updated_at=2000.0,
    )
    d = mem.to_dict()
    assert d["memory_id"] == "test-id"
    assert d["memory"] == "User prefers Python"
    assert d["topics"] == ["preferences"]
    assert d["user_id"] == "u1"
    assert d["agent_id"] == "a1"
    assert d["input"] == "I prefer Python"
    assert d["created_at"] == 1000.0
    assert d["updated_at"] == 2000.0

  def test_from_dict(self):
    """UserMemory.from_dict() round-trips correctly."""
    original = UserMemory(
      memory="User lives in NYC",
      topics=["location"],
      user_id="u1",
    )
    d = original.to_dict()
    restored = UserMemory.from_dict(d)
    assert restored.memory == original.memory
    assert restored.memory_id == original.memory_id
    assert restored.topics == original.topics
    assert restored.user_id == original.user_id

  def test_from_dict_minimal(self):
    """UserMemory.from_dict() works with minimal data."""
    d = {"memory": "Just the memory text"}
    mem = UserMemory.from_dict(d)
    assert mem.memory == "Just the memory text"
    assert mem.memory_id is not None  # auto-generated


# ---------------------------------------------------------------------------
# MemoryManager format_memories_for_prompt
# ---------------------------------------------------------------------------


class TestMemoryManagerFormat:
  def test_format_empty(self):
    """format_memories_for_prompt returns empty string for no memories."""
    mgr = MemoryManager()
    assert mgr.format_memories_for_prompt([]) == ""

  def test_format_single(self):
    """format_memories_for_prompt formats a single memory correctly."""
    mgr = MemoryManager()
    mem = UserMemory(memory="User likes Python", memory_id="m1", topics=["lang"])
    result = mgr.format_memories_for_prompt([mem])
    assert "[m1]" in result
    assert "[lang]" in result
    assert "User likes Python" in result

  def test_format_multiple(self):
    """format_memories_for_prompt numbers memories sequentially."""
    mgr = MemoryManager()
    memories = [
      UserMemory(memory="Fact A", memory_id="m1"),
      UserMemory(memory="Fact B", memory_id="m2"),
      UserMemory(memory="Fact C", memory_id="m3"),
    ]
    result = mgr.format_memories_for_prompt(memories)
    assert "1. [m1]" in result
    assert "2. [m2]" in result
    assert "3. [m3]" in result

  def test_format_no_topics(self):
    """format_memories_for_prompt handles memories without topics."""
    mgr = MemoryManager()
    mem = UserMemory(memory="No topics here", memory_id="m1", topics=[])
    result = mgr.format_memories_for_prompt([mem])
    assert "[m1]" in result
    assert "No topics here" in result


# ---------------------------------------------------------------------------
# MemoryManager tool generation
# ---------------------------------------------------------------------------


class TestMemoryManagerTools:
  def test_default_tools(self):
    """Default MemoryManager generates add_memory and update_memory tools."""
    from definable.memory.store.in_memory import InMemoryStore

    mgr = MemoryManager(store=InMemoryStore())
    funcs, tool_map = mgr._build_memory_tools()
    names = {f["name"] for f in funcs}
    assert "add_memory" in names
    assert "update_memory" in names
    assert "delete_memory" not in names  # Off by default
    assert "clear_memory" not in names  # Off by default

  def test_all_tools(self):
    """MemoryManager with all capabilities generates all 4 tools."""
    from definable.memory.store.in_memory import InMemoryStore

    mgr = MemoryManager(store=InMemoryStore(), delete_memories=True, clear_memories=True)
    funcs, tool_map = mgr._build_memory_tools()
    names = {f["name"] for f in funcs}
    assert names == {"add_memory", "update_memory", "delete_memory", "clear_memory"}

  def test_no_tools(self):
    """MemoryManager with all capabilities disabled generates no tools."""
    from definable.memory.store.in_memory import InMemoryStore

    mgr = MemoryManager(store=InMemoryStore(), add_memories=False, update_memories=False)
    funcs, tool_map = mgr._build_memory_tools()
    assert len(funcs) == 0
    assert len(tool_map) == 0

  def test_tool_schema_structure(self):
    """Generated tool schemas have correct OpenAI-compatible structure."""
    from definable.memory.store.in_memory import InMemoryStore

    mgr = MemoryManager(store=InMemoryStore())
    funcs, _ = mgr._build_memory_tools()
    for f in funcs:
      assert "name" in f
      assert "description" in f
      assert "parameters" in f
      assert f["parameters"]["type"] == "object"
      assert "properties" in f["parameters"]


# ---------------------------------------------------------------------------
# MemoryManager initialization
# ---------------------------------------------------------------------------


class TestMemoryManagerInit:
  @pytest.mark.asyncio
  async def test_lazy_init_creates_store(self):
    """_ensure_initialized creates InMemoryStore when store is None."""
    mgr = MemoryManager()
    assert mgr.store is None
    await mgr._ensure_initialized()
    assert mgr.store is not None
    assert mgr._initialized is True  # type: ignore[unreachable]
    await mgr.close()

  @pytest.mark.asyncio
  async def test_close_resets_state(self):
    """close() resets _initialized flag."""
    mgr = MemoryManager()
    await mgr._ensure_initialized()
    assert mgr._initialized is True
    await mgr.close()
    assert mgr._initialized is False

  @pytest.mark.asyncio
  async def test_requires_model_for_create(self):
    """acreate_user_memories raises ValueError when no model is set."""
    mgr = MemoryManager()
    with pytest.raises(ValueError, match="requires a model"):
      await mgr.acreate_user_memories(message="test")
    await mgr.close()
