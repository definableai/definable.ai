"""
Unit tests for Memory (MemoryManager) orchestrator.

Tests format_memories_for_prompt, tool generation, initialization,
and close behavior. No API calls — all external deps mocked.

Covers:
  - format_memories_for_prompt with empty / single / multiple memories
  - _build_memory_tools with various capability flags
  - Tool schema structure matches OpenAI format
  - Lazy initialization creates InMemoryStore
  - close() resets state
  - acreate_user_memories requires model
"""

import pytest

from definable.memory.manager import Memory, MemoryManager
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# MemoryManager format_memories_for_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryManagerFormat:
  """Memory.format_memories_for_prompt formatting."""

  def test_format_empty(self):
    """format_memories_for_prompt returns empty string for no memories."""
    mgr = Memory()
    assert mgr.format_memories_for_prompt([]) == ""

  def test_format_single(self):
    """format_memories_for_prompt formats a single memory correctly."""
    mgr = Memory()
    mem = UserMemory(memory="User likes Python", memory_id="m1", topics=["lang"])
    result = mgr.format_memories_for_prompt([mem])
    assert "[m1]" in result
    assert "[lang]" in result
    assert "User likes Python" in result

  def test_format_multiple(self):
    """format_memories_for_prompt numbers memories sequentially."""
    mgr = Memory()
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
    mgr = Memory()
    mem = UserMemory(memory="No topics here", memory_id="m1", topics=[])
    result = mgr.format_memories_for_prompt([mem])
    assert "[m1]" in result
    assert "No topics here" in result

  def test_format_multiple_topics(self):
    """format_memories_for_prompt joins multiple topics with comma."""
    mgr = Memory()
    mem = UserMemory(memory="Multi-topic", memory_id="m1", topics=["work", "preferences"])
    result = mgr.format_memories_for_prompt([mem])
    assert "[work, preferences]" in result


# ---------------------------------------------------------------------------
# MemoryManager tool generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryManagerTools:
  """Memory._build_memory_tools generates correct tool definitions."""

  def test_default_tools(self):
    """Default Memory generates add_memory and update_memory tools."""
    mgr = Memory(store=InMemoryStore())
    funcs, tool_map = mgr._build_memory_tools()
    names = {f["name"] for f in funcs}
    assert "add_memory" in names
    assert "update_memory" in names
    assert "delete_memory" not in names  # Off by default
    assert "clear_memory" not in names  # Off by default

  def test_all_tools(self):
    """Memory with all capabilities generates all 4 tools."""
    mgr = Memory(store=InMemoryStore(), delete_memories=True, clear_memories=True)
    funcs, tool_map = mgr._build_memory_tools()
    names = {f["name"] for f in funcs}
    assert names == {"add_memory", "update_memory", "delete_memory", "clear_memory"}

  def test_no_tools(self):
    """Memory with all capabilities disabled generates no tools."""
    mgr = Memory(store=InMemoryStore(), add_memories=False, update_memories=False)
    funcs, tool_map = mgr._build_memory_tools()
    assert len(funcs) == 0
    assert len(tool_map) == 0

  def test_tool_schema_structure(self):
    """Generated tool schemas have correct OpenAI-compatible structure."""
    mgr = Memory(store=InMemoryStore())
    funcs, _ = mgr._build_memory_tools()
    for f in funcs:
      assert "name" in f
      assert "description" in f
      assert "parameters" in f
      assert f["parameters"]["type"] == "object"
      assert "properties" in f["parameters"]

  def test_add_memory_schema_has_required_fields(self):
    """add_memory tool requires 'memory' parameter."""
    mgr = Memory(store=InMemoryStore())
    funcs, _ = mgr._build_memory_tools()
    add_func = next(f for f in funcs if f["name"] == "add_memory")
    assert "memory" in add_func["parameters"]["required"]

  def test_update_memory_schema_has_required_fields(self):
    """update_memory tool requires 'memory_id' and 'memory' parameters."""
    mgr = Memory(store=InMemoryStore())
    funcs, _ = mgr._build_memory_tools()
    update_func = next(f for f in funcs if f["name"] == "update_memory")
    assert "memory_id" in update_func["parameters"]["required"]
    assert "memory" in update_func["parameters"]["required"]

  def test_tool_map_has_callables(self):
    """Tool map values are callable."""
    mgr = Memory(store=InMemoryStore())
    _, tool_map = mgr._build_memory_tools()
    for name, fn in tool_map.items():
      assert callable(fn)

  def test_only_delete_tool(self):
    """Memory with only delete_memories=True generates only delete_memory tool."""
    mgr = Memory(store=InMemoryStore(), add_memories=False, update_memories=False, delete_memories=True)
    funcs, tool_map = mgr._build_memory_tools()
    names = {f["name"] for f in funcs}
    assert names == {"delete_memory"}


# ---------------------------------------------------------------------------
# MemoryManager initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryManagerInit:
  """Memory initialization and lifecycle."""

  @pytest.mark.asyncio
  async def test_lazy_init_creates_store(self):
    """_ensure_initialized creates InMemoryStore when store is None."""
    mgr = Memory()
    assert mgr.store is None
    await mgr._ensure_initialized()
    assert mgr.store is not None  # type: ignore[unreachable]  # mypy can't track mutation through async
    assert mgr._initialized is True  # type: ignore[unreachable]
    await mgr.close()  # type: ignore[unreachable]

  @pytest.mark.asyncio
  async def test_close_resets_state(self):
    """close() resets _initialized flag."""
    mgr = Memory()
    await mgr._ensure_initialized()
    assert mgr._initialized is True
    await mgr.close()
    assert mgr._initialized is False

  @pytest.mark.asyncio
  async def test_requires_model_for_create(self):
    """acreate_user_memories raises ValueError when no model is set."""
    mgr = Memory()
    with pytest.raises(ValueError, match="requires a model"):
      await mgr.acreate_user_memories(message="test")
    await mgr.close()

  @pytest.mark.asyncio
  async def test_double_init_is_idempotent(self):
    """Calling _ensure_initialized twice does not break."""
    mgr = Memory()
    await mgr._ensure_initialized()
    store1 = mgr.store
    await mgr._ensure_initialized()
    assert mgr.store is store1  # same store
    await mgr.close()

  def test_default_capability_flags(self):
    """Default Memory has add and update enabled, delete and clear disabled."""
    mgr = Memory()
    assert mgr.add_memories is True
    assert mgr.update_memories is True
    assert mgr.delete_memories is False
    assert mgr.clear_memories is False

  def test_default_trigger_is_always(self):
    """Default trigger is 'always'."""
    mgr = Memory()
    assert mgr.trigger == "always"

  def test_default_update_on_run_is_true(self):
    """Default update_on_run is True."""
    mgr = Memory()
    assert mgr.update_on_run is True

  def test_backward_compat_alias(self):
    """MemoryManager is an alias for Memory."""
    assert MemoryManager is Memory

  @pytest.mark.asyncio
  async def test_acreate_returns_no_input_message(self):
    """acreate_user_memories with no input returns early message."""
    mgr = Memory(model=object())  # dummy model, won't be called
    result = await mgr.acreate_user_memories()
    assert result == "No input provided."
    await mgr.close()

  @pytest.mark.asyncio
  async def test_acreate_returns_empty_content_message(self):
    """acreate_user_memories with whitespace-only message returns early."""
    mgr = Memory(model=object())  # dummy model, won't be called
    result = await mgr.acreate_user_memories(message="   ")
    assert result == "No user content to process."
    await mgr.close()
