"""Tests for CortexMemory main API."""

import pytest
from unittest.mock import MagicMock

from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.cortex import CortexMemory


@pytest.fixture
async def cortex(tmp_path):
  config = CortexConfig(
    db_path=str(tmp_path / "cortex_test.db"),
    slow_path_enabled=False,
    enable_learning=False,
    enable_consolidation=False,
    enable_signatures=False,  # Simplify for unit tests
  )
  memory = CortexMemory(config=config)
  await memory._ensure_initialized()
  yield memory
  await memory.close()


@pytest.fixture
async def cortex_full(tmp_path):
  """CortexMemory with all features enabled (no LLM model)."""
  config = CortexConfig(
    db_path=str(tmp_path / "cortex_full.db"),
    slow_path_enabled=False,  # No model = skip slow path
    enable_learning=True,
    enable_consolidation=False,
    enable_signatures=True,
    enable_graph=True,
    enable_tags=True,
  )
  memory = CortexMemory(config=config)
  await memory._ensure_initialized()
  yield memory
  await memory.close()


@pytest.mark.asyncio
class TestCortexMemoryCompatibility:
  """Tests that CortexMemory duck-types the Memory interface."""

  async def test_has_semantic_search(self, cortex):
    assert cortex.has_semantic_search is True

  async def test_enabled(self, cortex):
    assert cortex.enabled is True

  async def test_add_message(self, cortex):
    msg = MagicMock()
    msg.content = "Hello from user"
    msg.role = "user"
    await cortex.add(msg, session_id="s1", user_id="u1")
    entries = await cortex.get_entries("s1", "u1")
    assert len(entries) == 1
    assert entries[0].content == "Hello from user"

  async def test_get_context_messages(self, cortex):
    msg = MagicMock()
    msg.content = "Test message"
    msg.role = "user"
    await cortex.add(msg, session_id="s1")
    messages = await cortex.get_context_messages("s1")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Test message"

  async def test_search(self, cortex):
    msg = MagicMock()
    msg.content = "Python testing frameworks"
    msg.role = "user"
    await cortex.add(msg, session_id="s1")
    results = await cortex.search("testing", session_id="s1")
    assert isinstance(results, list)

  async def test_close(self, tmp_path):
    config = CortexConfig(
      db_path=str(tmp_path / "close_test.db"),
      slow_path_enabled=False,
      enable_learning=False,
      enable_signatures=False,
    )
    mem = CortexMemory(config=config)
    await mem._ensure_initialized()
    await mem.close()
    assert not mem._initialized

  async def test_disabled(self, tmp_path):
    config = CortexConfig(
      db_path=str(tmp_path / "disabled.db"),
      slow_path_enabled=False,
      enable_signatures=False,
    )
    mem = CortexMemory(config=config, enabled=False)
    msg = MagicMock()
    msg.content = "ignored"
    msg.role = "user"
    await mem.add(msg, session_id="s1")
    # Should not initialize or store anything
    assert not mem._initialized


@pytest.mark.asyncio
class TestCortexNativeInterface:
  async def test_remember(self, cortex):
    record_id = await cortex.remember("Important fact", session_id="s1")
    assert record_id
    assert len(record_id) > 8

  async def test_recall(self, cortex):
    await cortex.remember("Python is a programming language", session_id="s1")
    await cortex.remember("Cooking pasta is fun", session_id="s1")
    result = await cortex.recall("programming", session_id="s1")
    assert result is not None
    assert isinstance(result.query, str)

  async def test_update(self, cortex):
    rid = await cortex.remember("Old content", session_id="s1")
    new_record = await cortex.update(rid, "New content", reason="correction")
    assert new_record is not None
    assert new_record.raw_content == "New content"

  async def test_forget(self, cortex):
    rid = await cortex.remember("To forget", session_id="s1")
    assert await cortex.forget(rid, reason="no longer needed")
    # Should not appear in active records
    assert cortex._store is not None
    record = await cortex._store.get_record(rid)
    assert record is not None
    assert record.superseded_by == "forgotten"

  async def test_scratchpad(self, cortex):
    await cortex.set_belief("name", "Test User", session_id="s1")
    state = await cortex.get_state(session_id="s1")
    assert state.get_belief("name") == "Test User"

  async def test_user_model(self, cortex_full):
    model = await cortex_full.get_user_model("u1")
    assert model.user_id == "u1"
    assert model.trait_count == 0

  async def test_as_tools(self, cortex):
    tools = cortex.as_tools()
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert "cortex_remember" in names
    assert "cortex_recall" in names
    assert "cortex_set_belief" in names
    assert "cortex_forget" in names


@pytest.mark.asyncio
class TestCortexLifecycle:
  async def test_context_manager(self, tmp_path):
    config = CortexConfig(
      db_path=str(tmp_path / "ctx.db"),
      slow_path_enabled=False,
      enable_learning=False,
      enable_signatures=False,
    )
    async with CortexMemory(config=config) as mem:
      rid = await mem.remember("context manager test")
      assert rid

  async def test_lazy_init(self, tmp_path):
    config = CortexConfig(
      db_path=str(tmp_path / "lazy.db"),
      slow_path_enabled=False,
      enable_signatures=False,
    )
    mem = CortexMemory(config=config)
    assert not mem._initialized
    await mem.remember("triggers init")
    assert mem._initialized
    await mem.close()

  async def test_import_from_memory(self):
    from definable.memory import CortexMemory as CM

    assert CM is CortexMemory


@pytest.mark.asyncio
class TestCortexWithSignatures:
  async def test_signature_indexing(self, cortex_full):
    await cortex_full.remember("Python programming patterns", session_id="s1")
    await cortex_full.remember("JavaScript web development", session_id="s1")
    await cortex_full.remember("Cooking Italian pasta", session_id="s1")
    result = await cortex_full.recall("programming", session_id="s1")
    assert result is not None
