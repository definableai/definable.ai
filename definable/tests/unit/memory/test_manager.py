"""Tests for Memory manager — the lego block."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.model.message import Message
from definable.memory.manager import Memory
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import MemoryEntry


@pytest.fixture
def store():
  return InMemoryStore()


@pytest.fixture
def mem(store):
  return Memory(store=store, max_messages=10)


class TestMemory:
  @pytest.mark.asyncio
  async def test_add_message(self, mem):
    msg = Message(role="user", content="Hello")
    await mem.add(msg, session_id="s1")

    entries = await mem.get_entries("s1")
    assert len(entries) == 1
    assert entries[0].role == "user"
    assert entries[0].content == "Hello"

  @pytest.mark.asyncio
  async def test_add_multiple_messages(self, mem):
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    await mem.add(Message(role="assistant", content="Hi!"), session_id="s1")
    await mem.add(Message(role="user", content="How are you?"), session_id="s1")

    entries = await mem.get_entries("s1")
    assert len(entries) == 3

  @pytest.mark.asyncio
  async def test_get_context_messages(self, mem):
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    await mem.add(Message(role="assistant", content="Hi!"), session_id="s1")

    messages = await mem.get_context_messages("s1")
    assert len(messages) == 2
    assert isinstance(messages[0], Message)
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

  @pytest.mark.asyncio
  async def test_get_context_messages_summary_becomes_system(self, store):
    """Summary entries become system messages in context."""
    mem = Memory(store=store)
    await mem._ensure_initialized()

    # Directly add a summary entry to the store
    summary = MemoryEntry(session_id="s1", role="summary", content="User discussed weather.", created_at=1.0, updated_at=1.0)
    await store.add(summary)

    messages = await mem.get_context_messages("s1")
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert "User discussed weather" in messages[0].content

  @pytest.mark.asyncio
  async def test_update_entry(self, mem):
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    entries = await mem.get_entries("s1")
    entry_id = entries[0].memory_id

    await mem.update(entry_id, "Updated content")
    entries = await mem.get_entries("s1")
    assert entries[0].content == "Updated content"

  @pytest.mark.asyncio
  async def test_delete_entry(self, mem):
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    entries = await mem.get_entries("s1")
    entry_id = entries[0].memory_id

    await mem.delete(entry_id)
    entries = await mem.get_entries("s1")
    assert len(entries) == 0

  @pytest.mark.asyncio
  async def test_clear_session(self, mem):
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    await mem.add(Message(role="assistant", content="Hi"), session_id="s1")
    await mem.clear("s1")

    entries = await mem.get_entries("s1")
    assert len(entries) == 0

  @pytest.mark.asyncio
  async def test_enabled_false_skips_add(self):
    mem = Memory(enabled=False)
    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    # Should not have initialized or added anything
    assert not mem._initialized

  @pytest.mark.asyncio
  async def test_default_store_is_in_memory(self):
    mem = Memory()
    await mem._ensure_initialized()
    assert isinstance(mem.store, InMemoryStore)
    await mem.close()

  @pytest.mark.asyncio
  async def test_auto_optimization_triggers(self, store):
    """When count > max_messages and model is set, optimization runs."""
    mock_model = MagicMock()
    response = MagicMock()
    response.content = "Summarized conversation."
    mock_model.ainvoke = AsyncMock(return_value=response)

    mem = Memory(store=store, model=mock_model, max_messages=5, pin_count=1, recent_count=2)

    # Add 6 messages (exceeds max_messages=5)
    for i in range(6):
      await mem.add(Message(role="user" if i % 2 == 0 else "assistant", content=f"Msg {i}"), session_id="s1")

    # After optimization: pin(1) + summary(1) + recent(2) = 4
    entries = await mem.get_entries("s1")
    assert len(entries) < 6
    assert any(e.role == "summary" for e in entries)
    mock_model.ainvoke.assert_called_once()

  @pytest.mark.asyncio
  async def test_no_optimization_without_model(self, store):
    """Without a model, optimization cannot run — entries accumulate."""
    mem = Memory(store=store, max_messages=3)

    for i in range(5):
      await mem.add(Message(role="user", content=f"Msg {i}"), session_id="s1")

    entries = await mem.get_entries("s1")
    assert len(entries) == 5  # No optimization, all entries kept

  @pytest.mark.asyncio
  async def test_message_data_roundtrip(self, mem):
    """Message with tool_calls survives serialization."""
    msg = Message(
      role="assistant",
      content="Let me search for that.",
      tool_calls=[{"id": "tc1", "function": {"name": "search", "arguments": '{"q":"test"}'}}],
    )
    await mem.add(msg, session_id="s1")

    messages = await mem.get_context_messages("s1")
    assert len(messages) == 1
    assert messages[0].tool_calls is not None
    assert messages[0].tool_calls[0]["function"]["name"] == "search"

  @pytest.mark.asyncio
  async def test_lifecycle(self, store):
    mem = Memory(store=store)
    await mem._ensure_initialized()
    assert mem._initialized
    await mem.close()
    assert not mem._initialized

  @pytest.mark.asyncio
  async def test_context_manager(self):
    async with Memory() as mem:
      assert mem._initialized
      await mem.add(Message(role="user", content="test"), session_id="s1")
      entries = await mem.get_entries("s1")
      assert len(entries) == 1
    assert not mem._initialized

  @pytest.mark.asyncio
  async def test_multi_session_isolation(self, mem):
    """Different sessions are isolated."""
    await mem.add(Message(role="user", content="Session 1"), session_id="s1")
    await mem.add(Message(role="user", content="Session 2"), session_id="s2")

    s1_entries = await mem.get_entries("s1")
    s2_entries = await mem.get_entries("s2")
    assert len(s1_entries) == 1
    assert len(s2_entries) == 1
    assert s1_entries[0].content == "Session 1"
    assert s2_entries[0].content == "Session 2"
