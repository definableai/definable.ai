"""Tests for SQLiteStore — same contract as InMemory/File stores."""

import pytest

from definable.memory.store.sqlite import SQLiteStore
from definable.memory.types import MemoryEntry


@pytest.fixture
def store(tmp_path):
  return SQLiteStore(db_path=str(tmp_path / "test_session.db"))


class TestSQLiteStore:
  @pytest.mark.asyncio
  async def test_lifecycle(self, store):
    assert not store._initialized
    await store.initialize()
    assert store._initialized
    await store.close()
    assert not store._initialized

  @pytest.mark.asyncio
  async def test_add_and_get_entries(self, store):
    await store.initialize()
    e1 = MemoryEntry(session_id="s1", role="user", content="Hello", created_at=1.0, updated_at=1.0)
    e2 = MemoryEntry(session_id="s1", role="assistant", content="Hi!", created_at=2.0, updated_at=2.0)
    await store.add(e1)
    await store.add(e2)

    entries = await store.get_entries("s1")
    assert len(entries) == 2
    assert entries[0].content == "Hello"
    assert entries[1].content == "Hi!"
    await store.close()

  @pytest.mark.asyncio
  async def test_get_entries_ordered_by_created_at(self, store):
    await store.initialize()
    e1 = MemoryEntry(session_id="s1", content="first", created_at=3.0, updated_at=3.0)
    e2 = MemoryEntry(session_id="s1", content="second", created_at=1.0, updated_at=1.0)
    await store.add(e1)
    await store.add(e2)

    entries = await store.get_entries("s1")
    assert entries[0].content == "second"
    assert entries[1].content == "first"
    await store.close()

  @pytest.mark.asyncio
  async def test_get_entries_with_limit(self, store):
    await store.initialize()
    for i in range(5):
      await store.add(MemoryEntry(session_id="s1", content=f"msg-{i}", created_at=float(i + 1), updated_at=float(i + 1)))

    entries = await store.get_entries("s1", limit=3)
    assert len(entries) == 3
    assert entries[0].content == "msg-0"
    await store.close()

  @pytest.mark.asyncio
  async def test_get_entries_filters_by_session_and_user(self, store):
    await store.initialize()
    await store.add(MemoryEntry(session_id="s1", user_id="alice", content="a1", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", user_id="bob", content="b1", created_at=2.0, updated_at=2.0))
    await store.add(MemoryEntry(session_id="s2", user_id="alice", content="a2", created_at=3.0, updated_at=3.0))

    entries = await store.get_entries("s1", "alice")
    assert len(entries) == 1
    assert entries[0].content == "a1"
    await store.close()

  @pytest.mark.asyncio
  async def test_get_entry(self, store):
    await store.initialize()
    e = MemoryEntry(memory_id="m1", session_id="s1", content="test", created_at=1.0, updated_at=1.0)
    await store.add(e)

    result = await store.get_entry("m1")
    assert result is not None
    assert result.content == "test"
    assert await store.get_entry("nonexistent") is None
    await store.close()

  @pytest.mark.asyncio
  async def test_update(self, store):
    await store.initialize()
    e = MemoryEntry(memory_id="m1", session_id="s1", content="original", created_at=1.0, updated_at=1.0)
    await store.add(e)

    e.content = "updated"
    await store.update(e)

    result = await store.get_entry("m1")
    assert result is not None
    assert result.content == "updated"
    await store.close()

  @pytest.mark.asyncio
  async def test_delete(self, store):
    await store.initialize()
    e = MemoryEntry(memory_id="m1", session_id="s1", content="test", created_at=1.0, updated_at=1.0)
    await store.add(e)
    await store.delete("m1")

    assert await store.get_entry("m1") is None
    assert await store.count("s1") == 0
    await store.close()

  @pytest.mark.asyncio
  async def test_delete_session(self, store):
    await store.initialize()
    await store.add(MemoryEntry(session_id="s1", content="a", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", content="b", created_at=2.0, updated_at=2.0))
    await store.add(MemoryEntry(session_id="s2", content="c", created_at=3.0, updated_at=3.0))

    await store.delete_session("s1")
    assert await store.count("s1") == 0
    assert await store.count("s2") == 1
    await store.close()

  @pytest.mark.asyncio
  async def test_delete_session_with_user_id(self, store):
    await store.initialize()
    await store.add(MemoryEntry(session_id="s1", user_id="alice", content="a", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", user_id="bob", content="b", created_at=2.0, updated_at=2.0))

    await store.delete_session("s1", user_id="alice")
    assert await store.count("s1", "alice") == 0
    assert await store.count("s1", "bob") == 1
    await store.close()

  @pytest.mark.asyncio
  async def test_count(self, store):
    await store.initialize()
    assert await store.count("s1") == 0
    await store.add(MemoryEntry(session_id="s1", content="a", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", content="b", created_at=2.0, updated_at=2.0))
    assert await store.count("s1") == 2
    await store.close()

  @pytest.mark.asyncio
  async def test_message_data_roundtrip(self, store):
    """Ensure JSON message_data survives SQLite serialization."""
    await store.initialize()
    msg_data = {
      "role": "assistant",
      "content": "Hello",
      "tool_calls": [{"id": "tc1", "function": {"name": "search", "arguments": '{"q":"test"}'}}],
    }
    e = MemoryEntry(memory_id="m1", session_id="s1", content="Hello", message_data=msg_data, created_at=1.0, updated_at=1.0)
    await store.add(e)

    result = await store.get_entry("m1")
    assert result is not None
    assert result.message_data == msg_data
    assert result.message_data["tool_calls"][0]["function"]["name"] == "search"
    await store.close()

  @pytest.mark.asyncio
  async def test_context_manager(self, tmp_path):
    async with SQLiteStore(db_path=str(tmp_path / "cm_test.db")) as store:
      assert store._initialized
      await store.add(MemoryEntry(session_id="s1", content="test", created_at=1.0, updated_at=1.0))
      assert await store.count("s1") == 1
    assert not store._initialized
