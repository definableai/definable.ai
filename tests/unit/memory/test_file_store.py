"""Tests for FileStore — JSONL file-based persistence."""

import json

import pytest

from definable.memory.store.file import FileStore
from definable.memory.types import MemoryEntry


@pytest.fixture
def store(tmp_path):
  return FileStore(base_dir=str(tmp_path / "memory"))


class TestFileStore:
  @pytest.mark.asyncio
  async def test_lifecycle(self, store, tmp_path):
    assert not store._initialized
    await store.initialize()
    assert store._initialized
    assert store.base_dir.exists()
    await store.close()
    assert not store._initialized

  @pytest.mark.asyncio
  async def test_add_creates_jsonl_file(self, store, tmp_path):
    e = MemoryEntry(session_id="s1", role="user", content="Hello", created_at=1.0, updated_at=1.0)
    await store.add(e)

    jsonl_path = store.base_dir / "s1" / "default.jsonl"
    assert jsonl_path.exists()

    with open(jsonl_path, "r") as f:
      lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["content"] == "Hello"

  @pytest.mark.asyncio
  async def test_add_and_get_entries(self, store):
    e1 = MemoryEntry(session_id="s1", role="user", content="Hello", created_at=1.0, updated_at=1.0)
    e2 = MemoryEntry(session_id="s1", role="assistant", content="Hi!", created_at=2.0, updated_at=2.0)
    await store.add(e1)
    await store.add(e2)

    entries = await store.get_entries("s1")
    assert len(entries) == 2
    assert entries[0].content == "Hello"
    assert entries[1].content == "Hi!"

  @pytest.mark.asyncio
  async def test_get_entries_ordered_by_created_at(self, store):
    e1 = MemoryEntry(session_id="s1", content="first", created_at=3.0, updated_at=3.0)
    e2 = MemoryEntry(session_id="s1", content="second", created_at=1.0, updated_at=1.0)
    await store.add(e1)
    await store.add(e2)

    entries = await store.get_entries("s1")
    assert entries[0].content == "second"
    assert entries[1].content == "first"

  @pytest.mark.asyncio
  async def test_get_entries_with_limit(self, store):
    for i in range(5):
      await store.add(MemoryEntry(session_id="s1", content=f"msg-{i}", created_at=float(i), updated_at=float(i)))

    entries = await store.get_entries("s1", limit=3)
    assert len(entries) == 3

  @pytest.mark.asyncio
  async def test_multi_user_files(self, store):
    await store.add(MemoryEntry(session_id="s1", user_id="alice", content="from alice", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", user_id="bob", content="from bob", created_at=2.0, updated_at=2.0))

    alice_path = store.base_dir / "s1" / "alice.jsonl"
    bob_path = store.base_dir / "s1" / "bob.jsonl"
    assert alice_path.exists()
    assert bob_path.exists()

    alice_entries = await store.get_entries("s1", "alice")
    assert len(alice_entries) == 1
    assert alice_entries[0].content == "from alice"

  @pytest.mark.asyncio
  async def test_get_entry(self, store):
    e = MemoryEntry(memory_id="m1", session_id="s1", content="test", created_at=1.0, updated_at=1.0)
    await store.add(e)

    result = await store.get_entry("m1")
    assert result is not None
    assert result.content == "test"
    assert await store.get_entry("nonexistent") is None

  @pytest.mark.asyncio
  async def test_update_rewrites_file(self, store):
    e = MemoryEntry(memory_id="m1", session_id="s1", content="original", created_at=1.0, updated_at=1.0)
    await store.add(e)

    e.content = "updated"
    await store.update(e)

    result = await store.get_entry("m1")
    assert result is not None
    assert result.content == "updated"

    # Verify file was rewritten correctly
    entries = await store.get_entries("s1")
    assert len(entries) == 1

  @pytest.mark.asyncio
  async def test_delete(self, store):
    e = MemoryEntry(memory_id="m1", session_id="s1", content="test", created_at=1.0, updated_at=1.0)
    await store.add(e)
    await store.delete("m1")

    assert await store.get_entry("m1") is None
    assert await store.count("s1") == 0

  @pytest.mark.asyncio
  async def test_delete_session_removes_directory(self, store):
    await store.add(MemoryEntry(session_id="s1", content="a", created_at=1.0, updated_at=1.0))
    session_dir = store.base_dir / "s1"
    assert session_dir.exists()

    await store.delete_session("s1")
    assert not session_dir.exists()
    assert await store.count("s1") == 0

  @pytest.mark.asyncio
  async def test_delete_session_with_user_id(self, store):
    await store.add(MemoryEntry(session_id="s1", user_id="alice", content="a", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", user_id="bob", content="b", created_at=2.0, updated_at=2.0))

    await store.delete_session("s1", user_id="alice")
    assert await store.count("s1", "alice") == 0
    assert await store.count("s1", "bob") == 1

  @pytest.mark.asyncio
  async def test_count(self, store):
    assert await store.count("s1") == 0
    await store.add(MemoryEntry(session_id="s1", content="a", created_at=1.0, updated_at=1.0))
    await store.add(MemoryEntry(session_id="s1", content="b", created_at=2.0, updated_at=2.0))
    assert await store.count("s1") == 2

  @pytest.mark.asyncio
  async def test_empty_session_returns_empty(self, store):
    entries = await store.get_entries("nonexistent")
    assert entries == []
    assert await store.count("nonexistent") == 0

  @pytest.mark.asyncio
  async def test_context_manager(self, tmp_path):
    async with FileStore(base_dir=str(tmp_path / "cm_memory")) as store:
      assert store._initialized
      await store.add(MemoryEntry(session_id="s1", content="test", created_at=1.0, updated_at=1.0))
      assert await store.count("s1") == 1
    assert not store._initialized
