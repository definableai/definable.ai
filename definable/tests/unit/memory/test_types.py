"""Tests for memory.types — MemoryEntry dataclass."""

import time

from definable.memory.types import MemoryEntry


class TestMemoryEntry:
  def test_auto_generates_id(self):
    entry = MemoryEntry(session_id="s1")
    assert entry.memory_id
    assert len(entry.memory_id) == 36  # UUID format

  def test_auto_generates_timestamps(self):
    before = time.time()
    entry = MemoryEntry(session_id="s1")
    after = time.time()
    assert before <= entry.created_at <= after
    assert before <= entry.updated_at <= after

  def test_preserves_explicit_values(self):
    entry = MemoryEntry(
      memory_id="custom-id",
      session_id="s1",
      user_id="alice",
      role="assistant",
      content="Hello",
      message_data={"role": "assistant", "content": "Hello"},
      created_at=1000.0,
      updated_at=2000.0,
    )
    assert entry.memory_id == "custom-id"
    assert entry.session_id == "s1"
    assert entry.user_id == "alice"
    assert entry.role == "assistant"
    assert entry.content == "Hello"
    assert entry.message_data == {"role": "assistant", "content": "Hello"}
    assert entry.created_at == 1000.0
    assert entry.updated_at == 2000.0

  def test_defaults(self):
    entry = MemoryEntry(session_id="s1")
    assert entry.user_id == "default"
    assert entry.role == "user"
    assert entry.content == ""
    assert entry.message_data is None

  def test_to_dict(self):
    entry = MemoryEntry(
      memory_id="m1",
      session_id="s1",
      user_id="default",
      role="user",
      content="test",
      created_at=1000.0,
      updated_at=1000.0,
    )
    d = entry.to_dict()
    assert d["memory_id"] == "m1"
    assert d["session_id"] == "s1"
    assert d["user_id"] == "default"
    assert d["role"] == "user"
    assert d["content"] == "test"
    assert d["message_data"] is None
    assert d["created_at"] == 1000.0

  def test_from_dict(self):
    data = {
      "memory_id": "m1",
      "session_id": "s1",
      "user_id": "alice",
      "role": "assistant",
      "content": "Hello!",
      "message_data": {"role": "assistant", "content": "Hello!"},
      "created_at": 1000.0,
      "updated_at": 2000.0,
    }
    entry = MemoryEntry.from_dict(data)
    assert entry.memory_id == "m1"
    assert entry.session_id == "s1"
    assert entry.user_id == "alice"
    assert entry.role == "assistant"
    assert entry.content == "Hello!"
    assert entry.message_data == {"role": "assistant", "content": "Hello!"}

  def test_roundtrip_dict(self):
    entry = MemoryEntry(
      session_id="s1",
      role="tool",
      content="result",
      message_data={"tool_call_id": "tc1", "content": "result"},
    )
    restored = MemoryEntry.from_dict(entry.to_dict())
    assert restored.memory_id == entry.memory_id
    assert restored.session_id == entry.session_id
    assert restored.role == entry.role
    assert restored.content == entry.content
    assert restored.message_data == entry.message_data
