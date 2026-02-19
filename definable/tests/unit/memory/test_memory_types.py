"""
Unit tests for the UserMemory dataclass.

Tests pure Python logic: creation, auto-generated fields, serialization.
No API calls. No external dependencies.

Covers:
  - UserMemory auto-generates UUID if memory_id not provided
  - UserMemory preserves explicit memory_id
  - Auto-set timestamps (created_at, updated_at)
  - Explicit timestamps preserved
  - topics defaults to empty list
  - topics=None becomes empty list
  - to_dict() / from_dict() round-trip
  - from_dict() with minimal data
"""

import time

import pytest

from definable.memory.types import UserMemory


@pytest.mark.unit
class TestUserMemory:
  """UserMemory dataclass construction and field defaults."""

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

  def test_topics_with_values(self):
    """UserMemory preserves explicit topics."""
    mem = UserMemory(memory="test", topics=["work", "coding"])
    assert mem.topics == ["work", "coding"]

  def test_user_id_default_none(self):
    """UserMemory defaults user_id to None."""
    mem = UserMemory(memory="test")
    assert mem.user_id is None

  def test_agent_id_default_none(self):
    """UserMemory defaults agent_id to None."""
    mem = UserMemory(memory="test")
    assert mem.agent_id is None

  def test_input_default_none(self):
    """UserMemory defaults input to None."""
    mem = UserMemory(memory="test")
    assert mem.input is None

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

  def test_two_memories_get_different_ids(self):
    """Two memories created without explicit IDs get different UUIDs."""
    mem1 = UserMemory(memory="fact one")
    mem2 = UserMemory(memory="fact two")
    assert mem1.memory_id != mem2.memory_id

  def test_to_dict_topics_empty_when_none(self):
    """to_dict() returns empty list for topics when they were None."""
    mem = UserMemory(memory="test", topics=None)
    d = mem.to_dict()
    assert d["topics"] == []
