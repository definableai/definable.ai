"""Tests for Cortex Scratchpad."""

from definable.memory.cortex.record.scratchpad import Scratchpad, merge_scratchpads


class TestScratchpad:
  def test_defaults(self):
    sp = Scratchpad()
    assert sp.session_id == "default"
    assert sp.beliefs == {}
    assert sp.active_topics == []
    assert sp.updated_at > 0

  def test_belief_crud(self):
    sp = Scratchpad()
    sp.set_belief("name", "Anandesh")
    assert sp.get_belief("name") == "Anandesh"
    assert sp.get_belief("missing", "fallback") == "fallback"
    sp.remove_belief("name")
    assert sp.get_belief("name") is None

  def test_topics(self):
    sp = Scratchpad()
    sp.add_topic("memory-systems")
    sp.add_topic("memory-systems")  # duplicate ignored
    assert sp.active_topics == ["memory-systems"]
    sp.remove_topic("memory-systems")
    assert sp.active_topics == []

  def test_format_for_prompt(self):
    sp = Scratchpad(
      beliefs={"style": "direct"},
      active_topics=["cortex"],
      pending_tasks=["review PR"],
    )
    xml = sp.format_for_prompt()
    assert "<scratchpad>" in xml
    assert 'key="style"' in xml
    assert "cortex" in xml
    assert "review PR" in xml

  def test_roundtrip(self):
    sp = Scratchpad(
      session_id="s1",
      user_id="u1",
      beliefs={"key": "value"},
      active_topics=["topic1"],
      pending_tasks=["task1"],
    )
    d = sp.to_dict()
    restored = Scratchpad.from_dict(d)
    assert restored.session_id == "s1"
    assert restored.beliefs == {"key": "value"}
    assert restored.active_topics == ["topic1"]


class TestMergeScratchpads:
  def test_merge_none(self):
    base = Scratchpad(beliefs={"a": 1})
    assert merge_scratchpads(base, None) is base

  def test_merge_beliefs(self):
    base = Scratchpad(beliefs={"a": 1, "b": 2})
    update = Scratchpad(beliefs={"b": 3, "c": 4})
    merged = merge_scratchpads(base, update)
    assert merged.beliefs == {"a": 1, "b": 3, "c": 4}

  def test_merge_topics(self):
    base = Scratchpad(active_topics=["x", "y"])
    update = Scratchpad(active_topics=["y", "z"])
    merged = merge_scratchpads(base, update)
    assert merged.active_topics == ["x", "y", "z"]
