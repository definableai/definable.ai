"""Unit tests for HistoryTrimmer — all strategies + tool-pair protection."""

import pytest

from definable.agent.context.history import (
  HistoryTrimmer,
  flatten_groups,
  group_messages,
  trim_head_and_tail,
  trim_tail,
)
from definable.model.message import Message


def _msg(role: str, content: str = "", tool_calls: list | None = None, tool_call_id: str | None = None) -> Message:
  """Helper to create test messages."""
  return Message(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)


def _user(content: str = "hi") -> Message:
  return _msg("user", content)


def _assistant(content: str = "ok", tool_calls: list | None = None) -> Message:
  return _msg("assistant", content, tool_calls=tool_calls)


def _tool(name: str = "search", call_id: str = "tc_1") -> Message:
  return _msg("tool", f"result from {name}", tool_call_id=call_id)


def _tool_call_list() -> list:
  """A minimal tool_calls list to mark an assistant message as having tool calls."""
  return [{"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]


# ── Grouping ──────────────────────────────────────────────────


@pytest.mark.unit
class TestGroupMessages:
  def test_standalone_messages(self):
    msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
    groups = group_messages(msgs)
    assert len(groups) == 4
    assert all(g.size == 1 for g in groups)

  def test_tool_call_group(self):
    msgs = [
      _user("search for X"),
      _assistant("I'll search", tool_calls=_tool_call_list()),
      _tool("search", "tc_1"),
      _user("thanks"),
    ]
    groups = group_messages(msgs)
    assert len(groups) == 3
    # Second group: assistant + tool = 2 messages
    assert groups[1].size == 2
    assert groups[1].messages[0].role == "assistant"
    assert groups[1].messages[1].role == "tool"

  def test_multiple_tool_results(self):
    """An assistant with tool_calls followed by 3 tool results = one group of 4."""
    msgs = [
      _assistant("calling tools", tool_calls=_tool_call_list()),
      _tool("search", "tc_1"),
      _tool("fetch", "tc_2"),
      _tool("compute", "tc_3"),
    ]
    groups = group_messages(msgs)
    assert len(groups) == 1
    assert groups[0].size == 4

  def test_empty_messages(self):
    assert group_messages([]) == []

  def test_flatten_roundtrip(self):
    msgs = [_user("q"), _assistant("a", tool_calls=_tool_call_list()), _tool("s")]
    groups = group_messages(msgs)
    flat = flatten_groups(groups)
    assert len(flat) == len(msgs)
    for original, restored in zip(msgs, flat):
      assert original.role == restored.role
      assert original.content == restored.content


# ── Tail trimming ─────────────────────────────────────────────


@pytest.mark.unit
class TestTrimTail:
  def test_no_trim_when_under_limit(self):
    msgs = [_user(), _assistant(), _user(), _assistant()]
    result = trim_tail(msgs, max_messages=10)
    assert len(result) == 4

  def test_trim_to_limit(self):
    msgs = [_user(f"q{i}") for i in range(20)]
    result = trim_tail(msgs, max_messages=5)
    assert len(result) == 5
    # Should keep the LAST 5
    assert result[0].content == "q15"
    assert result[-1].content == "q19"

  def test_trim_preserves_tool_pairs(self):
    """Trimming should not orphan tool results from their assistant."""
    msgs = [
      _user("q1"),
      _assistant("a1"),
      _user("q2"),
      _assistant("calling", tool_calls=_tool_call_list()),
      _tool("search"),
      _user("q3"),
      _assistant("a3"),
    ]
    # max_messages=4 should keep the last few, but not split the tool group
    result = trim_tail(msgs, max_messages=4)
    # The tool group (assistant+tool = 2) should stay together
    roles = [m.role for m in result]
    # If "tool" is present, the preceding "assistant" with tool_calls must also be present
    if "tool" in roles:
      tool_idx = roles.index("tool")
      assert tool_idx > 0
      assert result[tool_idx - 1].role == "assistant"
      assert result[tool_idx - 1].tool_calls is not None

  def test_trim_empty_list(self):
    assert trim_tail([], max_messages=5) == []

  def test_trim_single_message(self):
    msgs = [_user("only")]
    result = trim_tail(msgs, max_messages=1)
    assert len(result) == 1


# ── Head and tail trimming ────────────────────────────────────


@pytest.mark.unit
class TestTrimHeadAndTail:
  def test_no_trim_when_under_limit(self):
    msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
    result = trim_head_and_tail(msgs, keep_first=2, keep_last=2)
    assert len(result) == 4

  def test_keeps_head_and_tail(self):
    msgs = [_user(f"q{i}") for i in range(20)]
    result = trim_head_and_tail(msgs, keep_first=3, keep_last=3)
    # Should keep first 3 + last 3
    assert len(result) == 6
    assert result[0].content == "q0"
    assert result[2].content == "q2"
    assert result[3].content == "q17"
    assert result[-1].content == "q19"

  def test_drops_middle(self):
    msgs = [_user(f"m{i}") for i in range(10)]
    result = trim_head_and_tail(msgs, keep_first=2, keep_last=2)
    contents = [m.content for m in result]
    assert "m0" in contents
    assert "m1" in contents
    assert "m8" in contents
    assert "m9" in contents
    # Middle should be dropped
    assert "m5" not in contents

  def test_head_and_tail_preserves_tool_pairs(self):
    msgs = [
      _user("q1"),
      _user("q2"),
      _assistant("calling", tool_calls=_tool_call_list()),
      _tool("search"),
      _user("q3"),
      _assistant("a3"),
      _user("q4"),
      _assistant("a4"),
    ]
    result = trim_head_and_tail(msgs, keep_first=2, keep_last=2)
    roles = [m.role for m in result]
    # Tool pairs should stay together
    if "tool" in roles:
      tool_idx = roles.index("tool")
      assert result[tool_idx - 1].role == "assistant"

  def test_empty_list(self):
    assert trim_head_and_tail([], keep_first=2, keep_last=2) == []


# ── HistoryTrimmer class ─────────────────────────────────────


@pytest.mark.unit
class TestHistoryTrimmer:
  def test_none_strategy_no_trimming(self):
    trimmer = HistoryTrimmer(strategy="none")
    msgs = [_user(f"q{i}") for i in range(100)]
    result = trimmer.trim(msgs)
    assert len(result) == 100

  def test_tail_strategy(self):
    trimmer = HistoryTrimmer(strategy="tail", max_messages=5)
    msgs = [_user(f"q{i}") for i in range(20)]
    result = trimmer.trim(msgs)
    assert len(result) == 5

  def test_head_and_tail_strategy(self):
    trimmer = HistoryTrimmer(strategy="head_and_tail", max_messages=5, keep_first=3)
    msgs = [_user(f"q{i}") for i in range(20)]
    result = trimmer.trim(msgs)
    assert len(result) == 8  # 3 head + 5 tail

  def test_summarize_falls_back_to_tail(self):
    trimmer = HistoryTrimmer(strategy="summarize", max_messages=5)
    msgs = [_user(f"q{i}") for i in range(20)]
    result = trimmer.trim(msgs)
    assert len(result) == 5  # Phase 3 fallback to tail

  def test_strategy_property(self):
    trimmer = HistoryTrimmer(strategy="tail")
    assert trimmer.strategy == "tail"

  def test_max_messages_none_no_trimming(self):
    trimmer = HistoryTrimmer(strategy="tail", max_messages=None)
    msgs = [_user(f"q{i}") for i in range(100)]
    result = trimmer.trim(msgs)
    assert len(result) == 100
