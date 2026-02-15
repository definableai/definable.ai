"""Tests for compare_runs — diffing two replays."""

from definable.replay import Replay, ReplayTokens, compare_runs
from definable.replay.types import ToolCallRecord


def _make_replay(
  content: str = "Hello",
  cost: float | None = 0.01,
  total_tokens: int = 100,
  duration: float | None = 1.0,
  tool_calls: list[ToolCallRecord] | None = None,
) -> Replay:
  return Replay(
    run_id="run-1",
    content=content,
    tokens=ReplayTokens(total_tokens=total_tokens),
    cost=cost,
    duration=duration,
    tool_calls=tool_calls or [],
  )


class TestCompareSameRun:
  def test_no_diffs(self):
    r = _make_replay()
    result = compare_runs(r, r)

    assert result.content_diff is None
    assert result.token_diff == 0
    assert result.cost_diff == 0.0
    assert result.duration_diff == 0.0
    assert result.tool_calls_diff.added == []
    assert result.tool_calls_diff.removed == []


class TestCompareDifferentContent:
  def test_content_diff_populated(self):
    a = _make_replay(content="Hello world")
    b = _make_replay(content="Hello universe")
    result = compare_runs(a, b)

    assert result.content_diff is not None
    assert "Hello world" in result.content_diff
    assert "Hello universe" in result.content_diff

  def test_one_empty_content(self):
    a = _make_replay(content="Hello")
    b = Replay(run_id="run-2")
    result = compare_runs(a, b)

    assert result.content_diff is not None


class TestCompareDifferentTools:
  def test_added_tool(self):
    a = _make_replay(tool_calls=[])
    b = _make_replay(
      tool_calls=[ToolCallRecord(tool_name="search")],
    )
    result = compare_runs(a, b)

    assert len(result.tool_calls_diff.added) == 1
    assert result.tool_calls_diff.added[0].tool_name == "search"
    assert result.tool_calls_diff.removed == []

  def test_removed_tool(self):
    a = _make_replay(
      tool_calls=[ToolCallRecord(tool_name="search")],
    )
    b = _make_replay(tool_calls=[])
    result = compare_runs(a, b)

    assert result.tool_calls_diff.removed[0].tool_name == "search"
    assert result.tool_calls_diff.added == []

  def test_common_tools(self):
    tc = ToolCallRecord(tool_name="search")
    a = _make_replay(tool_calls=[tc])
    b = _make_replay(tool_calls=[tc])
    result = compare_runs(a, b)

    assert result.tool_calls_diff.common == 1
    assert result.tool_calls_diff.added == []
    assert result.tool_calls_diff.removed == []


class TestCompareCostAndTokens:
  def test_cost_diff(self):
    a = _make_replay(cost=0.01)
    b = _make_replay(cost=0.03)
    result = compare_runs(a, b)

    assert result.cost_diff is not None
    assert abs(result.cost_diff - 0.02) < 1e-9

  def test_token_diff(self):
    a = _make_replay(total_tokens=100)
    b = _make_replay(total_tokens=150)
    result = compare_runs(a, b)

    assert result.token_diff == 50

  def test_duration_diff(self):
    a = _make_replay(duration=1.0)
    b = _make_replay(duration=2.5)
    result = compare_runs(a, b)

    assert result.duration_diff is not None
    assert abs(result.duration_diff - 1.5) < 1e-9

  def test_none_cost(self):
    a = _make_replay(cost=None)
    b = _make_replay(cost=0.05)
    result = compare_runs(a, b)

    assert result.cost_diff is None

  def test_none_duration(self):
    a = _make_replay(duration=None)
    b = _make_replay(duration=1.0)
    result = compare_runs(a, b)

    assert result.duration_diff is None
