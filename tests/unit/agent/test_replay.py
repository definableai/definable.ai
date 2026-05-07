"""Unit tests for the agent/replay module.

Tests cover Replay construction from events and RunOutput, compare_runs,
and all dataclass types. No real agent runs or trace files needed.
"""

import pytest

from definable.agent.replay.types import (
  KnowledgeRetrievalRecord,
  MemoryRecallRecord,
  ReplayComparison,
  ReplayStep,
  ReplayTokens,
  ToolCallRecord,
  ToolCallsDiff,
)
from definable.agent.replay.replay import Replay
from definable.agent.replay.compare import compare_runs


# ===========================================================================
# Dataclass types
# ===========================================================================


@pytest.mark.unit
class TestReplayTypes:
  """Tests for replay dataclass types."""

  def test_tool_call_record_defaults(self):
    r = ToolCallRecord()
    assert r.tool_name == ""
    assert r.tool_args is None
    assert r.result is None
    assert r.error is None
    assert r.started_at == 0
    assert r.completed_at is None
    assert r.duration_ms is None

  def test_tool_call_record_fields(self):
    r = ToolCallRecord(
      tool_name="search",
      tool_args={"query": "test"},
      result="found 3 results",
      started_at=1000,
      completed_at=1500,
      duration_ms=500.0,
    )
    assert r.tool_name == "search"
    assert r.tool_args == {"query": "test"}
    assert r.duration_ms == 500.0

  def test_replay_tokens_defaults(self):
    t = ReplayTokens()
    assert t.input_tokens == 0
    assert t.output_tokens == 0
    assert t.total_tokens == 0
    assert t.reasoning_tokens == 0

  def test_replay_step_defaults(self):
    s = ReplayStep()
    assert s.step_type == ""
    assert s.name is None
    assert s.started_at == 0

  def test_knowledge_retrieval_record(self):
    r = KnowledgeRetrievalRecord(
      query="test query",
      documents_found=10,
      documents_used=3,
      duration_ms=25.0,
    )
    assert r.query == "test query"
    assert r.documents_found == 10

  def test_memory_recall_record(self):
    r = MemoryRecallRecord(
      query="test",
      tokens_used=100,
      chunks_included=2,
      chunks_available=5,
    )
    assert r.tokens_used == 100
    assert r.chunks_available == 5

  def test_tool_calls_diff_defaults(self):
    d = ToolCallsDiff()
    assert d.added == []
    assert d.removed == []
    assert d.common == 0

  def test_replay_comparison_defaults(self):
    c = ReplayComparison()
    assert c.original is None
    assert c.replayed is None
    assert c.content_diff is None
    assert c.cost_diff is None
    assert c.token_diff == 0


# ===========================================================================
# Replay construction
# ===========================================================================


@pytest.mark.unit
class TestReplayConstruction:
  """Tests for Replay dataclass construction."""

  def test_empty_replay(self):
    r = Replay()
    assert r.run_id == ""
    assert r.content is None
    assert r.tool_calls == []
    assert r.steps == []
    assert r.status == "completed"
    assert r.source == "trace_file"

  def test_replay_with_fields(self):
    r = Replay(
      run_id="r1",
      session_id="s1",
      agent_name="test-agent",
      model="gpt-4o",
      content="Hello world",
      status="completed",
    )
    assert r.run_id == "r1"
    assert r.agent_name == "test-agent"
    assert r.content == "Hello world"


# ===========================================================================
# Replay.from_events
# ===========================================================================


@pytest.mark.unit
class TestReplayFromEvents:
  """Tests for Replay.from_events class method."""

  def test_empty_events_returns_empty_replay(self):
    r = Replay.from_events([])
    assert r.run_id == ""
    assert r.source == "trace_file"

  def test_from_run_started_event(self):
    from definable.run.agent import RunStartedEvent

    evt = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="test",
      session_id="s1",
      model="gpt-4o",
      model_provider="openai",
      created_at=1000,
    )
    r = Replay.from_events([evt])
    assert r.run_id == "r1"
    assert r.agent_id == "a1"
    assert r.agent_name == "test"
    assert r.model == "gpt-4o"

  def test_from_run_completed_event(self):
    from definable.run.agent import RunCompletedEvent, RunStartedEvent

    start = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    completed = RunCompletedEvent(
      run_id="r1",
      content="Answer",
      created_at=2000,
    )
    r = Replay.from_events([start, completed])
    assert r.content == "Answer"
    assert r.status == "completed"

  def test_from_error_event(self):
    from definable.run.agent import RunErrorEvent, RunStartedEvent

    start = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    error = RunErrorEvent(run_id="r1", content="boom", created_at=2000)
    r = Replay.from_events([start, error])
    assert r.status == "error"
    assert r.error == "boom"

  def test_from_cancelled_event(self):
    from definable.run.agent import RunCancelledEvent, RunStartedEvent

    start = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    cancelled = RunCancelledEvent(run_id="r1", created_at=2000)
    r = Replay.from_events([start, cancelled])
    assert r.status == "cancelled"

  def test_filter_by_run_id(self):
    from definable.run.agent import RunStartedEvent

    evt1 = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    evt2 = RunStartedEvent(
      run_id="r2",
      agent_id="a2",
      agent_name="t2",
      session_id="s2",
      model="m",
      model_provider="p",
      created_at=2000,
    )
    r = Replay.from_events([evt1, evt2], run_id="r2")
    assert r.run_id == "r2"
    assert r.agent_id == "a2"

  def test_duration_computed_from_events(self):
    from definable.run.agent import RunStartedEvent, RunCompletedEvent

    start = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    end = RunCompletedEvent(run_id="r1", content="done", created_at=3000)
    r = Replay.from_events([start, end])
    assert r.duration == 2000.0

  def test_tool_calls_recorded(self):
    from definable.run.agent import (
      RunStartedEvent,
      ToolCallCompletedEvent,
      ToolCallStartedEvent,
    )
    from definable.model.response import ToolExecution

    start = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="t",
      session_id="s1",
      model="m",
      model_provider="p",
      created_at=1000,
    )
    tool_start = ToolCallStartedEvent(
      run_id="r1",
      tool=ToolExecution(tool_name="search", tool_call_id="tc1", tool_args={"q": "test"}),
      created_at=1100,
    )
    tool_end = ToolCallCompletedEvent(
      run_id="r1",
      tool=ToolExecution(tool_name="search", tool_call_id="tc1", result="3 results"),
      created_at=1200,
    )
    r = Replay.from_events([start, tool_start, tool_end])
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].tool_name == "search"
    assert r.tool_calls[0].result == "3 results"


# ===========================================================================
# compare_runs
# ===========================================================================


@pytest.mark.unit
class TestCompareRuns:
  """Tests for compare_runs function."""

  def test_same_content_no_diff(self):
    a = Replay(content="Hello", tokens=ReplayTokens(total_tokens=10))
    b = Replay(content="Hello", tokens=ReplayTokens(total_tokens=10))
    cmp = compare_runs(a, b)
    assert cmp.content_diff is None
    assert cmp.token_diff == 0

  def test_different_content_has_diff(self):
    a = Replay(content="Hello", tokens=ReplayTokens(total_tokens=10))
    b = Replay(content="World", tokens=ReplayTokens(total_tokens=15))
    cmp = compare_runs(a, b)
    assert cmp.content_diff is not None
    assert cmp.token_diff == 5

  def test_cost_diff(self):
    a = Replay(cost=0.01, tokens=ReplayTokens())
    b = Replay(cost=0.02, tokens=ReplayTokens())
    cmp = compare_runs(a, b)
    assert cmp.cost_diff == pytest.approx(0.01)

  def test_cost_diff_none_when_missing(self):
    a = Replay(tokens=ReplayTokens())
    b = Replay(tokens=ReplayTokens())
    cmp = compare_runs(a, b)
    assert cmp.cost_diff is None

  def test_duration_diff(self):
    a = Replay(duration=1.0, tokens=ReplayTokens())
    b = Replay(duration=2.5, tokens=ReplayTokens())
    cmp = compare_runs(a, b)
    assert cmp.duration_diff == pytest.approx(1.5)

  def test_tool_calls_diff(self):
    a = Replay(
      tool_calls=[ToolCallRecord(tool_name="search")],
      tokens=ReplayTokens(),
    )
    b = Replay(
      tool_calls=[ToolCallRecord(tool_name="search"), ToolCallRecord(tool_name="calculate")],
      tokens=ReplayTokens(),
    )
    cmp = compare_runs(a, b)
    assert cmp.tool_calls_diff.common == 1
    assert len(cmp.tool_calls_diff.added) == 1
    assert cmp.tool_calls_diff.added[0].tool_name == "calculate"

  def test_tool_calls_removed(self):
    a = Replay(
      tool_calls=[ToolCallRecord(tool_name="search"), ToolCallRecord(tool_name="old_tool")],
      tokens=ReplayTokens(),
    )
    b = Replay(
      tool_calls=[ToolCallRecord(tool_name="search")],
      tokens=ReplayTokens(),
    )
    cmp = compare_runs(a, b)
    assert len(cmp.tool_calls_diff.removed) == 1
    assert cmp.tool_calls_diff.removed[0].tool_name == "old_tool"

  def test_references_original_and_replayed(self):
    a = Replay(run_id="a", tokens=ReplayTokens())
    b = Replay(run_id="b", tokens=ReplayTokens())
    cmp = compare_runs(a, b)
    assert cmp.original.run_id == "a"  # type: ignore[union-attr]
    assert cmp.replayed.run_id == "b"  # type: ignore[union-attr]
