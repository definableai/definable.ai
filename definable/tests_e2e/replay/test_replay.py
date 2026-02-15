"""Tests for Replay class — from_events, from_run_output, from_trace_file."""

from definable.models.metrics import Metrics
from definable.models.response import ToolExecution
from definable.replay import Replay, ReplayTokens
from definable.run.agent import (
  KnowledgeRetrievalCompletedEvent,
  MemoryRecallCompletedEvent,
  RunCompletedEvent,
  RunErrorEvent,
  RunInput,
  RunOutput,
  RunStartedEvent,
  ToolCallCompletedEvent,
  ToolCallStartedEvent,
)
from definable.run.base import RunStatus


def _make_started(
  run_id: str = "run-1",
  session_id: str = "sess-1",
  ts: int = 1000,
) -> RunStartedEvent:
  return RunStartedEvent(
    created_at=ts,
    run_id=run_id,
    session_id=session_id,
    agent_id="agent-1",
    agent_name="TestAgent",
    model="gpt-4o-mini",
    model_provider="openai",
    run_input=RunInput(input_content="Hello"),
  )


def _make_completed(
  run_id: str = "run-1",
  session_id: str = "sess-1",
  ts: int = 1002,
  content: str = "Hi there!",
  metrics: Metrics | None = None,
) -> RunCompletedEvent:
  return RunCompletedEvent(
    created_at=ts,
    run_id=run_id,
    session_id=session_id,
    agent_id="agent-1",
    agent_name="TestAgent",
    content=content,
    metrics=metrics or Metrics(input_tokens=10, output_tokens=5, total_tokens=15),
  )


def _make_tool_started(
  run_id: str = "run-1",
  session_id: str = "sess-1",
  ts: int = 1001,
  tool_name: str = "search",
  tool_call_id: str = "tc-1",
) -> ToolCallStartedEvent:
  return ToolCallStartedEvent(
    created_at=ts,
    run_id=run_id,
    session_id=session_id,
    agent_id="agent-1",
    agent_name="TestAgent",
    tool=ToolExecution(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_args={"query": "test"},
    ),
  )


def _make_tool_completed(
  run_id: str = "run-1",
  session_id: str = "sess-1",
  ts: int = 1002,
  tool_name: str = "search",
  tool_call_id: str = "tc-1",
  result: str = "found it",
) -> ToolCallCompletedEvent:
  return ToolCallCompletedEvent(
    created_at=ts,
    run_id=run_id,
    session_id=session_id,
    agent_id="agent-1",
    agent_name="TestAgent",
    tool=ToolExecution(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_args={"query": "test"},
      result=result,
    ),
    content=result,
  )


class TestFromEventsBasic:
  def test_minimal(self):
    events = [_make_started(), _make_completed()]
    replay = Replay.from_events(events)

    assert replay.run_id == "run-1"
    assert replay.session_id == "sess-1"
    assert replay.agent_id == "agent-1"
    assert replay.agent_name == "TestAgent"
    assert replay.model == "gpt-4o-mini"
    assert replay.model_provider == "openai"
    assert replay.content == "Hi there!"
    assert replay.status == "completed"
    assert replay.source == "trace_file"

  def test_input_preserved(self):
    events = [_make_started(), _make_completed()]
    replay = Replay.from_events(events)

    assert replay.input is not None
    assert replay.input.input_content == "Hello"

  def test_empty_events(self):
    replay = Replay.from_events([])
    assert replay.run_id == ""
    assert replay.source == "trace_file"

  def test_run_id_filter(self):
    events = [
      _make_started(run_id="run-1"),
      _make_completed(run_id="run-1", content="first"),
      _make_started(run_id="run-2"),
      _make_completed(run_id="run-2", content="second"),
    ]
    replay = Replay.from_events(events, run_id="run-2")
    assert replay.run_id == "run-2"
    assert replay.content == "second"


class TestFromEventsWithToolCalls:
  def test_tool_call_extraction(self):
    events = [
      _make_started(),
      _make_tool_started(),
      _make_tool_completed(),
      _make_completed(ts=1003),
    ]
    replay = Replay.from_events(events)

    assert len(replay.tool_calls) == 1
    tc = replay.tool_calls[0]
    assert tc.tool_name == "search"
    assert tc.tool_args == {"query": "test"}
    assert tc.result == "found it"

  def test_multiple_tool_calls(self):
    events = [
      _make_started(),
      _make_tool_started(tool_name="search", tool_call_id="tc-1", ts=1001),
      _make_tool_completed(tool_name="search", tool_call_id="tc-1", ts=1002),
      _make_tool_started(tool_name="calculate", tool_call_id="tc-2", ts=1003),
      _make_tool_completed(tool_name="calculate", tool_call_id="tc-2", ts=1004, result="42"),
      _make_completed(ts=1005),
    ]
    replay = Replay.from_events(events)

    assert len(replay.tool_calls) == 2
    assert replay.tool_calls[0].tool_name == "search"
    assert replay.tool_calls[1].tool_name == "calculate"
    assert replay.tool_calls[1].result == "42"

  def test_tool_call_timing(self):
    events = [
      _make_started(ts=1000),
      _make_tool_started(ts=1001, tool_call_id="tc-1"),
      _make_tool_completed(ts=1003, tool_call_id="tc-1"),
      _make_completed(ts=1004),
    ]
    replay = Replay.from_events(events)

    tc = replay.tool_calls[0]
    assert tc.started_at == 1001
    assert tc.completed_at == 1003
    assert tc.duration_ms == 2000.0


class TestFromEventsWithKnowledgeAndMemory:
  def test_knowledge_retrieval(self):
    events = [
      _make_started(),
      KnowledgeRetrievalCompletedEvent(
        created_at=1001,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        query="test query",
        documents_found=10,
        documents_used=3,
        duration_ms=50.0,
      ),
      _make_completed(),
    ]
    replay = Replay.from_events(events)

    assert len(replay.knowledge_retrievals) == 1
    kr = replay.knowledge_retrievals[0]
    assert kr.query == "test query"
    assert kr.documents_found == 10
    assert kr.documents_used == 3
    assert kr.duration_ms == 50.0

  def test_memory_recall(self):
    events = [
      _make_started(),
      MemoryRecallCompletedEvent(
        created_at=1001,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        agent_name="TestAgent",
        query="what happened?",
        tokens_used=200,
        chunks_included=3,
        chunks_available=10,
        duration_ms=30.0,
      ),
      _make_completed(),
    ]
    replay = Replay.from_events(events)

    assert len(replay.memory_recalls) == 1
    mr = replay.memory_recalls[0]
    assert mr.query == "what happened?"
    assert mr.tokens_used == 200
    assert mr.chunks_included == 3
    assert mr.chunks_available == 10
    assert mr.duration_ms == 30.0


class TestFromRunOutput:
  def test_basic(self):
    output = RunOutput(
      run_id="run-1",
      session_id="sess-1",
      agent_id="agent-1",
      agent_name="TestAgent",
      model="gpt-4o-mini",
      model_provider="openai",
      content="Hello!",
      status=RunStatus.completed,
      metrics=Metrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.001),
      input=RunInput(input_content="Hi"),
    )
    replay = Replay.from_run_output(output)

    assert replay.run_id == "run-1"
    assert replay.content == "Hello!"
    assert replay.status == "completed"
    assert replay.source == "run_output"
    assert replay.tokens.input_tokens == 10
    assert replay.tokens.output_tokens == 5
    assert replay.tokens.total_tokens == 15
    assert replay.cost == 0.001

  def test_with_tools(self):
    output = RunOutput(
      run_id="run-1",
      session_id="sess-1",
      content="done",
      status=RunStatus.completed,
      tools=[
        ToolExecution(
          tool_call_id="tc-1",
          tool_name="search",
          tool_args={"q": "test"},
          result="found",
        ),
      ],
    )
    replay = Replay.from_run_output(output)

    assert len(replay.tool_calls) == 1
    assert replay.tool_calls[0].tool_name == "search"
    assert replay.tool_calls[0].result == "found"

  def test_error_status(self):
    output = RunOutput(
      run_id="run-1",
      status=RunStatus.error,
    )
    replay = Replay.from_run_output(output)
    assert replay.status == "error"


class TestTokensAndCost:
  def test_from_events(self):
    metrics = Metrics(
      input_tokens=100,
      output_tokens=50,
      total_tokens=150,
      reasoning_tokens=20,
      cache_read_tokens=10,
      cache_write_tokens=5,
      cost=0.01,
      duration=1.5,
    )
    events = [
      _make_started(),
      _make_completed(metrics=metrics),
    ]
    replay = Replay.from_events(events)

    assert replay.tokens == ReplayTokens(
      input_tokens=100,
      output_tokens=50,
      total_tokens=150,
      reasoning_tokens=20,
      cache_read_tokens=10,
      cache_write_tokens=5,
    )
    assert replay.cost == 0.01
    assert replay.duration == 1.5


class TestStepsOrdering:
  def test_chronological(self):
    events = [
      _make_started(ts=1000),
      _make_tool_started(ts=1001, tool_call_id="tc-1"),
      _make_tool_completed(ts=1002, tool_call_id="tc-1"),
      _make_completed(ts=1003),
    ]
    replay = Replay.from_events(events)

    assert len(replay.steps) >= 2
    # Steps should be in order of started_at
    for i in range(len(replay.steps) - 1):
      assert replay.steps[i].started_at <= replay.steps[i + 1].started_at


class TestMultipleRunsInSession:
  def test_filter_by_run_id(self):
    events = [
      _make_started(run_id="run-1", ts=1000),
      _make_completed(run_id="run-1", ts=1001, content="first"),
      _make_started(run_id="run-2", ts=1002),
      _make_tool_started(run_id="run-2", ts=1003, tool_call_id="tc-1"),
      _make_tool_completed(run_id="run-2", ts=1004, tool_call_id="tc-1"),
      _make_completed(run_id="run-2", ts=1005, content="second"),
    ]

    r1 = Replay.from_events(events, run_id="run-1")
    assert r1.content == "first"
    assert len(r1.tool_calls) == 0

    r2 = Replay.from_events(events, run_id="run-2")
    assert r2.content == "second"
    assert len(r2.tool_calls) == 1


class TestErrorRun:
  def test_error_event(self):
    events = [
      _make_started(),
      RunErrorEvent(
        created_at=1001,
        run_id="run-1",
        session_id="sess-1",
        agent_id="agent-1",
        error_type="ValueError",
        content="Something went wrong",
      ),
    ]
    replay = Replay.from_events(events)

    assert replay.status == "error"
    assert replay.error == "Something went wrong"


class TestFromTraceFile:
  def test_round_trip(self, tmp_path):
    """Write events to JSONL, read them back via from_trace_file."""
    events = [_make_started(), _make_completed()]
    jsonl_path = tmp_path / "test_session.jsonl"
    with open(jsonl_path, "w") as f:
      for evt in events:
        f.write(evt.to_json(indent=None) + "\n")

    replay = Replay.from_trace_file(jsonl_path)

    assert replay.run_id == "run-1"
    assert replay.content == "Hi there!"
    assert replay.status == "completed"

  def test_round_trip_with_tools(self, tmp_path):
    events = [
      _make_started(),
      _make_tool_started(),
      _make_tool_completed(),
      _make_completed(ts=1003),
    ]
    jsonl_path = tmp_path / "test_session.jsonl"
    with open(jsonl_path, "w") as f:
      for evt in events:
        f.write(evt.to_json(indent=None) + "\n")

    replay = Replay.from_trace_file(jsonl_path)

    assert len(replay.tool_calls) == 1
    assert replay.tool_calls[0].tool_name == "search"
