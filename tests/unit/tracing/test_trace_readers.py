"""Unit tests for read_trace_file and read_trace_events functions.

Covers JSONL reading, malformed line handling, empty files, and event
deserialization from trace files.
"""

import json

import pytest

from definable.agent.tracing.jsonl import read_trace_file, read_trace_events


# ===========================================================================
# read_trace_file
# ===========================================================================


@pytest.mark.unit
class TestReadTraceFile:
  """Tests for read_trace_file()."""

  def test_reads_single_event(self, tmp_path):
    p = tmp_path / "session.jsonl"
    event = {"event": "RunStarted", "run_id": "r1", "created_at": 1000}
    p.write_text(json.dumps(event) + "\n")

    result = read_trace_file(p)
    assert len(result) == 1
    assert result[0]["event"] == "RunStarted"
    assert result[0]["run_id"] == "r1"

  def test_reads_multiple_events(self, tmp_path):
    p = tmp_path / "session.jsonl"
    events = [
      {"event": "RunStarted", "run_id": "r1", "created_at": 1000},
      {"event": "ToolCallStarted", "run_id": "r1", "created_at": 1100},
      {"event": "RunCompleted", "run_id": "r1", "created_at": 2000},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    result = read_trace_file(p)
    assert len(result) == 3
    assert result[0]["event"] == "RunStarted"
    assert result[2]["event"] == "RunCompleted"

  def test_empty_file(self, tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    result = read_trace_file(p)
    assert result == []

  def test_skips_blank_lines(self, tmp_path):
    p = tmp_path / "session.jsonl"
    content = (
      json.dumps({"event": "RunStarted", "run_id": "r1", "created_at": 1000})
      + "\n\n\n"
      + json.dumps({"event": "RunCompleted", "run_id": "r1", "created_at": 2000})
      + "\n"
    )
    p.write_text(content)
    result = read_trace_file(p)
    assert len(result) == 2

  def test_returns_dicts_not_objects(self, tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text(json.dumps({"event": "RunStarted", "run_id": "r1", "created_at": 1000}) + "\n")
    result = read_trace_file(p)
    assert isinstance(result[0], dict)

  def test_nonexistent_file_raises(self, tmp_path):
    p = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError):
      read_trace_file(p)

  def test_preserves_nested_data(self, tmp_path):
    p = tmp_path / "session.jsonl"
    event = {
      "event": "ToolCallCompleted",
      "run_id": "r1",
      "tool": {"tool_name": "search", "tool_args": {"q": "test"}},
      "created_at": 1000,
    }
    p.write_text(json.dumps(event) + "\n")
    result = read_trace_file(p)
    assert result[0]["tool"]["tool_name"] == "search"


# ===========================================================================
# read_trace_events
# ===========================================================================


@pytest.mark.unit
class TestReadTraceEvents:
  """Tests for read_trace_events()."""

  def test_deserializes_run_started_event(self, tmp_path):
    from definable.run.agent import RunStartedEvent

    p = tmp_path / "session.jsonl"
    event_data = {
      "event": "RunStarted",
      "run_id": "r1",
      "agent_id": "a1",
      "agent_name": "test",
      "session_id": "s1",
      "model": "gpt-4o",
      "model_provider": "openai",
      "created_at": 1000,
    }
    p.write_text(json.dumps(event_data) + "\n")

    events = read_trace_events(p)
    assert len(events) == 1
    assert isinstance(events[0], RunStartedEvent)
    assert events[0].run_id == "r1"

  def test_deserializes_run_completed_event(self, tmp_path):
    from definable.run.agent import RunCompletedEvent

    p = tmp_path / "session.jsonl"
    event_data = {
      "event": "RunCompleted",
      "run_id": "r1",
      "content": "done",
      "created_at": 2000,
    }
    p.write_text(json.dumps(event_data) + "\n")

    events = read_trace_events(p)
    assert len(events) == 1
    assert isinstance(events[0], RunCompletedEvent)
    assert events[0].content == "done"

  def test_deserializes_multiple_events(self, tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
      json.dumps({
        "event": "RunStarted",
        "run_id": "r1",
        "agent_id": "a1",
        "agent_name": "test",
        "session_id": "s1",
        "model": "gpt-4o",
        "model_provider": "openai",
        "created_at": 1000,
      }),
      json.dumps({
        "event": "RunCompleted",
        "run_id": "r1",
        "content": "result",
        "created_at": 2000,
      }),
    ]
    p.write_text("\n".join(lines) + "\n")

    events = read_trace_events(p)
    assert len(events) == 2

  def test_empty_file(self, tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    events = read_trace_events(p)
    assert events == []

  def test_nonexistent_file_raises(self, tmp_path):
    p = tmp_path / "nonexistent.jsonl"
    with pytest.raises(FileNotFoundError):
      read_trace_events(p)

  def test_roundtrip_with_jsonl_exporter(self, tmp_path):
    """Write events via JSONLExporter, then read them back via read_trace_events."""
    from definable.run.agent import RunCompletedEvent, RunStartedEvent
    from definable.agent.tracing.jsonl import JSONLExporter

    exporter = JSONLExporter(trace_dir=str(tmp_path), mirror_stdout=False)

    start_evt = RunStartedEvent(
      run_id="r1",
      agent_id="a1",
      agent_name="test",
      session_id="s1",
      model="gpt-4o",
      model_provider="openai",
      created_at=1000,
    )
    end_evt = RunCompletedEvent(
      run_id="r1",
      content="Hello",
      created_at=2000,
    )
    # RunCompletedEvent also needs session_id for proper file routing
    end_evt.session_id = "s1"

    exporter.export(start_evt)
    exporter.export(end_evt)
    exporter.shutdown()

    trace_path = tmp_path / "s1.jsonl"
    events = read_trace_events(trace_path)
    assert len(events) == 2
    assert isinstance(events[0], RunStartedEvent)
    assert isinstance(events[1], RunCompletedEvent)
    assert events[1].content == "Hello"
