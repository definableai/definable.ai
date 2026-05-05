"""Tests for TraceBrowser — JSONL file scanning, session/run listing, corrupt file handling."""

import json
import tempfile
from pathlib import Path

import pytest

from definable.agent.observability.trace_browser import TraceBrowser


def _write_events(path: Path, events: list):
  """Write events to a JSONL file."""
  with open(path, "w", encoding="utf-8") as f:
    for evt in events:
      f.write(json.dumps(evt) + "\n")


class TestListSessions:
  """Tests for TraceBrowser.list_sessions()."""

  def test_empty_directory(self):
    """Empty trace dir should return empty list."""
    with tempfile.TemporaryDirectory() as d:
      tb = TraceBrowser(trace_dir=d)
      assert tb.list_sessions() == []

  def test_lists_jsonl_files(self):
    """Should list all .jsonl files as sessions."""
    with tempfile.TemporaryDirectory() as d:
      _write_events(Path(d) / "sess1.jsonl", [{"event": "RunStarted", "run_id": "r1"}])
      _write_events(Path(d) / "sess2.jsonl", [{"event": "RunStarted", "run_id": "r2"}])

      tb = TraceBrowser(trace_dir=d)
      sessions = tb.list_sessions()

      assert len(sessions) == 2
      ids = {s.session_id for s in sessions}
      assert ids == {"sess1", "sess2"}

  def test_sorted_by_modified_time(self):
    """Sessions should be sorted by most recently modified."""
    import time

    with tempfile.TemporaryDirectory() as d:
      p1 = Path(d) / "old.jsonl"
      _write_events(p1, [{"event": "RunStarted", "run_id": "r1"}])
      time.sleep(0.05)
      p2 = Path(d) / "new.jsonl"
      _write_events(p2, [{"event": "RunStarted", "run_id": "r2"}])

      tb = TraceBrowser(trace_dir=d)
      sessions = tb.list_sessions()

      assert sessions[0].session_id == "new"
      assert sessions[1].session_id == "old"

  def test_counts_unique_run_ids(self):
    """run_count should reflect unique run IDs in the file."""
    with tempfile.TemporaryDirectory() as d:
      _write_events(
        Path(d) / "sess.jsonl",
        [
          {"event": "RunStarted", "run_id": "r1"},
          {"event": "ToolCallStarted", "run_id": "r1"},
          {"event": "RunStarted", "run_id": "r2"},
        ],
      )

      tb = TraceBrowser(trace_dir=d)
      sessions = tb.list_sessions()

      assert sessions[0].run_count == 2

  def test_nonexistent_trace_dir_created(self):
    """TraceBrowser should create the trace dir if it doesn't exist."""
    with tempfile.TemporaryDirectory() as d:
      new_dir = Path(d) / "subdir" / "traces"
      tb = TraceBrowser(trace_dir=str(new_dir))
      assert new_dir.is_dir()
      assert tb.list_sessions() == []


class TestListRuns:
  """Tests for TraceBrowser.list_runs()."""

  def test_list_runs_for_session(self):
    """Should list all runs in a session."""
    with tempfile.TemporaryDirectory() as d:
      _write_events(
        Path(d) / "sess.jsonl",
        [
          {"event": "RunStarted", "run_id": "r1", "created_at": 1000, "agent_name": "test", "model": "gpt-4o"},
          {"event": "RunCompleted", "run_id": "r1", "created_at": 1005, "metrics": {"total_tokens": 100, "cost": 0.01, "duration": 5.0}},
          {"event": "RunStarted", "run_id": "r2", "created_at": 2000, "agent_name": "test", "model": "gpt-4o"},
          {"event": "RunError", "run_id": "r2", "created_at": 2003},
        ],
      )

      tb = TraceBrowser(trace_dir=d)
      runs = tb.list_runs("sess")

      assert len(runs) == 2
      r1 = next(r for r in runs if r.run_id == "r1")
      assert r1.status == "completed"
      assert r1.tokens == 100
      assert r1.cost == 0.01

      r2 = next(r for r in runs if r.run_id == "r2")
      assert r2.status == "error"

  def test_missing_session_raises(self):
    """Should raise FileNotFoundError for nonexistent session."""
    with tempfile.TemporaryDirectory() as d:
      tb = TraceBrowser(trace_dir=d)
      with pytest.raises(FileNotFoundError):
        tb.list_runs("nonexistent")

  def test_empty_file_returns_empty(self):
    """Empty JSONL file should return no runs."""
    with tempfile.TemporaryDirectory() as d:
      (Path(d) / "empty.jsonl").touch()
      tb = TraceBrowser(trace_dir=d)
      runs = tb.list_runs("empty")
      assert runs == []

  def test_duration_computed_from_timestamps(self):
    """If metrics.duration is not available, duration should be computed from timestamps."""
    with tempfile.TemporaryDirectory() as d:
      _write_events(
        Path(d) / "sess.jsonl",
        [
          {"event": "RunStarted", "run_id": "r1", "created_at": 1000},
          {"event": "ToolCallStarted", "run_id": "r1", "created_at": 1002},
          {"event": "RunCompleted", "run_id": "r1", "created_at": 1005, "metrics": {"total_tokens": 50}},
        ],
      )

      tb = TraceBrowser(trace_dir=d)
      runs = tb.list_runs("sess")
      assert runs[0].duration == 5.0  # 1005 - 1000


class TestLoadRunEvents:
  """Tests for TraceBrowser.load_run_events()."""

  def test_filters_by_run_id(self):
    """Should only return events for the specified run_id."""
    with tempfile.TemporaryDirectory() as d:
      _write_events(
        Path(d) / "sess.jsonl",
        [
          {"event": "RunStarted", "run_id": "r1"},
          {"event": "RunStarted", "run_id": "r2"},
          {"event": "RunCompleted", "run_id": "r1"},
        ],
      )

      tb = TraceBrowser(trace_dir=d)
      events = tb.load_run_events("sess", "r1")
      assert len(events) == 2
      assert all(e["run_id"] == "r1" for e in events)


class TestCorruptFileHandling:
  """Tests for corrupt/malformed JSONL handling."""

  def test_corrupt_line_skipped(self):
    """Corrupt JSON lines should be skipped with warning."""
    with tempfile.TemporaryDirectory() as d:
      p = Path(d) / "sess.jsonl"
      with open(p, "w") as f:
        f.write('{"event": "RunStarted", "run_id": "r1"}\n')
        f.write("this is not json\n")
        f.write('{"event": "RunCompleted", "run_id": "r1"}\n')

      tb = TraceBrowser(trace_dir=d)
      runs = tb.list_runs("sess")
      # Should not crash, should still parse valid lines
      assert len(runs) == 1
      assert runs[0].run_id == "r1"

  def test_session_exists(self):
    """session_exists() should return True for existing sessions."""
    with tempfile.TemporaryDirectory() as d:
      (Path(d) / "exists.jsonl").touch()
      tb = TraceBrowser(trace_dir=d)
      assert tb.session_exists("exists") is True
      assert tb.session_exists("nope") is False
