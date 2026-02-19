"""Unit tests for JSONLExporter and the Tracing block.

Tests cover creation, file writing, session management,
context manager usage, and the Tracing dataclass. No real
agent events are used; lightweight stubs are substituted.
"""

import json
from typing import Any

import pytest
from unittest.mock import MagicMock

from definable.agent.tracing.base import NoOpExporter, Tracing, TraceWriter
from definable.agent.tracing.jsonl import JSONLExporter


class _StubEvent:
  """Minimal stub that mimics BaseRunOutputEvent for export."""

  def __init__(self, session_id: str = "default", data: dict | None = None):
    self.session_id: str | None = session_id
    self._data = data or {"event": "TestEvent", "run_id": "r1"}

  def to_json(self, indent: int | None = None) -> str:
    return json.dumps(self._data)


def _stub(session_id: str = "default", data: dict | None = None) -> Any:
  """Create a _StubEvent typed as Any to satisfy export/write signatures."""
  return _StubEvent(session_id=session_id, data=data)


@pytest.mark.unit
class TestJSONLExporterCreation:
  """Tests for JSONLExporter instantiation."""

  def test_creates_trace_directory(self, tmp_path):
    """JSONLExporter creates the trace_dir on init."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))
    assert trace_dir.is_dir()
    exporter.shutdown()

  def test_default_settings(self, tmp_path):
    """Default flush_each=True and mirror_stdout=True."""
    exporter = JSONLExporter(str(tmp_path / "t"))
    assert exporter.flush_each is True
    assert exporter.mirror_stdout is True
    exporter.shutdown()

  def test_open_sessions_starts_at_zero(self, tmp_path):
    """No file handles are open initially."""
    exporter = JSONLExporter(str(tmp_path / "t"))
    assert exporter.open_sessions == 0
    exporter.shutdown()


@pytest.mark.unit
class TestJSONLExporterWrite:
  """Tests for JSONLExporter.export() writing valid JSONL."""

  def test_writes_single_event(self, tmp_path):
    """export() writes one JSON line to the session file."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))

    event = _stub(session_id="sess1", data={"event": "RunStarted", "run_id": "abc"})
    exporter.export(event)
    exporter.shutdown()

    file_path = trace_dir / "sess1.jsonl"
    assert file_path.exists()
    lines = file_path.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "RunStarted"
    assert parsed["run_id"] == "abc"

  def test_writes_multiple_events(self, tmp_path):
    """Multiple export() calls produce multiple JSONL lines."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))

    for i in range(3):
      event = _stub(session_id="multi", data={"event": f"Event{i}", "run_id": "r"})
      exporter.export(event)
    exporter.shutdown()

    lines = (trace_dir / "multi.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3

  def test_default_session_id(self, tmp_path):
    """Events without session_id go to 'default.jsonl'."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))

    event = _stub()
    # Simulate missing session_id via attribute override
    event.session_id = None
    exporter.export(event)
    exporter.shutdown()

    assert (trace_dir / "default.jsonl").exists()

  def test_separate_files_per_session(self, tmp_path):
    """Different session_ids write to different files."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))

    exporter.export(_stub(session_id="a"))
    exporter.export(_stub(session_id="b"))
    exporter.shutdown()

    assert (trace_dir / "a.jsonl").exists()
    assert (trace_dir / "b.jsonl").exists()
    assert exporter.open_sessions == 0  # after shutdown


@pytest.mark.unit
class TestJSONLExporterLifecycle:
  """Tests for flush, shutdown, context manager."""

  def test_flush_does_not_raise(self, tmp_path):
    """flush() on an exporter with no open handles does not raise."""
    exporter = JSONLExporter(str(tmp_path / "t"))
    exporter.flush()
    exporter.shutdown()

  def test_shutdown_closes_handles(self, tmp_path):
    """shutdown() closes all file handles."""
    exporter = JSONLExporter(str(tmp_path / "t"))
    exporter.export(_stub(session_id="s1"))
    assert exporter.open_sessions == 1
    exporter.shutdown()
    assert exporter.open_sessions == 0

  def test_context_manager(self, tmp_path):
    """Using as context manager calls shutdown on exit."""
    trace_dir = tmp_path / "traces"
    with JSONLExporter(str(trace_dir)) as exporter:
      exporter.export(_stub(session_id="ctx"))
      assert exporter.open_sessions == 1
    # After exit, handles should be closed
    assert exporter.open_sessions == 0

  def test_get_trace_path(self, tmp_path):
    """get_trace_path returns the expected file path."""
    trace_dir = tmp_path / "traces"
    exporter = JSONLExporter(str(trace_dir))
    path = exporter.get_trace_path("my_session")
    assert path == trace_dir / "my_session.jsonl"
    exporter.shutdown()


@pytest.mark.unit
class TestTracingBlock:
  """Tests for the Tracing dataclass."""

  def test_default_enabled(self):
    """Tracing is enabled by default."""
    t = Tracing()
    assert t.enabled is True

  def test_default_exporters_none(self):
    """Exporters default to None."""
    t = Tracing()
    assert t.exporters is None

  def test_custom_settings(self):
    """Tracing accepts custom batch_size and flush_interval_ms."""
    t = Tracing(batch_size=10, flush_interval_ms=1000)
    assert t.batch_size == 10
    assert t.flush_interval_ms == 1000

  def test_disabled_tracing(self):
    """Tracing can be disabled."""
    t = Tracing(enabled=False)
    assert t.enabled is False


@pytest.mark.unit
class TestTraceWriter:
  """Tests for TraceWriter orchestration."""

  def test_write_dispatches_to_exporters(self, tmp_path):
    """TraceWriter.write() calls export on each exporter."""
    mock_exporter = MagicMock()
    mock_exporter.export = MagicMock()
    mock_exporter.flush = MagicMock()
    mock_exporter.shutdown = MagicMock()

    config = Tracing(exporters=[mock_exporter])
    writer = TraceWriter(config)
    event = _stub()
    writer.write(event)

    mock_exporter.export.assert_called_once_with(event)

  def test_write_skipped_when_disabled(self):
    """TraceWriter.write() is a no-op when tracing is disabled."""
    mock_exporter = MagicMock()
    config = Tracing(enabled=False, exporters=[mock_exporter])
    writer = TraceWriter(config)
    writer.write(_stub())

    mock_exporter.export.assert_not_called()

  def test_event_filter_blocks_event(self):
    """TraceWriter respects event_filter; filtered events are not exported."""
    mock_exporter = MagicMock()
    config = Tracing(
      exporters=[mock_exporter],
      event_filter=lambda e: False,
    )
    writer = TraceWriter(config)
    writer.write(_stub())

    mock_exporter.export.assert_not_called()

  def test_event_filter_allows_event(self):
    """TraceWriter exports events that pass the filter."""
    mock_exporter = MagicMock()
    config = Tracing(
      exporters=[mock_exporter],
      event_filter=lambda e: True,
    )
    writer = TraceWriter(config)
    writer.write(_stub())

    mock_exporter.export.assert_called_once()

  def test_exporter_count(self):
    """exporter_count reflects the number of configured exporters."""
    config = Tracing(exporters=[MagicMock(), MagicMock()])
    writer = TraceWriter(config)
    assert writer.exporter_count == 2

  def test_add_exporter(self):
    """add_exporter increases the exporter count."""
    config = Tracing(exporters=[])
    writer = TraceWriter(config)
    assert writer.exporter_count == 0
    writer.add_exporter(MagicMock())
    assert writer.exporter_count == 1

  def test_remove_exporter(self):
    """remove_exporter decreases the count and returns True."""
    mock = MagicMock()
    config = Tracing(exporters=[mock])
    writer = TraceWriter(config)
    assert writer.remove_exporter(mock) is True
    assert writer.exporter_count == 0

  def test_remove_nonexistent_exporter(self):
    """remove_exporter returns False for an unknown exporter."""
    config = Tracing(exporters=[])
    writer = TraceWriter(config)
    assert writer.remove_exporter(MagicMock()) is False

  def test_shutdown_calls_flush_and_shutdown_on_exporters(self):
    """shutdown() flushes and shuts down each exporter."""
    mock = MagicMock()
    config = Tracing(exporters=[mock])
    writer = TraceWriter(config)
    writer.shutdown()

    mock.flush.assert_called_once()
    mock.shutdown.assert_called_once()


@pytest.mark.unit
class TestNoOpExporter:
  """Tests for the NoOpExporter."""

  def test_export_does_not_raise(self):
    """NoOpExporter.export() silently discards the event."""
    noop = NoOpExporter()
    noop.export(_stub())

  def test_flush_does_not_raise(self):
    """NoOpExporter.flush() is a no-op."""
    noop = NoOpExporter()
    noop.flush()

  def test_shutdown_does_not_raise(self):
    """NoOpExporter.shutdown() is a no-op."""
    noop = NoOpExporter()
    noop.shutdown()
