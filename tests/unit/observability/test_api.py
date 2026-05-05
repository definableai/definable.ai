"""Tests for observability API router — endpoint responses, SSE, filtering, error handling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from definable.agent.observability.collector import ObservabilityExporter
from definable.agent.observability.metrics import MetricsAggregator
from definable.agent.observability.trace_browser import TraceBrowser


def _write_events(path: Path, events: list):
  """Write events to a JSONL file."""
  with open(path, "w", encoding="utf-8") as f:
    for evt in events:
      f.write(json.dumps(evt) + "\n")


def _make_router(*, trace_dir: str | None = None, buffer_size: int = 100):
  """Create router components for testing."""
  if trace_dir is None:
    trace_dir = tempfile.mkdtemp()
  collector = ObservabilityExporter(buffer_size=buffer_size)
  trace_browser = TraceBrowser(trace_dir=trace_dir)
  metrics_agg = MetricsAggregator()
  from definable.agent.observability.api import create_observability_router

  router = create_observability_router(
    collector=collector,
    trace_browser=trace_browser,
    metrics_aggregator=metrics_agg,
  )
  return router, collector, trace_browser, metrics_agg


@pytest.fixture
def app_client():
  """Create a FastAPI TestClient with the observability router mounted."""
  from fastapi import FastAPI
  from starlette.testclient import TestClient

  with tempfile.TemporaryDirectory() as d:
    router, collector, tb, ma = _make_router(trace_dir=d)
    app = FastAPI()
    app.include_router(router, prefix="/obs/api")
    client = TestClient(app)
    yield client, collector, tb, d


class TestHealthEndpoint:
  """Tests for GET /obs/api/health."""

  def test_health_returns_ok(self, app_client):
    """Health endpoint should return 200 with status ok."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "observability"

  def test_health_reflects_buffer_state(self, app_client):
    """Health should show buffered event count."""
    client, collector, _, _ = app_client

    # Add an event to the buffer
    evt = MagicMock()
    evt.to_dict.return_value = {"event": "RunStarted", "run_id": "r1"}
    collector.export(evt)

    resp = client.get("/obs/api/health")
    data = resp.json()
    assert data["buffered_events"] == "1"


class TestSessionsEndpoint:
  """Tests for GET /obs/api/sessions."""

  def test_empty_sessions(self, app_client):
    """Should return empty session list for empty trace dir."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessions"] == []
    assert data["total"] == 0

  def test_lists_sessions_from_trace_dir(self, app_client):
    """Should list JSONL files as sessions."""
    client, _, _, trace_dir = app_client
    _write_events(Path(trace_dir) / "sess1.jsonl", [{"event": "RunStarted", "run_id": "r1"}])
    _write_events(Path(trace_dir) / "sess2.jsonl", [{"event": "RunStarted", "run_id": "r2"}])

    resp = client.get("/obs/api/sessions")
    data = resp.json()
    assert data["total"] == 2
    assert len(data["sessions"]) == 2

  def test_pagination(self, app_client):
    """Should support limit and offset pagination."""
    client, _, _, trace_dir = app_client
    for i in range(5):
      _write_events(Path(trace_dir) / f"sess{i}.jsonl", [{"event": "RunStarted", "run_id": f"r{i}"}])

    resp = client.get("/obs/api/sessions?limit=2&offset=0")
    data = resp.json()
    assert len(data["sessions"]) == 2
    assert data["total"] == 5
    assert data["limit"] == 2
    assert data["offset"] == 0


class TestSessionRunsEndpoint:
  """Tests for GET /obs/api/sessions/{session_id}."""

  def test_lists_runs_for_session(self, app_client):
    """Should list runs within a session."""
    client, _, _, trace_dir = app_client
    _write_events(
      Path(trace_dir) / "sess.jsonl",
      [
        {"event": "RunStarted", "run_id": "r1", "agent_name": "test"},
        {"event": "RunCompleted", "run_id": "r1", "metrics": {"total_tokens": 100, "cost": 0.01, "duration": 1.0}},
        {"event": "RunStarted", "run_id": "r2", "agent_name": "test"},
        {"event": "RunError", "run_id": "r2"},
      ],
    )

    resp = client.get("/obs/api/sessions/sess")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess"
    assert len(data["runs"]) == 2

  def test_missing_session_returns_404(self, app_client):
    """Should return 404 for nonexistent session."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/sessions/nonexistent")
    assert resp.status_code == 404
    data = resp.json()
    assert "not found" in data["error"].lower()


class TestRunReplayEndpoint:
  """Tests for GET /obs/api/runs/{run_id}."""

  def test_run_not_found(self, app_client):
    """Should return 404 for nonexistent run."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/runs/nonexistent")
    assert resp.status_code == 404

  def test_run_from_live_buffer(self, app_client):
    """Should return events from live buffer when available."""
    client, collector, _, _ = app_client

    # Add events to the buffer
    for evt_data in [
      {"event": "RunStarted", "run_id": "live-run"},
      {"event": "RunCompleted", "run_id": "live-run", "metrics": {"total_tokens": 50}},
    ]:
      evt = MagicMock()
      evt.to_dict.return_value = evt_data
      collector.export(evt)

    resp = client.get("/obs/api/runs/live-run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "live-run"


class TestMetricsEndpoint:
  """Tests for GET /obs/api/metrics."""

  def test_empty_metrics(self, app_client):
    """Should return zero metrics for empty buffer."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_runs"] == 0
    assert data["total_tokens"] == 0

  def test_metrics_from_buffer(self, app_client):
    """Should compute metrics from buffered events."""
    client, collector, _, _ = app_client

    events = [
      {"event": "RunCompleted", "run_id": "r1", "metrics": {"total_tokens": 100, "cost": 0.01, "duration": 1.0}},
      {"event": "RunCompleted", "run_id": "r2", "metrics": {"total_tokens": 200, "cost": 0.02, "duration": 2.0}},
      {"event": "RunError", "run_id": "r3"},
    ]
    for evt_data in events:
      evt = MagicMock()
      evt.to_dict.return_value = evt_data
      collector.export(evt)

    resp = client.get("/obs/api/metrics")
    data = resp.json()
    assert data["total_runs"] == 3
    assert data["completed_runs"] == 2
    assert data["total_tokens"] == 300


class TestTimelineEndpoint:
  """Tests for GET /obs/api/metrics/timeline."""

  def test_timeline_returns_buckets(self, app_client):
    """Should return time-series buckets."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/metrics/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bucket"] in ("5min", "30min", "hour", "day")
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "bucket_seconds" in data
    assert "range_hours" in data

  def test_timeline_range_param(self, app_client):
    """Should accept range parameter."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/metrics/timeline?range=7d")
    assert resp.status_code == 200
    data = resp.json()
    assert data["range"] == "7d"


class TestEventsExportEndpoint:
  """Tests for GET /obs/api/events/export/{session_id}."""

  def test_export_existing_session(self, app_client):
    """Should stream session JSONL for download."""
    client, _, _, trace_dir = app_client
    events = [{"event": "RunStarted", "run_id": "r1"}, {"event": "RunCompleted", "run_id": "r1"}]
    _write_events(Path(trace_dir) / "sess.jsonl", events)

    resp = client.get("/obs/api/events/export/sess")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-ndjson"
    assert "attachment" in resp.headers.get("content-disposition", "")

    # Verify content is valid JSONL
    lines = resp.text.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "RunStarted"

  def test_export_missing_session_returns_404(self, app_client):
    """Should return 404 for nonexistent session export."""
    client, _, _, _ = app_client
    resp = client.get("/obs/api/events/export/nonexistent")
    assert resp.status_code == 404


class TestSSEEventsEndpoint:
  """Tests for SSE subscriber/unsubscribe mechanics (tested via collector)."""

  def test_sse_subscribe_creates_queue(self, app_client):
    """SSE subscription should work at the collector level."""
    _, collector, _, _ = app_client
    assert collector.client_count == 0
    q = collector.subscribe()
    assert collector.client_count == 1
    collector.unsubscribe(q)
    assert collector.client_count == 0


class TestMatchesFilter:
  """Tests for the _matches_filter helper."""

  def test_no_filters(self):
    """No filters means everything matches."""
    from definable.agent.observability.api import _matches_filter

    assert _matches_filter({"event": "RunStarted"}) is True

  def test_event_type_filter(self):
    """Should filter by event type."""
    from definable.agent.observability.api import _matches_filter

    assert _matches_filter({"event": "RunStarted"}, event_type="RunStarted") is True
    assert _matches_filter({"event": "RunStarted"}, event_type="RunCompleted") is False

  def test_run_id_filter(self):
    """Should filter by run_id."""
    from definable.agent.observability.api import _matches_filter

    assert _matches_filter({"run_id": "r1"}, run_id="r1") is True
    assert _matches_filter({"run_id": "r1"}, run_id="r2") is False

  def test_session_id_filter(self):
    """Should filter by session_id."""
    from definable.agent.observability.api import _matches_filter

    assert _matches_filter({"session_id": "s1"}, session_id="s1") is True
    assert _matches_filter({"session_id": "s1"}, session_id="s2") is False

  def test_combined_filters(self):
    """Should match all filters simultaneously."""
    from definable.agent.observability.api import _matches_filter

    evt = {"event": "RunStarted", "run_id": "r1", "session_id": "s1"}
    assert _matches_filter(evt, event_type="RunStarted", run_id="r1", session_id="s1") is True
    assert _matches_filter(evt, event_type="RunStarted", run_id="r2") is False
