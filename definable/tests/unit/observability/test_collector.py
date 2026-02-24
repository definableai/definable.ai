"""Tests for ObservabilityExporter (collector) — ring buffer, SSE fan-out, protocol compliance."""

import asyncio
from unittest.mock import MagicMock


from definable.agent.observability.collector import ObservabilityExporter
from definable.agent.tracing.base import TraceExporter


class TestObservabilityExporter:
  """Core collector tests."""

  def test_trace_exporter_protocol(self):
    """ObservabilityExporter must satisfy the TraceExporter protocol."""
    exporter = ObservabilityExporter(buffer_size=10)
    assert isinstance(exporter, TraceExporter)

  def test_export_appends_to_buffer(self):
    """export() should append event dict to the ring buffer."""
    exporter = ObservabilityExporter(buffer_size=100)
    event = MagicMock()
    event.to_dict.return_value = {"event": "RunStarted", "run_id": "r1"}

    exporter.export(event)

    assert len(exporter.buffer) == 1
    assert exporter.buffer[0]["event"] == "RunStarted"
    assert exporter.event_count == 1

  def test_ring_buffer_overflow(self):
    """Buffer should drop oldest events when maxlen exceeded."""
    exporter = ObservabilityExporter(buffer_size=3)

    for i in range(5):
      event = MagicMock()
      event.to_dict.return_value = {"event": "E", "run_id": "r1", "idx": i}
      exporter.export(event)

    assert len(exporter.buffer) == 3
    assert exporter.buffer[0]["idx"] == 2
    assert exporter.buffer[-1]["idx"] == 4
    assert exporter.event_count == 5

  def test_run_indexing(self):
    """Events should be indexed by run_id."""
    exporter = ObservabilityExporter(buffer_size=100)

    for run_id in ["r1", "r1", "r2", "r1"]:
      event = MagicMock()
      event.to_dict.return_value = {"event": "E", "run_id": run_id}
      exporter.export(event)

    assert len(exporter.get_events_for_run("r1")) == 3
    assert len(exporter.get_events_for_run("r2")) == 1
    assert len(exporter.get_events_for_run("nonexistent")) == 0

  def test_get_run_ids(self):
    """get_run_ids() should return all seen run IDs."""
    exporter = ObservabilityExporter(buffer_size=100)

    for run_id in ["r1", "r2", "r1", "r3"]:
      event = MagicMock()
      event.to_dict.return_value = {"event": "E", "run_id": run_id}
      exporter.export(event)

    assert exporter.get_run_ids() == ["r1", "r2", "r3"]

  def test_get_recent_events(self):
    """get_recent_events() should return the most recent N events."""
    exporter = ObservabilityExporter(buffer_size=100)

    for i in range(10):
      event = MagicMock()
      event.to_dict.return_value = {"event": "E", "idx": i}
      exporter.export(event)

    recent = exporter.get_recent_events(limit=3)
    assert len(recent) == 3
    assert recent[0]["idx"] == 7
    assert recent[-1]["idx"] == 9

  def test_get_recent_events_less_than_limit(self):
    """get_recent_events() returns all if fewer than limit."""
    exporter = ObservabilityExporter(buffer_size=100)

    event = MagicMock()
    event.to_dict.return_value = {"event": "E", "idx": 0}
    exporter.export(event)

    recent = exporter.get_recent_events(limit=50)
    assert len(recent) == 1

  def test_flush_is_noop(self):
    """flush() should not raise."""
    exporter = ObservabilityExporter(buffer_size=10)
    exporter.flush()  # no-op

  def test_export_handles_serialization_error(self):
    """export() should log warning on serialization failure, not crash."""
    exporter = ObservabilityExporter(buffer_size=10)
    event = MagicMock()
    event.to_dict.side_effect = RuntimeError("serialize fail")

    exporter.export(event)  # Should not raise
    assert len(exporter.buffer) == 0

  def test_export_adds_obs_timestamp(self):
    """Exported events should have an _obs_ts field."""
    exporter = ObservabilityExporter(buffer_size=10)
    event = MagicMock()
    event.to_dict.return_value = {"event": "RunStarted"}

    exporter.export(event)
    assert "_obs_ts" in exporter.buffer[0]
    assert isinstance(exporter.buffer[0]["_obs_ts"], float)


class TestSSEFanOut:
  """SSE subscription and fan-out tests."""

  def test_subscribe_creates_queue(self):
    """subscribe() should add a client queue."""
    exporter = ObservabilityExporter(buffer_size=10)
    assert exporter.client_count == 0

    q = exporter.subscribe()
    assert exporter.client_count == 1
    assert isinstance(q, asyncio.Queue)

  def test_unsubscribe_removes_queue(self):
    """unsubscribe() should remove the client queue."""
    exporter = ObservabilityExporter(buffer_size=10)
    q = exporter.subscribe()
    assert exporter.client_count == 1

    exporter.unsubscribe(q)
    assert exporter.client_count == 0

  def test_export_fans_out_to_clients(self):
    """export() should put events on all client queues."""
    exporter = ObservabilityExporter(buffer_size=10)
    q1 = exporter.subscribe()
    q2 = exporter.subscribe()

    event = MagicMock()
    event.to_dict.return_value = {"event": "E", "run_id": "r1"}
    exporter.export(event)

    assert not q1.empty()
    assert not q2.empty()
    evt1 = q1.get_nowait()
    evt2 = q2.get_nowait()
    assert evt1 is not None and evt1["event"] == "E"
    assert evt2 is not None and evt2["event"] == "E"

  def test_fan_out_drops_when_queue_full(self):
    """If a client queue is full, the event should be dropped for that client."""
    exporter = ObservabilityExporter(buffer_size=100)
    q = exporter.subscribe(maxsize=1)

    # Fill the queue
    event1 = MagicMock()
    event1.to_dict.return_value = {"event": "E1"}
    exporter.export(event1)

    # This should not raise even though queue is full
    event2 = MagicMock()
    event2.to_dict.return_value = {"event": "E2"}
    exporter.export(event2)

    # Queue should have only the first event
    assert q.qsize() == 1
    evt = q.get_nowait()
    assert evt is not None and evt["event"] == "E1"

  def test_shutdown_sends_sentinel(self):
    """shutdown() should send None sentinel to all clients."""
    exporter = ObservabilityExporter(buffer_size=10)
    q = exporter.subscribe()

    exporter.shutdown()
    assert q.get_nowait() is None
    assert exporter.client_count == 0
