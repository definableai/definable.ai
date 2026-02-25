"""Tests for UsageTracker and UsageSnapshot."""

import pytest

from definable.agent.usage import UsageSnapshot, UsageTracker
from definable.model.metrics import Metrics


class TestUsageSnapshot:
  def test_defaults(self):
    s = UsageSnapshot()
    assert s.input_tokens == 0
    assert s.output_tokens == 0
    assert s.total_tokens == 0
    assert s.estimated_cost == 0.0
    assert s.runs == 0

  def test_add(self):
    a = UsageSnapshot(input_tokens=10, output_tokens=5, total_tokens=15, estimated_cost=0.01, runs=1)
    b = UsageSnapshot(input_tokens=20, output_tokens=10, total_tokens=30, estimated_cost=0.02, runs=1)
    c = a + b
    assert c.input_tokens == 30
    assert c.output_tokens == 15
    assert c.total_tokens == 45
    assert abs(c.estimated_cost - 0.03) < 1e-6
    assert c.runs == 2

  def test_to_dict(self):
    s = UsageSnapshot(input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost=0.005, runs=1, model_id="gpt-4o")
    d = s.to_dict()
    assert d["input_tokens"] == 100
    assert d["model_id"] == "gpt-4o"
    assert d["runs"] == 1

  def test_str(self):
    s = UsageSnapshot(total_tokens=1500, estimated_cost=0.0045, runs=3)
    text = str(s)
    assert "1500" in text
    assert "3 runs" in text


class TestUsageTracker:
  def test_record_run(self):
    tracker = UsageTracker()
    metrics = Metrics(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.01)
    snapshot = tracker.record_run(metrics, model_id="gpt-4o-mini")
    assert snapshot.input_tokens == 100
    assert snapshot.output_tokens == 50
    assert snapshot.model_id == "gpt-4o-mini"

  def test_session_total_accumulates(self):
    tracker = UsageTracker()
    tracker.record_run(Metrics(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.01))
    tracker.record_run(Metrics(input_tokens=200, output_tokens=100, total_tokens=300, cost=0.02))
    total = tracker.session_total
    assert total.input_tokens == 300
    assert total.output_tokens == 150
    assert total.total_tokens == 450
    assert abs(total.estimated_cost - 0.03) < 1e-6
    assert total.runs == 2

  def test_last_run(self):
    tracker = UsageTracker()
    assert tracker.last_run is None
    tracker.record_run(Metrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.001))
    assert tracker.last_run is not None
    assert tracker.last_run.input_tokens == 10

  def test_run_count(self):
    tracker = UsageTracker()
    assert tracker.run_count == 0
    tracker.record_run(Metrics(input_tokens=10, output_tokens=5, total_tokens=15))
    assert tracker.run_count == 1

  def test_all_runs(self):
    tracker = UsageTracker()
    tracker.record_run(Metrics(input_tokens=10, output_tokens=5, total_tokens=15))
    tracker.record_run(Metrics(input_tokens=20, output_tokens=10, total_tokens=30))
    assert len(tracker.all_runs) == 2

  def test_reset(self):
    tracker = UsageTracker()
    tracker.record_run(Metrics(input_tokens=100, output_tokens=50, total_tokens=150))
    tracker.reset()
    assert tracker.run_count == 0
    assert tracker.session_total.total_tokens == 0

  def test_disabled(self):
    tracker = UsageTracker(enabled=False)
    snapshot = tracker.record_run(Metrics(input_tokens=100, output_tokens=50, total_tokens=150))
    assert snapshot.total_tokens == 0
    assert tracker.run_count == 0

  def test_none_metrics(self):
    tracker = UsageTracker()
    snapshot = tracker.record_run(None)
    assert snapshot.total_tokens == 0

  @pytest.mark.asyncio
  async def test_async_record(self):
    tracker = UsageTracker()
    snapshot = await tracker.arecord_run(Metrics(input_tokens=50, output_tokens=25, total_tokens=75, cost=0.005))
    assert snapshot.total_tokens == 75
    assert tracker.session_total.total_tokens == 75
