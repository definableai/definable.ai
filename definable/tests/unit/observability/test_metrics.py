"""Tests for MetricsAggregator — aggregation accuracy, time-series, per-tool/model stats."""

import time


from definable.agent.observability.metrics import MetricsAggregator, MetricsSnapshot


class TestMetricsAggregator:
  """Core metrics computation tests."""

  def setup_method(self):
    self.agg = MetricsAggregator()

  def test_zero_events(self):
    """Zero events should produce zero metrics."""
    snap = self.agg.compute([])
    assert snap.total_runs == 0
    assert snap.success_rate == 0.0
    assert snap.total_tokens == 0
    assert snap.total_cost == 0.0
    assert snap.avg_duration_ms == 0.0
    assert snap.latency_p50_ms == 0.0

  def test_completed_runs(self):
    """Should count completed runs and aggregate tokens/cost."""
    events = [
      {
        "event": "RunCompleted",
        "run_id": "r1",
        "metrics": {"total_tokens": 100, "input_tokens": 60, "output_tokens": 40, "cost": 0.01, "duration": 1.5},
      },
      {
        "event": "RunCompleted",
        "run_id": "r2",
        "metrics": {"total_tokens": 200, "input_tokens": 120, "output_tokens": 80, "cost": 0.02, "duration": 2.5},
      },
      {
        "event": "RunCompleted",
        "run_id": "r3",
        "metrics": {"total_tokens": 150, "input_tokens": 90, "output_tokens": 60, "cost": 0.015, "duration": 1.0},
      },
    ]
    snap = self.agg.compute(events)

    assert snap.total_runs == 3
    assert snap.completed_runs == 3
    assert snap.success_rate == 1.0
    assert snap.total_tokens == 450
    assert snap.total_input_tokens == 270
    assert snap.total_output_tokens == 180
    assert abs(snap.total_cost - 0.045) < 1e-6
    assert snap.avg_tokens_per_run == 150.0

  def test_mixed_status_runs(self):
    """Should correctly count runs by status."""
    events: list[dict[str, object]] = [
      {"event": "RunCompleted", "run_id": "r1", "metrics": {"total_tokens": 100, "cost": 0.01, "duration": 1.0}},
      {"event": "RunCompleted", "run_id": "r2", "metrics": {"total_tokens": 100, "cost": 0.01, "duration": 1.0}},
      {"event": "RunError", "run_id": "r3"},
      {"event": "RunCancelled", "run_id": "r4"},
      {"event": "RunPaused", "run_id": "r5"},
    ]
    snap = self.agg.compute(events)  # type: ignore[arg-type]

    assert snap.total_runs == 5
    assert snap.completed_runs == 2
    assert snap.error_runs == 1
    assert snap.cancelled_runs == 1
    assert snap.paused_runs == 1
    assert snap.success_rate == 0.4  # 2/5

  def test_latency_percentiles(self):
    """Should compute P50, P95, P99 from durations."""
    events = [
      {"event": "RunCompleted", "run_id": f"r{i}", "metrics": {"total_tokens": 10, "cost": 0.001, "duration": float(i) / 10.0}}
      for i in range(1, 101)  # 100 runs, durations 0.1s to 10.0s
    ]
    snap = self.agg.compute(events)

    # P50 should be around 5000ms (5.0s * 1000)
    assert 4500 < snap.latency_p50_ms < 5500
    # P95 should be around 9500ms
    assert 9000 < snap.latency_p95_ms < 10000

  def test_per_tool_stats(self):
    """Should aggregate per-tool call counts and errors."""
    events = [
      {"event": "ToolCallCompleted", "tool": {"tool_name": "search", "duration_ms": 100}},
      {"event": "ToolCallCompleted", "tool": {"tool_name": "search", "duration_ms": 200}},
      {"event": "ToolCallCompleted", "tool": {"tool_name": "calculate", "duration_ms": 50}},
      {"event": "ToolCallCompleted", "tool": {"tool_name": "search", "tool_call_error": True, "duration_ms": 150}},
    ]
    snap = self.agg.compute(events)

    assert len(snap.tool_stats) == 2
    search = next(ts for ts in snap.tool_stats if ts.tool_name == "search")
    assert search.call_count == 3
    assert search.error_count == 1
    assert abs(search.avg_duration_ms - 150.0) < 1e-6  # (100+200+150)/3

    calc = next(ts for ts in snap.tool_stats if ts.tool_name == "calculate")
    assert calc.call_count == 1
    assert calc.error_count == 0

  def test_per_model_stats(self):
    """Should aggregate per-model token/cost usage."""
    events = [
      {"event": "ModelCallCompleted", "model_id": "gpt-4o", "metrics": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.01}},
      {
        "event": "ModelCallCompleted",
        "model_id": "gpt-4o",
        "metrics": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "cost": 0.02},
      },
      {"event": "ModelCallCompleted", "model_id": "gpt-3.5", "metrics": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75, "cost": 0.001}},
    ]
    snap = self.agg.compute(events)

    assert len(snap.model_stats) == 2
    gpt4 = next(ms for ms in snap.model_stats if ms.model_id == "gpt-4o")
    assert gpt4.call_count == 2
    assert gpt4.total_tokens == 450
    assert abs(gpt4.avg_tokens - 225.0) < 1e-6

  def test_metrics_snapshot_to_dict(self):
    """to_dict() should return a serializable dict."""
    snap = MetricsSnapshot(
      total_runs=5,
      completed_runs=3,
      success_rate=0.6,
      total_tokens=1000,
    )
    d = snap.to_dict()
    assert d["total_runs"] == 5
    assert d["success_rate"] == 0.6
    assert isinstance(d["tool_stats"], list)
    assert isinstance(d["model_stats"], list)

  def test_no_division_by_zero(self):
    """Should handle edge cases without division by zero."""
    events = [{"event": "RunError", "run_id": "r1"}]
    snap = self.agg.compute(events)
    assert snap.total_runs == 1
    assert snap.avg_tokens_per_run == 0.0
    assert snap.avg_cost_per_run == 0.0
    assert snap.avg_duration_ms == 0.0

  def test_missing_metrics_fields(self):
    """Should handle events with missing or None metrics fields."""
    events = [
      {"event": "RunCompleted", "run_id": "r1", "metrics": {"total_tokens": None, "cost": None, "duration": None}},
    ]
    snap = self.agg.compute(events)
    assert snap.total_runs == 1
    assert snap.total_tokens == 0
    assert snap.total_cost == 0.0


class TestTimeline:
  """Time-series bucketing tests."""

  def test_timeline_returns_buckets(self):
    """Should return time-series buckets."""
    now = time.time()
    events = [
      {"event": "RunCompleted", "run_id": "r1", "created_at": now - 100, "metrics": {"total_tokens": 50, "cost": 0.001}},
      {"event": "RunCompleted", "run_id": "r2", "created_at": now - 50, "metrics": {"total_tokens": 100, "cost": 0.002}},
    ]

    agg = MetricsAggregator()
    buckets = agg.compute_timeline(events, bucket="hour", range_hours=1)

    assert len(buckets) >= 1
    total_runs = sum(b.run_count for b in buckets)
    assert total_runs == 2

  def test_events_in_same_hour_same_bucket(self):
    """Events within the same hour should land in the same bucket."""
    now = time.time()
    events = [
      {"event": "RunCompleted", "run_id": "r1", "created_at": now - 10, "metrics": {"total_tokens": 50, "cost": 0.001}},
      {"event": "RunCompleted", "run_id": "r2", "created_at": now - 5, "metrics": {"total_tokens": 100, "cost": 0.002}},
    ]

    agg = MetricsAggregator()
    buckets = agg.compute_timeline(events, bucket="hour", range_hours=1)

    # Last bucket should have both runs
    active_buckets = [b for b in buckets if b.run_count > 0]
    assert len(active_buckets) == 1
    assert active_buckets[0].run_count == 2
