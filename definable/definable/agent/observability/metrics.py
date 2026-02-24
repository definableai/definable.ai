"""Metrics aggregation — computed on-the-fly from collector buffer and JSONL traces."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolStats:
  """Per-tool aggregated statistics."""

  tool_name: str = ""
  call_count: int = 0
  error_count: int = 0
  total_duration_ms: float = 0.0
  avg_duration_ms: float = 0.0


@dataclass
class ModelStats:
  """Per-model aggregated statistics."""

  model_id: str = ""
  call_count: int = 0
  total_input_tokens: int = 0
  total_output_tokens: int = 0
  total_tokens: int = 0
  total_cost: float = 0.0
  avg_tokens: float = 0.0
  avg_cost: float = 0.0


@dataclass
class RecentRun:
  """Summary of a single run for the overview page."""

  run_id: str = ""
  session_id: str = ""
  status: str = "PENDING"
  tokens: int = 0
  cost: float = 0.0
  duration: float = 0.0
  agent_name: str = ""
  model: str = ""
  created_at: float = 0.0


@dataclass
class TimeSeriesBucket:
  """A single time-series bucket."""

  timestamp: float = 0.0
  label: str = ""
  run_count: int = 0
  error_count: int = 0
  tokens: int = 0
  cost: float = 0.0


@dataclass
class MetricsSnapshot:
  """Aggregated metrics snapshot returned by the API."""

  total_runs: int = 0
  completed_runs: int = 0
  error_runs: int = 0
  paused_runs: int = 0
  cancelled_runs: int = 0
  success_rate: float = 0.0

  total_input_tokens: int = 0
  total_output_tokens: int = 0
  total_tokens: int = 0
  total_cost: float = 0.0

  avg_tokens_per_run: float = 0.0
  avg_cost_per_run: float = 0.0
  avg_duration_ms: float = 0.0

  latency_p50_ms: float = 0.0
  latency_p95_ms: float = 0.0
  latency_p99_ms: float = 0.0

  tool_stats: List[ToolStats] = field(default_factory=list)
  model_stats: List[ModelStats] = field(default_factory=list)
  recent_runs: List[RecentRun] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to dict for JSON response."""
    return {
      "total_runs": self.total_runs,
      "completed_runs": self.completed_runs,
      "error_runs": self.error_runs,
      "paused_runs": self.paused_runs,
      "cancelled_runs": self.cancelled_runs,
      "success_rate": round(self.success_rate, 4),
      "total_input_tokens": self.total_input_tokens,
      "total_output_tokens": self.total_output_tokens,
      "total_tokens": self.total_tokens,
      "total_cost": round(self.total_cost, 6),
      "avg_tokens_per_run": round(self.avg_tokens_per_run, 1),
      "avg_cost_per_run": round(self.avg_cost_per_run, 6),
      "avg_duration_ms": round(self.avg_duration_ms, 1),
      "latency_p50_ms": round(self.latency_p50_ms, 1),
      "latency_p95_ms": round(self.latency_p95_ms, 1),
      "latency_p99_ms": round(self.latency_p99_ms, 1),
      "tool_stats": [
        {
          "tool_name": ts.tool_name,
          "call_count": ts.call_count,
          "error_count": ts.error_count,
          "total_duration_ms": round(ts.total_duration_ms, 1),
          "avg_duration_ms": round(ts.avg_duration_ms, 1),
        }
        for ts in self.tool_stats
      ],
      "model_stats": [
        {
          "model_id": ms.model_id,
          "call_count": ms.call_count,
          "total_input_tokens": ms.total_input_tokens,
          "total_output_tokens": ms.total_output_tokens,
          "total_tokens": ms.total_tokens,
          "total_cost": round(ms.total_cost, 6),
          "avg_tokens": round(ms.avg_tokens, 1),
          "avg_cost": round(ms.avg_cost, 6),
        }
        for ms in self.model_stats
      ],
      "recent_runs": [
        {
          "run_id": r.run_id,
          "session_id": r.session_id,
          "status": r.status,
          "tokens": r.tokens,
          "cost": round(r.cost, 6),
          "duration": round(r.duration, 3),
          "agent_name": r.agent_name,
          "model": r.model,
          "created_at": r.created_at,
        }
        for r in self.recent_runs
      ],
    }


class MetricsAggregator:
  """Compute aggregated metrics from a list of event dicts.

  Processes RunStarted, RunCompleted, RunError, RunCancelled, RunPaused,
  ToolCallCompleted, and ModelCallCompleted events.
  Stateless — call ``compute()`` with the current event buffer each time.
  """

  def compute(self, events: List[Dict[str, Any]]) -> MetricsSnapshot:
    """Compute a MetricsSnapshot from a list of event dicts.

    Args:
      events: List of event dicts (from collector buffer or JSONL).

    Returns:
      Aggregated MetricsSnapshot.
    """
    snap = MetricsSnapshot()

    # Track per-run data
    run_starts: Dict[str, Dict[str, Any]] = {}
    run_completions: Dict[str, Dict[str, Any]] = {}
    run_errors: set[str] = set()
    durations: List[float] = []

    # Per-tool accumulators
    tool_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "errors": 0, "total_ms": 0.0})

    # Per-model accumulators
    model_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0})

    for evt in events:
      event_type = evt.get("event", "")

      if event_type == "RunStarted":
        run_id = evt.get("run_id")
        if run_id:
          run_starts[run_id] = evt

      elif event_type == "RunCompleted":
        snap.completed_runs += 1
        run_id = evt.get("run_id")
        if run_id:
          run_completions[run_id] = evt

        metrics = evt.get("metrics")
        if metrics:
          inp = _get(metrics, "input_tokens", 0) or 0
          out = _get(metrics, "output_tokens", 0) or 0
          total = _get(metrics, "total_tokens", 0) or 0
          cost = _get(metrics, "cost", 0) or 0
          dur = _get(metrics, "duration")

          snap.total_input_tokens += inp
          snap.total_output_tokens += out
          snap.total_tokens += total
          snap.total_cost += cost

          if dur is not None:
            durations.append(float(dur) * 1000.0)  # seconds to ms

      elif event_type == "RunError":
        snap.error_runs += 1
        run_id = evt.get("run_id")
        if run_id:
          run_errors.add(run_id)

      elif event_type == "RunCancelled":
        snap.cancelled_runs += 1

      elif event_type == "RunPaused":
        snap.paused_runs += 1

      elif event_type == "ToolCallCompleted":
        tool = evt.get("tool") or {}
        name = _get(tool, "tool_name", "unknown")
        td = tool_data[name]
        td["count"] += 1
        if _get(tool, "tool_call_error"):
          td["errors"] += 1
        # Compute duration from created_at delta if available
        dur = _get(tool, "duration_ms")
        if dur is not None:
          td["total_ms"] += float(dur)

      elif event_type == "ModelCallCompleted":
        model_id = evt.get("model_id", "unknown")
        md = model_data[model_id]
        md["count"] += 1
        metrics = evt.get("metrics")
        if metrics:
          md["input_tokens"] += _get(metrics, "input_tokens", 0) or 0
          md["output_tokens"] += _get(metrics, "output_tokens", 0) or 0
          md["total_tokens"] += _get(metrics, "total_tokens", 0) or 0
          md["cost"] += _get(metrics, "cost", 0) or 0

    # Finalize
    snap.total_runs = snap.completed_runs + snap.error_runs + snap.cancelled_runs + snap.paused_runs
    if snap.total_runs > 0:
      snap.success_rate = snap.completed_runs / snap.total_runs
      snap.avg_tokens_per_run = snap.total_tokens / snap.total_runs
      snap.avg_cost_per_run = snap.total_cost / snap.total_runs

    if durations:
      snap.avg_duration_ms = sum(durations) / len(durations)
      snap.latency_p50_ms = _percentile(durations, 50)
      snap.latency_p95_ms = _percentile(durations, 95)
      snap.latency_p99_ms = _percentile(durations, 99)

    # Build tool stats
    for name, td in sorted(tool_data.items()):
      avg_ms = td["total_ms"] / td["count"] if td["count"] > 0 else 0.0
      snap.tool_stats.append(
        ToolStats(
          tool_name=name,
          call_count=td["count"],
          error_count=td["errors"],
          total_duration_ms=td["total_ms"],
          avg_duration_ms=avg_ms,
        )
      )

    # Build model stats
    for model_id, md in sorted(model_data.items()):
      avg_tokens = md["total_tokens"] / md["count"] if md["count"] > 0 else 0.0
      avg_cost = md["cost"] / md["count"] if md["count"] > 0 else 0.0
      snap.model_stats.append(
        ModelStats(
          model_id=model_id,
          call_count=md["count"],
          total_input_tokens=md["input_tokens"],
          total_output_tokens=md["output_tokens"],
          total_tokens=md["total_tokens"],
          total_cost=md["cost"],
          avg_tokens=avg_tokens,
          avg_cost=avg_cost,
        )
      )

    # Build recent runs — merge RunStarted metadata with RunCompleted metrics
    all_run_ids = set(run_starts.keys()) | set(run_completions.keys()) | set(run_errors)
    recent: List[RecentRun] = []
    for rid in all_run_ids:
      start_evt = run_starts.get(rid, {})
      comp_evt = run_completions.get(rid)
      metrics = _get(comp_evt or {}, "metrics") or {}
      if comp_evt:
        status = "COMPLETED"
      elif rid in run_errors:
        status = "ERROR"
      else:
        status = "RUNNING"
      recent.append(
        RecentRun(
          run_id=rid,
          session_id=start_evt.get("session_id", ""),
          status=status,
          tokens=_get(metrics, "total_tokens", 0) or 0,
          cost=_get(metrics, "cost", 0) or 0,
          duration=_get(metrics, "duration", 0) or 0,
          agent_name=start_evt.get("agent_name", ""),
          model=start_evt.get("model", ""),
          created_at=start_evt.get("created_at", 0) or 0,
        )
      )
    # Sort newest first, keep top 20
    recent.sort(key=lambda r: r.created_at, reverse=True)
    snap.recent_runs = recent[:20]

    return snap

  def compute_timeline(
    self,
    events: List[Dict[str, Any]],
    *,
    bucket: str = "hour",
    range_hours: float = 24,
  ) -> "TimelineResult":
    """Compute time-series buckets from events.

    Args:
      events: List of event dicts.
      bucket: Bucket granularity — ``"5min"``, ``"30min"``, ``"hour"``,
              ``"day"``, or ``"auto"`` (picks based on data spread).
      range_hours: How far back to look. Use ``0`` for auto-range
                   (derived from earliest event to now).

    Returns:
      TimelineResult with buckets and resolved bucket/range info.
    """
    now = time.time()

    # Auto-range: find earliest event timestamp
    if range_hours <= 0:
      earliest = now
      for evt in events:
        ts = evt.get("created_at")
        if ts is not None and ts < earliest:
          earliest = ts
      spread = now - earliest
      # Pad by 10% on each side for visual breathing room, minimum 1h
      range_hours = max(1.0, (spread * 1.2) / 3600)

    cutoff = now - (range_hours * 3600)

    # Resolve bucket granularity
    if bucket == "auto":
      if range_hours <= 1:
        bucket = "5min"
      elif range_hours <= 6:
        bucket = "30min"
      elif range_hours <= 48:
        bucket = "hour"
      else:
        bucket = "day"

    bucket_seconds = _bucket_to_seconds(bucket)

    # Initialize empty buckets from cutoff to now
    buckets: Dict[int, TimeSeriesBucket] = {}
    t = cutoff
    while t < now + bucket_seconds:
      bucket_ts = int(t // bucket_seconds) * bucket_seconds
      if bucket_ts not in buckets:
        buckets[bucket_ts] = TimeSeriesBucket(
          timestamp=float(bucket_ts),
          label=_format_bucket_label(bucket_ts, bucket),
        )
      t += bucket_seconds

    # Fill buckets from events
    for evt in events:
      event_type = evt.get("event", "")
      ts = evt.get("created_at")
      if ts is None or ts < cutoff:
        continue

      bucket_ts = int(ts // bucket_seconds) * bucket_seconds
      if bucket_ts not in buckets:
        buckets[bucket_ts] = TimeSeriesBucket(
          timestamp=float(bucket_ts),
          label=_format_bucket_label(bucket_ts, bucket),
        )

      b = buckets[bucket_ts]

      if event_type == "RunCompleted":
        b.run_count += 1
        metrics = evt.get("metrics")
        if metrics:
          b.tokens += _get(metrics, "total_tokens", 0) or 0
          b.cost += _get(metrics, "cost", 0) or 0
      elif event_type == "RunError":
        b.run_count += 1
        b.error_count += 1

    sorted_buckets = sorted(buckets.values(), key=lambda b: b.timestamp)
    return TimelineResult(
      buckets=sorted_buckets,
      resolved_bucket=bucket,
      bucket_seconds=bucket_seconds,
      range_hours=range_hours,
    )


@dataclass
class TimelineResult:
  """Result of compute_timeline with resolved metadata."""

  buckets: List[TimeSeriesBucket] = field(default_factory=list)
  resolved_bucket: str = "hour"
  bucket_seconds: int = 3600
  range_hours: float = 24.0


def _get(obj: Any, key: str, default: Any = None) -> Any:
  """Get a value from a dict or object attribute, handling both cases."""
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


def _percentile(data: List[float], pct: float) -> float:
  """Compute a percentile from sorted data."""
  if not data:
    return 0.0
  sorted_data = sorted(data)
  k = (len(sorted_data) - 1) * (pct / 100.0)
  f = int(k)
  c = f + 1
  if c >= len(sorted_data):
    return sorted_data[-1]
  return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _bucket_to_seconds(bucket: str) -> int:
  """Convert bucket name to duration in seconds."""
  return {"5min": 300, "30min": 1800, "hour": 3600, "day": 86400}.get(bucket, 3600)


def _format_bucket_label(ts: float, bucket: str) -> str:
  """Format a bucket timestamp into a human-readable label.

  Note: labels are UTC. The frontend converts to local time using
  the ``timestamp`` field (Unix epoch) via ``new Date()``.
  """
  import datetime

  dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
  if bucket == "day":
    return dt.strftime("%Y-%m-%d")
  if bucket in ("5min", "30min"):
    return dt.strftime("%H:%M")
  return dt.strftime("%H:00")
