"""Usage and cost tracking for agent runs.

Tracks per-run and per-session token usage, cost estimates, and emits
UsageEvent through the event system.

Usage::

    agent = Agent(model=model, usage=True)
    output = await agent.arun("Hello")
    print(agent.usage_tracker.session_total)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from definable.model.metrics import Metrics


@dataclass
class UsageSnapshot:
  """Snapshot of token usage and cost for a single run or session aggregate.

  Attributes:
    input_tokens: Number of input/prompt tokens.
    output_tokens: Number of output/completion tokens.
    total_tokens: Total tokens (input + output).
    estimated_cost: Estimated cost in USD.
    runs: Number of runs contributing to this snapshot.
    model_id: Model identifier (for per-run snapshots).
  """

  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  estimated_cost: float = 0.0
  runs: int = 0
  model_id: Optional[str] = None

  def __add__(self, other: UsageSnapshot) -> UsageSnapshot:
    return UsageSnapshot(
      input_tokens=self.input_tokens + other.input_tokens,
      output_tokens=self.output_tokens + other.output_tokens,
      total_tokens=self.total_tokens + other.total_tokens,
      estimated_cost=self.estimated_cost + other.estimated_cost,
      runs=self.runs + other.runs,
    )

  def to_dict(self) -> Dict[str, Any]:
    result: Dict[str, Any] = {
      "input_tokens": self.input_tokens,
      "output_tokens": self.output_tokens,
      "total_tokens": self.total_tokens,
      "estimated_cost": round(self.estimated_cost, 6),
      "runs": self.runs,
    }
    if self.model_id:
      result["model_id"] = self.model_id
    return result

  def __str__(self) -> str:
    return f"Usage({self.total_tokens} tokens, ${self.estimated_cost:.4f}, {self.runs} run{'s' if self.runs != 1 else ''})"


@dataclass
class UsageTracker:
  """Tracks token usage and cost across agent runs.

  Thread-safe via asyncio.Lock. Accumulates per-run snapshots
  and maintains a session-level aggregate.

  Attributes:
    enabled: Whether tracking is active.
  """

  enabled: bool = True
  _session_total: UsageSnapshot = field(default_factory=UsageSnapshot)
  _run_snapshots: List[UsageSnapshot] = field(default_factory=list)
  _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

  def record_run(self, metrics: Optional[Metrics], model_id: Optional[str] = None) -> UsageSnapshot:
    """Record usage from a completed run.

    Args:
      metrics: Metrics from the model response.
      model_id: Optional model identifier.

    Returns:
      The per-run UsageSnapshot.
    """
    if not self.enabled or metrics is None:
      return UsageSnapshot()

    snapshot = UsageSnapshot(
      input_tokens=metrics.input_tokens or 0,
      output_tokens=metrics.output_tokens or 0,
      total_tokens=metrics.total_tokens or 0,
      estimated_cost=metrics.cost or 0.0,
      runs=1,
      model_id=model_id,
    )

    self._session_total = self._session_total + snapshot
    self._run_snapshots.append(snapshot)
    return snapshot

  async def arecord_run(self, metrics: Optional[Metrics], model_id: Optional[str] = None) -> UsageSnapshot:
    """Async version of record_run (thread-safe)."""
    async with self._lock:
      return self.record_run(metrics, model_id)

  @property
  def session_total(self) -> UsageSnapshot:
    """Cumulative usage across all runs in this session."""
    return self._session_total

  @property
  def last_run(self) -> Optional[UsageSnapshot]:
    """Usage from the most recent run."""
    return self._run_snapshots[-1] if self._run_snapshots else None

  @property
  def run_count(self) -> int:
    """Number of recorded runs."""
    return len(self._run_snapshots)

  @property
  def all_runs(self) -> List[UsageSnapshot]:
    """All recorded run snapshots."""
    return list(self._run_snapshots)

  def reset(self) -> None:
    """Reset all usage tracking."""
    self._session_total = UsageSnapshot()
    self._run_snapshots.clear()
