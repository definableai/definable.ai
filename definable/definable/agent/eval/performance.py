"""PerformanceEval — runtime and memory profiling for agents."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from definable.agent.eval.base import BaseEval, EvalCase
from definable.agent.eval.result import PerformanceResult
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.team.team import Team


@dataclass
class PerformanceEval(BaseEval):
  """Evaluate agent execution performance (runtime + memory).

  Runs the agent multiple times and measures execution time and peak
  memory delta using ``tracemalloc``. Supports warmup runs that are
  excluded from results.

  Args:
      duration_threshold_ms: Maximum allowed p95 execution time (ms). None = no check.
      memory_threshold_mb: Maximum allowed peak memory delta (MB). None = no check.
      runs: Number of profiling runs to execute.
      warmup_runs: Number of warmup runs (excluded from results).

  Example::

      eval = PerformanceEval(
          duration_threshold_ms=5000,
          memory_threshold_mb=50,
          runs=3,
      )
      result = await eval.arun(agent, EvalCase(input="Complex query"))
      print(f"p95 duration: {result.duration_ms:.0f}ms")
      print(f"peak memory: {result.peak_memory_mb:.1f}MB")
  """

  name: str = "performance"
  duration_threshold_ms: Optional[float] = None
  memory_threshold_mb: Optional[float] = None
  runs: int = 3
  warmup_runs: int = 0
  _id: str = field(default_factory=lambda: str(uuid4()))

  async def evaluate(self, agent: "Agent", case: EvalCase) -> PerformanceResult:
    """Run the agent multiple times and profile performance."""
    return await self._profile(agent, None, case)

  async def evaluate_team(self, team: "Team", case: EvalCase) -> PerformanceResult:
    """Run the team multiple times and profile performance."""
    return await self._profile(None, team, case)

  async def _profile(
    self,
    agent: Optional["Agent"],
    team: Optional["Team"],
    case: EvalCase,
  ) -> PerformanceResult:
    """Core profiling logic shared between agent and team."""
    durations: list[float] = []
    peak_memory_mb = 0.0

    # Warmup runs (not measured)
    for i in range(self.warmup_runs):
      log_info(f"[{self.name}] Warmup run {i + 1}/{self.warmup_runs}")
      try:
        if agent:
          await agent.arun(case.input)
        elif team:
          await team.arun(case.input)
      except Exception as e:
        log_error(f"PerformanceEval: warmup run failed: {e}")

    # Profiling runs
    for i in range(self.runs):
      log_info(f"[{self.name}] Profiling run {i + 1}/{self.runs}")

      # Start memory tracking
      tracemalloc.start()

      start = time()
      try:
        if agent:
          await agent.arun(case.input)
        elif team:
          await team.arun(case.input)
      except Exception as e:
        log_error(f"PerformanceEval: profiling run {i + 1} failed: {e}")
      finally:
        duration_ms = (time() - start) * 1000
        durations.append(duration_ms)

        # Capture peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        run_peak_mb = peak / (1024 * 1024)
        peak_memory_mb = max(peak_memory_mb, run_peak_mb)

    if not durations:
      return PerformanceResult(
        eval_name=self.name,
        success=False,
        reason="No successful profiling runs.",
        runs=0,
      )

    # Calculate p95 duration
    sorted_durations = sorted(durations)
    p95_index = max(0, int(len(sorted_durations) * 0.95) - 1)
    p95_duration = sorted_durations[p95_index]

    # Determine success
    duration_ok = self.duration_threshold_ms is None or p95_duration <= self.duration_threshold_ms
    memory_ok = self.memory_threshold_mb is None or peak_memory_mb <= self.memory_threshold_mb
    success = duration_ok and memory_ok

    reasons: list[str] = []
    if not duration_ok:
      reasons.append(f"p95 duration {p95_duration:.0f}ms exceeds threshold {self.duration_threshold_ms:.0f}ms")
    if not memory_ok:
      reasons.append(f"peak memory {peak_memory_mb:.1f}MB exceeds threshold {self.memory_threshold_mb:.1f}MB")

    return PerformanceResult(
      eval_name=self.name,
      score=10.0 if success else 0.0,
      success=success,
      reason="; ".join(reasons) if reasons else "Performance within thresholds.",
      duration_ms=p95_duration,
      peak_memory_mb=peak_memory_mb,
      duration_threshold_ms=self.duration_threshold_ms,
      memory_threshold_mb=self.memory_threshold_mb,
      runs=len(durations),
      durations=durations,
    )
