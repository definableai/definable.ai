"""MetricsPlugin — collects timing and usage metrics per run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

from definable.agent.plugin.base import Plugin

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.pipeline.state import LoopState


@dataclass
class RunMetrics:
  """Metrics collected for a single agent run."""

  run_id: str = ""
  phase_durations: Dict[str, float] = field(default_factory=dict)
  total_duration_ms: float = 0.0
  tool_call_count: int = 0
  message_count: int = 0
  _phase_start: float = field(default=0.0, repr=False)


class MetricsPlugin(Plugin):
  """Collects per-run metrics: phase durations, tool calls, totals.

  Metrics are stored in ``self.history`` (capped at ``max_history``).
  The latest run's metrics are in ``self.last``.

  Args:
    max_history: Maximum number of run metrics to keep (default 100).

  Example::

    metrics = MetricsPlugin()
    agent = Agent(model="gpt-4o", plugins=[metrics])
    await agent.arun("Hello")
    print(metrics.last.total_duration_ms)
  """

  def __init__(self, *, max_history: int = 100) -> None:
    self._max_history = max_history
    self.history: List[RunMetrics] = []
    self.last: Optional[RunMetrics] = None
    self._current: Optional[RunMetrics] = None

  @property
  def name(self) -> str:
    return "metrics"

  @property
  def description(self) -> str:
    return "Collects timing and usage metrics per agent run."

  @property
  def modifies(self) -> FrozenSet[str]:
    return frozenset({"*"})

  async def on_load(self, agent: "Agent") -> None:
    agent.pipeline.hook("before:prepare", self._on_run_start, priority=100)
    agent.pipeline.hook("before:*", self._before_phase, priority=99)
    agent.pipeline.hook("after:*", self._after_phase, priority=-99)
    agent.pipeline.hook("after:store", self._on_run_end, priority=-100)

  async def _on_run_start(self, state: "LoopState") -> "LoopState":
    self._current = RunMetrics(run_id=state.run_id)
    self._current._phase_start = time.perf_counter()
    return state

  async def _before_phase(self, state: "LoopState") -> "LoopState":
    if self._current is not None:
      self._current._phase_start = time.perf_counter()
    return state

  async def _after_phase(self, state: "LoopState") -> "LoopState":
    if self._current is not None and self._current._phase_start > 0:
      duration_ms = (time.perf_counter() - self._current._phase_start) * 1000
      self._current.phase_durations[state.phase] = duration_ms
    return state

  async def _on_run_end(self, state: "LoopState") -> "LoopState":
    if self._current is not None:
      self._current.total_duration_ms = sum(self._current.phase_durations.values())
      self._current.message_count = len(state.all_messages)
      self._current.tool_call_count = len(getattr(state, "tool_results", []))
      self.last = self._current
      self.history.append(self._current)
      if len(self.history) > self._max_history:
        self.history = self.history[-self._max_history :]
      self._current = None
    return state

  @property
  def average_duration_ms(self) -> float:
    """Average total duration across all recorded runs."""
    if not self.history:
      return 0.0
    return sum(r.total_duration_ms for r in self.history) / len(self.history)
