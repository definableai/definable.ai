"""Interval trigger — fires at regular intervals."""

from __future__ import annotations

from definable.agent.trigger.base import BaseTrigger


class Interval(BaseTrigger):
  """Trigger that fires at regular intervals.

  Args:
    seconds: Interval between executions in seconds.

  Example::

    trigger = Interval(seconds=60)  # Fire every minute
    trigger.next_run(1000.0)  # → 1060.0

  Raises:
    ValueError: If seconds <= 0.
  """

  def __init__(self, seconds: float) -> None:
    if seconds <= 0:
      raise ValueError(f"Interval seconds must be positive, got {seconds}")
    self._seconds = seconds

  @property
  def name(self) -> str:
    return f"interval({self._seconds}s)"

  @property
  def seconds(self) -> float:
    """The interval duration in seconds."""
    return self._seconds

  def next_run(self, base_time: float) -> float:
    """Return base_time + interval seconds."""
    return base_time + self._seconds
