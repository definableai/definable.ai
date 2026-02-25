"""OneShot trigger — fires once at a specific time or after a delay."""

from __future__ import annotations

import math
from time import time

from definable.agent.trigger.base import BaseTrigger


class OneShot(BaseTrigger):
  """Trigger that fires exactly once.

  Specify either ``delay`` (seconds from now) or ``fire_at``
  (absolute Unix timestamp). If neither is given, fires immediately.

  Args:
    delay: Seconds from now until fire (default 0).
    fire_at: Absolute Unix timestamp to fire at (default 0 = use delay).

  Example::

    # Fire in 60 seconds
    trigger = OneShot(delay=60)

    # Fire at a specific time
    trigger = OneShot(fire_at=1700000000.0)
  """

  def __init__(self, *, delay: float = 0.0, fire_at: float = 0.0) -> None:
    if fire_at <= 0 and delay <= 0:
      raise ValueError("OneShot requires either 'delay' > 0 or 'fire_at' > 0")
    self._delay = delay
    self._fire_at = fire_at if fire_at > 0 else (time() + delay)
    self._fired = False

  @property
  def name(self) -> str:
    return f"oneshot(at={self._fire_at:.0f})"

  @property
  def fire_at(self) -> float:
    """The absolute fire time."""
    return self._fire_at

  @property
  def fired(self) -> bool:
    """Whether this trigger has already fired."""
    return self._fired

  def mark_fired(self) -> None:
    """Mark this trigger as having fired."""
    self._fired = True

  def next_run(self, base_time: float) -> float:
    """Return fire_at if not yet fired, else infinity."""
    if self._fired:
      return math.inf
    return self._fire_at
