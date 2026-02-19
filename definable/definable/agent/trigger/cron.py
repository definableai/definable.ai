"""Cron trigger — runs a handler on a schedule."""

from definable.agent.trigger.base import BaseTrigger


class Cron(BaseTrigger):
  """Cron-based trigger.

  Validates the schedule expression at construction time using
  ``croniter`` (lazy-imported, optional dependency).

  Args:
    schedule: Cron expression (e.g. ``"*/5 * * * *"`` for every 5 minutes).
    timezone: IANA timezone string (default ``"UTC"``).

  Raises:
    ValueError: If the schedule expression is invalid.
    ImportError: If ``croniter`` is not installed.

  Example::

    @agent.on(Cron("0 9 * * *"))
    async def daily_summary(event):
        return "Summarize yesterday's metrics."
  """

  def __init__(self, schedule: str, *, timezone: str = "UTC") -> None:
    try:
      from croniter import croniter  # type: ignore[import-untyped]
    except ImportError as e:
      raise ImportError("croniter is required for Cron triggers. Install it with: pip install 'definable[cron]'") from e

    if not croniter.is_valid(schedule):
      raise ValueError(f"Invalid cron expression: {schedule!r}")

    self.schedule = schedule
    self.timezone = timezone

  @property
  def name(self) -> str:
    return f"cron({self.schedule})"

  def next_run(self, base_time: float) -> float:
    """Return the next fire time as a Unix timestamp.

    Args:
      base_time: Unix timestamp to compute the next run from.

    Returns:
      Unix timestamp of the next scheduled execution.
    """
    from croniter import croniter

    return float(croniter(self.schedule, base_time).get_next())
