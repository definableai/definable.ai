"""Scheduler — job lifecycle, stores, and scheduling loop."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.scheduler.job import JobStatus, ScheduledJob
  from definable.agent.scheduler.scheduler import Scheduler
  from definable.agent.scheduler.store import InMemoryJobStore, JobStore, SQLiteJobStore

__all__ = [
  "ScheduledJob",
  "JobStatus",
  "JobStore",
  "InMemoryJobStore",
  "SQLiteJobStore",
  "Scheduler",
]


def __getattr__(name: str):
  if name in ("ScheduledJob", "JobStatus"):
    from definable.agent.scheduler import job as _j

    return getattr(_j, name)
  if name in ("JobStore", "InMemoryJobStore", "SQLiteJobStore"):
    from definable.agent.scheduler import store as _st

    return getattr(_st, name)
  if name == "Scheduler":
    from definable.agent.scheduler.scheduler import Scheduler

    return Scheduler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
