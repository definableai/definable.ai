"""ScheduledJob — job lifecycle and state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional
from uuid import uuid4

from definable.agent.trigger.base import BaseTrigger


class JobStatus(str, Enum):
  """Lifecycle state of a scheduled job."""

  PENDING = "pending"
  ACTIVE = "active"
  PAUSED = "paused"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
  """A single scheduled job wrapping a trigger.

  Tracks lifecycle state (status, run count, timing) and provides
  a persistent identity via job_id.

  Args:
    trigger: The trigger that fires this job.
    job_id: Unique identifier (auto-generated if omitted).
    name: Human-readable name for the job.
    max_runs: Maximum number of executions (None = unlimited).
    metadata: Arbitrary user metadata.

  Example::

    job = ScheduledJob(
      trigger=Interval(seconds=60),
      name="health-check",
      max_runs=100,
    )
  """

  trigger: BaseTrigger
  job_id: str = field(default_factory=lambda: str(uuid4()))
  name: str = ""
  status: JobStatus = JobStatus.PENDING
  max_runs: Optional[int] = None
  metadata: Dict[str, Any] = field(default_factory=dict)

  # Timing
  created_at: float = field(default_factory=time)
  next_run_at: float = 0.0
  last_run_at: float = 0.0
  run_count: int = 0
  failure_count: int = 0
  last_error: Optional[str] = None

  def __post_init__(self) -> None:
    if not self.name:
      self.name = self.trigger.name
    if self.next_run_at == 0.0:
      self.next_run_at = self.trigger.next_run(self.created_at)  # type: ignore[attr-defined]

  @property
  def is_runnable(self) -> bool:
    """True if the job is active and hasn't exceeded max_runs."""
    if self.status not in (JobStatus.PENDING, JobStatus.ACTIVE):
      return False
    if self.max_runs is not None and self.run_count >= self.max_runs:
      return False
    return True

  def activate(self) -> None:
    """Transition to ACTIVE."""
    self.status = JobStatus.ACTIVE

  def pause(self) -> None:
    """Transition to PAUSED."""
    self.status = JobStatus.PAUSED

  def resume(self) -> None:
    """Resume from PAUSED to ACTIVE."""
    if self.status == JobStatus.PAUSED:
      self.status = JobStatus.ACTIVE

  def cancel(self) -> None:
    """Transition to CANCELLED."""
    self.status = JobStatus.CANCELLED

  def record_run(self) -> None:
    """Record a successful run."""
    self.run_count += 1
    self.last_run_at = time()
    self.last_error = None
    self.next_run_at = self.trigger.next_run(self.last_run_at)  # type: ignore[attr-defined]

    # Check if max runs reached
    if self.max_runs is not None and self.run_count >= self.max_runs:
      self.status = JobStatus.COMPLETED

  def record_failure(self, error: str) -> None:
    """Record a failed run."""
    self.run_count += 1
    self.failure_count += 1
    self.last_run_at = time()
    self.last_error = error
    self.next_run_at = self.trigger.next_run(self.last_run_at)  # type: ignore[attr-defined]

  def to_dict(self) -> Dict[str, Any]:
    """Serialize to a plain dict (for storage)."""
    return {
      "job_id": self.job_id,
      "name": self.name,
      "status": self.status.value,
      "max_runs": self.max_runs,
      "metadata": self.metadata,
      "created_at": self.created_at,
      "next_run_at": self.next_run_at,
      "last_run_at": self.last_run_at,
      "run_count": self.run_count,
      "failure_count": self.failure_count,
      "last_error": self.last_error,
      "trigger_name": self.trigger.name,
    }
