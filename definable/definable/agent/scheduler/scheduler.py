"""Scheduler — the core scheduling loop."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from definable.agent.scheduler.job import JobStatus, ScheduledJob
from definable.agent.scheduler.store import InMemoryJobStore
from definable.agent.trigger.base import BaseTrigger, TriggerEvent
from definable.agent.trigger.oneshot import OneShot
from definable.utils.log import log_debug, log_error, log_info

if TYPE_CHECKING:
  from definable.agent.scheduler.store import JobStore
  from definable.agent.trigger.executor import TriggerExecutor


class Scheduler:
  """Central scheduling loop that manages jobs and fires triggers.

  Replaces the ad-hoc ``_run_cron_scheduler`` in AgentRuntime with a proper
  job management system supporting cron, interval, oneshot, and event triggers.

  Args:
    store: Job store for persistence (defaults to InMemoryJobStore).
    tick_interval: How often to check for due jobs (seconds, default 1.0).
    max_concurrent: Maximum concurrent job executions (default 10).

  Example::

    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60), name="health-check")
    await scheduler.start(executor)  # blocks until stopped
  """

  def __init__(
    self,
    *,
    store: Optional["JobStore"] = None,
    tick_interval: float = 1.0,
    max_concurrent: int = 10,
  ) -> None:
    self._store: "JobStore" = store or InMemoryJobStore()
    self._tick_interval = tick_interval
    self._max_concurrent = max_concurrent
    self._jobs: Dict[str, ScheduledJob] = {}
    self._running = False
    self._stop_event = asyncio.Event()
    self._semaphore = asyncio.Semaphore(max_concurrent)
    self._active_tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

    # Callbacks
    self.on_job_started: Optional[Callable[[ScheduledJob], None]] = None
    self.on_job_completed: Optional[Callable[[ScheduledJob], None]] = None
    self.on_job_failed: Optional[Callable[[ScheduledJob, str], None]] = None

  @property
  def is_running(self) -> bool:
    return self._running

  @property
  def job_count(self) -> int:
    return len(self._jobs)

  def add(
    self,
    trigger: BaseTrigger,
    *,
    name: str = "",
    max_runs: Optional[int] = None,
    metadata: Optional[Dict] = None,
  ) -> ScheduledJob:
    """Create and register a new scheduled job.

    Args:
      trigger: The trigger to schedule.
      name: Human-readable job name.
      max_runs: Max executions (None = unlimited).
      metadata: Arbitrary metadata dict.

    Returns:
      The created ScheduledJob.
    """
    job = ScheduledJob(
      trigger=trigger,
      name=name or trigger.name,
      max_runs=max_runs,
      metadata=metadata or {},
    )
    job.activate()
    self._jobs[job.job_id] = job
    log_info(f"[scheduler] Added job {job.job_id!r} ({job.name})")
    return job

  def add_job(self, job: ScheduledJob) -> ScheduledJob:
    """Register an existing ScheduledJob.

    Useful for restoring jobs from a store.
    """
    self._jobs[job.job_id] = job
    return job

  def get(self, job_id: str) -> Optional[ScheduledJob]:
    """Get a job by ID."""
    return self._jobs.get(job_id)

  def list_jobs(self, status: Optional[JobStatus] = None) -> List[ScheduledJob]:
    """List all registered jobs, optionally filtered by status."""
    jobs = list(self._jobs.values())
    if status is not None:
      jobs = [j for j in jobs if j.status == status]
    return jobs

  def pause(self, job_id: str) -> bool:
    """Pause a job. Returns True if found and paused."""
    job = self._jobs.get(job_id)
    if job is None:
      return False
    job.pause()
    return True

  def resume(self, job_id: str) -> bool:
    """Resume a paused job. Returns True if found and resumed."""
    job = self._jobs.get(job_id)
    if job is None:
      return False
    job.resume()
    return True

  def cancel(self, job_id: str) -> bool:
    """Cancel a job. Returns True if found and cancelled."""
    job = self._jobs.get(job_id)
    if job is None:
      return False
    job.cancel()
    # Cancel running task if any
    task = self._active_tasks.pop(job_id, None)
    if task is not None:
      task.cancel()
    return True

  def remove(self, job_id: str) -> bool:
    """Remove a job entirely. Returns True if found and removed."""
    self.cancel(job_id)
    return self._jobs.pop(job_id, None) is not None

  async def start(self, executor: "TriggerExecutor") -> None:
    """Start the scheduling loop (blocks until stopped).

    Args:
      executor: TriggerExecutor to run trigger handlers.
    """
    self._running = True
    self._stop_event.clear()
    log_info(f"[scheduler] Started with {len(self._jobs)} job(s)")

    try:
      while not self._stop_event.is_set():
        await self._tick(executor)
        try:
          await asyncio.wait_for(self._stop_event.wait(), timeout=self._tick_interval)
          break  # Stop event was set
        except asyncio.TimeoutError:
          pass  # Normal tick cycle
    finally:
      self._running = False
      # Wait for active tasks to finish (with timeout)
      if self._active_tasks:
        log_info(f"[scheduler] Waiting for {len(self._active_tasks)} active task(s)")
        tasks = list(self._active_tasks.values())
        await asyncio.gather(*tasks, return_exceptions=True)
      log_info("[scheduler] Stopped")

  def stop(self) -> None:
    """Signal the scheduler to stop."""
    self._stop_event.set()

  async def _tick(self, executor: "TriggerExecutor") -> None:
    """Check all jobs and fire any that are due."""
    now = time.time()

    for job in list(self._jobs.values()):
      if not job.is_runnable:
        continue

      if job.job_id in self._active_tasks:
        continue  # Already running

      if now >= job.next_run_at:
        # Fire!
        task = asyncio.create_task(self._execute_job(job, executor))
        self._active_tasks[job.job_id] = task

        def _on_done(t: asyncio.Task, _jid: str = job.job_id) -> None:  # type: ignore[type-arg]
          self._active_tasks.pop(_jid, None)

        task.add_done_callback(_on_done)

  async def _execute_job(self, job: ScheduledJob, executor: "TriggerExecutor") -> None:
    """Execute a single job with concurrency control."""
    async with self._semaphore:
      log_debug(f"[scheduler] Firing job {job.job_id!r} ({job.name})")

      if self.on_job_started:
        self.on_job_started(job)

      try:
        event = TriggerEvent(source=job.trigger.name)
        await executor.execute(job.trigger, event)
        job.record_run()

        # Handle OneShot: mark as fired
        if isinstance(job.trigger, OneShot):
          job.trigger.mark_fired()
          if job.status == JobStatus.ACTIVE:
            job.status = JobStatus.COMPLETED

        if self.on_job_completed:
          self.on_job_completed(job)

        # Persist state
        await self._store.save(job)

      except Exception as e:
        error_msg = str(e)
        job.record_failure(error_msg)
        log_error(f"[scheduler] Job {job.job_id!r} failed: {error_msg}")

        if self.on_job_failed:
          self.on_job_failed(job, error_msg)

        await self._store.save(job)

  async def save_all(self) -> None:
    """Persist all jobs to the store."""
    for job in self._jobs.values():
      await self._store.save(job)
