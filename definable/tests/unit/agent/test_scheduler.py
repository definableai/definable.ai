"""Tests for Scheduler — job lifecycle, stores, and scheduling loop."""

import asyncio
import time

import pytest

from definable.agent.scheduler.job import JobStatus, ScheduledJob
from definable.agent.scheduler.scheduler import Scheduler
from definable.agent.scheduler.store import InMemoryJobStore
from definable.agent.trigger.base import BaseTrigger, TriggerEvent
from definable.agent.trigger.interval import Interval
from definable.agent.trigger.oneshot import OneShot


# --- Fixtures ---


class InstantTrigger(BaseTrigger):
  """Trigger that always fires immediately (for testing)."""

  @property
  def name(self) -> str:
    return "instant"

  def next_run(self, base_time: float) -> float:
    return base_time  # Fire immediately


class FakeExecutor:
  """Fake TriggerExecutor that records executions."""

  def __init__(self, *, fail: bool = False):
    self.executions: list = []
    self._fail = fail

  async def execute(self, trigger: BaseTrigger, event: TriggerEvent) -> None:
    self.executions.append((trigger, event))
    if self._fail:
      raise RuntimeError("Execution failed")


# --- ScheduledJob Tests ---


class TestScheduledJob:
  def test_defaults(self):
    trigger = Interval(seconds=60)
    job = ScheduledJob(trigger=trigger)
    assert job.status == JobStatus.PENDING
    assert job.run_count == 0
    assert job.failure_count == 0
    assert job.max_runs is None
    assert job.name == trigger.name

  def test_custom_name(self):
    job = ScheduledJob(trigger=Interval(seconds=60), name="my-job")
    assert job.name == "my-job"

  def test_auto_sets_next_run(self):
    before = time.time()
    job = ScheduledJob(trigger=Interval(seconds=10))
    assert job.next_run_at >= before + 10

  def test_is_runnable_pending(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    assert job.is_runnable is True

  def test_is_runnable_active(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.activate()
    assert job.is_runnable is True

  def test_not_runnable_paused(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.pause()
    assert job.is_runnable is False

  def test_not_runnable_cancelled(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.cancel()
    assert job.is_runnable is False

  def test_not_runnable_completed(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.status = JobStatus.COMPLETED
    assert job.is_runnable is False

  def test_max_runs_limit(self):
    job = ScheduledJob(trigger=Interval(seconds=60), max_runs=2)
    job.activate()
    job.record_run()
    assert job.is_runnable is True
    job.record_run()
    assert job.is_runnable is False
    assert job.status == JobStatus.COMPLETED

  def test_record_run(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.activate()
    job.record_run()
    assert job.run_count == 1
    assert job.last_run_at > 0
    assert job.last_error is None

  def test_record_failure(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.activate()
    job.record_failure("oops")
    assert job.run_count == 1
    assert job.failure_count == 1
    assert job.last_error == "oops"

  def test_pause_resume(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.activate()
    job.pause()
    assert job.status == JobStatus.PAUSED
    job.resume()
    assert job.status == JobStatus.ACTIVE

  def test_resume_only_from_paused(self):
    job = ScheduledJob(trigger=Interval(seconds=60))
    job.cancel()
    job.resume()  # Should not change status
    assert job.status == JobStatus.CANCELLED

  def test_to_dict(self):
    job = ScheduledJob(trigger=Interval(seconds=60), name="test-job")
    d = job.to_dict()
    assert d["name"] == "test-job"
    assert d["status"] == "pending"
    assert d["run_count"] == 0
    assert "trigger_name" in d

  def test_metadata(self):
    job = ScheduledJob(trigger=Interval(seconds=60), metadata={"env": "prod"})
    assert job.metadata["env"] == "prod"


# --- InMemoryJobStore Tests ---


class TestInMemoryJobStore:
  @pytest.mark.asyncio
  async def test_save_and_get(self):
    store = InMemoryJobStore()
    job = ScheduledJob(trigger=Interval(seconds=60))
    await store.save(job)
    retrieved = await store.get(job.job_id)
    assert retrieved is job

  @pytest.mark.asyncio
  async def test_get_missing(self):
    store = InMemoryJobStore()
    assert await store.get("nonexistent") is None

  @pytest.mark.asyncio
  async def test_list_jobs(self):
    store = InMemoryJobStore()
    j1 = ScheduledJob(trigger=Interval(seconds=60))
    j2 = ScheduledJob(trigger=Interval(seconds=120))
    await store.save(j1)
    await store.save(j2)
    jobs = await store.list_jobs()
    assert len(jobs) == 2

  @pytest.mark.asyncio
  async def test_list_jobs_filtered(self):
    store = InMemoryJobStore()
    j1 = ScheduledJob(trigger=Interval(seconds=60))
    j1.activate()
    j2 = ScheduledJob(trigger=Interval(seconds=120))
    j2.pause()
    await store.save(j1)
    await store.save(j2)
    active = await store.list_jobs(status=JobStatus.ACTIVE)
    assert len(active) == 1
    assert active[0].job_id == j1.job_id

  @pytest.mark.asyncio
  async def test_delete(self):
    store = InMemoryJobStore()
    job = ScheduledJob(trigger=Interval(seconds=60))
    await store.save(job)
    assert await store.delete(job.job_id) is True
    assert await store.get(job.job_id) is None

  @pytest.mark.asyncio
  async def test_delete_missing(self):
    store = InMemoryJobStore()
    assert await store.delete("nonexistent") is False

  @pytest.mark.asyncio
  async def test_count(self):
    store = InMemoryJobStore()
    await store.save(ScheduledJob(trigger=Interval(seconds=60)))
    await store.save(ScheduledJob(trigger=Interval(seconds=120)))
    assert await store.count() == 2

  @pytest.mark.asyncio
  async def test_count_filtered(self):
    store = InMemoryJobStore()
    j1 = ScheduledJob(trigger=Interval(seconds=60))
    j1.activate()
    j2 = ScheduledJob(trigger=Interval(seconds=120))
    await store.save(j1)
    await store.save(j2)
    assert await store.count(status=JobStatus.ACTIVE) == 1
    assert await store.count(status=JobStatus.PENDING) == 1


# --- Scheduler Tests ---


class TestSchedulerAdd:
  def test_add_creates_job(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60), name="my-job")
    assert job.name == "my-job"
    assert job.status == JobStatus.ACTIVE
    assert scheduler.job_count == 1

  def test_add_auto_name(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    assert "interval" in job.name

  def test_add_with_max_runs(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60), max_runs=5)
    assert job.max_runs == 5

  def test_add_with_metadata(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60), metadata={"key": "val"})
    assert job.metadata == {"key": "val"}

  def test_add_job_existing(self):
    scheduler = Scheduler()
    job = ScheduledJob(trigger=Interval(seconds=60))
    scheduler.add_job(job)
    assert scheduler.get(job.job_id) is job


class TestSchedulerManagement:
  def test_get_existing(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    assert scheduler.get(job.job_id) is job

  def test_get_missing(self):
    scheduler = Scheduler()
    assert scheduler.get("nonexistent") is None

  def test_list_jobs(self):
    scheduler = Scheduler()
    scheduler.add(Interval(seconds=60))
    scheduler.add(Interval(seconds=120))
    assert len(scheduler.list_jobs()) == 2

  def test_list_jobs_by_status(self):
    scheduler = Scheduler()
    j1 = scheduler.add(Interval(seconds=60))
    j2 = scheduler.add(Interval(seconds=120))
    scheduler.pause(j2.job_id)
    active = scheduler.list_jobs(status=JobStatus.ACTIVE)
    assert len(active) == 1
    assert active[0].job_id == j1.job_id

  def test_pause(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    assert scheduler.pause(job.job_id) is True
    assert job.status == JobStatus.PAUSED

  def test_pause_missing(self):
    scheduler = Scheduler()
    assert scheduler.pause("nonexistent") is False

  def test_resume(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    scheduler.pause(job.job_id)
    assert scheduler.resume(job.job_id) is True
    assert job.status == JobStatus.ACTIVE

  def test_cancel(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    assert scheduler.cancel(job.job_id) is True
    assert job.status == JobStatus.CANCELLED

  def test_remove(self):
    scheduler = Scheduler()
    job = scheduler.add(Interval(seconds=60))
    assert scheduler.remove(job.job_id) is True
    assert scheduler.job_count == 0

  def test_remove_missing(self):
    scheduler = Scheduler()
    assert scheduler.remove("nonexistent") is False


class TestSchedulerLoop:
  @pytest.mark.asyncio
  async def test_fires_due_job(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor()

    # Add a job that's already due
    trigger = InstantTrigger()
    job = scheduler.add(trigger)
    job.next_run_at = time.time() - 1  # Already past

    # Run for a short time
    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert len(executor.executions) >= 1
    assert job.run_count >= 1

  @pytest.mark.asyncio
  async def test_does_not_fire_future_job(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor()

    # Add a job far in the future
    scheduler.add(Interval(seconds=9999))

    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert len(executor.executions) == 0

  @pytest.mark.asyncio
  async def test_handles_job_failure(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor(fail=True)

    trigger = InstantTrigger()
    job = scheduler.add(trigger)
    job.next_run_at = time.time() - 1

    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert job.failure_count >= 1
    assert job.last_error is not None

  @pytest.mark.asyncio
  async def test_oneshot_completes_after_fire(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor()

    trigger = OneShot(fire_at=time.time() - 1)  # Already past
    job = scheduler.add(trigger)
    job.next_run_at = time.time() - 1

    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert trigger.fired is True
    assert job.status == JobStatus.COMPLETED

  @pytest.mark.asyncio
  async def test_paused_job_not_fired(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor()

    trigger = InstantTrigger()
    job = scheduler.add(trigger)
    job.next_run_at = time.time() - 1
    scheduler.pause(job.job_id)

    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert len(executor.executions) == 0

  @pytest.mark.asyncio
  async def test_is_running_flag(self):
    scheduler = Scheduler(tick_interval=0.05)
    executor = FakeExecutor()

    assert scheduler.is_running is False

    running_during = None

    async def check_and_stop():
      nonlocal running_during
      await asyncio.sleep(0.05)
      running_during = scheduler.is_running
      scheduler.stop()

    asyncio.create_task(check_and_stop())
    await scheduler.start(executor)

    assert running_during is True
    assert scheduler.is_running is False

  @pytest.mark.asyncio
  async def test_callbacks(self):
    started = []
    completed = []
    failed = []

    scheduler = Scheduler(tick_interval=0.05)
    scheduler.on_job_started = started.append
    scheduler.on_job_completed = completed.append
    scheduler.on_job_failed = lambda j, e: failed.append((j, e))

    executor = FakeExecutor()
    trigger = InstantTrigger()
    job = scheduler.add(trigger)
    job.next_run_at = time.time() - 1

    async def stop_soon():
      await asyncio.sleep(0.15)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    assert len(started) >= 1
    assert len(completed) >= 1
    assert len(failed) == 0

  @pytest.mark.asyncio
  async def test_max_concurrent_limit(self):
    scheduler = Scheduler(tick_interval=0.05, max_concurrent=1)
    executor = FakeExecutor()

    # Add two instant triggers
    for _ in range(3):
      trigger = InstantTrigger()
      job = scheduler.add(trigger)
      job.next_run_at = time.time() - 1

    async def stop_soon():
      await asyncio.sleep(0.2)
      scheduler.stop()

    asyncio.create_task(stop_soon())
    await scheduler.start(executor)

    # All should eventually fire
    assert len(executor.executions) >= 3

  @pytest.mark.asyncio
  async def test_save_all(self):
    store = InMemoryJobStore()
    scheduler = Scheduler(store=store)
    j1 = scheduler.add(Interval(seconds=60))
    j2 = scheduler.add(Interval(seconds=120))
    await scheduler.save_all()
    assert await store.get(j1.job_id) is j1
    assert await store.get(j2.job_id) is j2
