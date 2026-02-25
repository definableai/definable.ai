"""Integration tests for Scheduler, trigger exports, and Agent wiring."""

from definable.agent.trigger.interval import Interval
from definable.agent.trigger.oneshot import OneShot


class TestTriggerExports:
  def test_interval_importable_from_trigger(self):
    from definable.agent.trigger import Interval

    assert Interval is not None

  def test_oneshot_importable_from_trigger(self):
    from definable.agent.trigger import OneShot

    assert OneShot is not None

  def test_interval_importable_from_agent(self):
    from definable.agent import Interval

    assert Interval is not None

  def test_oneshot_importable_from_agent(self):
    from definable.agent import OneShot

    assert OneShot is not None


class TestSchedulerExports:
  def test_scheduler_importable_from_agent(self):
    from definable.agent import Scheduler

    assert Scheduler is not None

  def test_scheduled_job_importable(self):
    from definable.agent import ScheduledJob

    assert ScheduledJob is not None

  def test_job_status_importable(self):
    from definable.agent import JobStatus

    assert JobStatus is not None

  def test_scheduler_package_importable(self):
    from definable.agent.scheduler import Scheduler, ScheduledJob, JobStatus, InMemoryJobStore

    assert Scheduler is not None
    assert ScheduledJob is not None
    assert JobStatus is not None
    assert InMemoryJobStore is not None


class TestTriggerBaseBehavior:
  def test_base_trigger_has_next_run(self):
    from definable.agent.trigger.base import BaseTrigger

    assert hasattr(BaseTrigger, "next_run")

  def test_base_trigger_default_next_run(self):
    """Default next_run returns base_time (fire immediately)."""
    from definable.agent.trigger.base import BaseTrigger

    class DummyTrigger(BaseTrigger):
      @property
      def name(self) -> str:
        return "dummy"

    trigger = DummyTrigger()
    assert trigger.next_run(100.0) == 100.0

  def test_interval_next_run_adds_seconds(self):
    trigger = Interval(seconds=30)
    assert trigger.next_run(100.0) == 130.0

  def test_oneshot_next_run_returns_fire_at(self):
    trigger = OneShot(fire_at=999.0)
    assert trigger.next_run(0) == 999.0


class TestSchedulerWithAgent:
  def test_agent_scheduler_property_no_triggers(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    assert agent.scheduler is None

  def test_agent_scheduler_with_interval_trigger(self):
    from definable.agent.testing import create_test_agent

    agent = create_test_agent()
    agent._triggers.append(Interval(seconds=60))
    scheduler = agent.scheduler
    assert scheduler is not None
    assert scheduler.job_count == 1

  def test_agent_scheduler_with_oneshot_trigger(self):
    from definable.agent.testing import create_test_agent
    import time

    agent = create_test_agent()
    agent._triggers.append(OneShot(fire_at=time.time() + 3600))
    scheduler = agent.scheduler
    assert scheduler is not None
    assert scheduler.job_count == 1
