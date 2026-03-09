"""Unit tests for the Cron trigger type.

Covers initialization, schedule validation, name property, and next_run.
"""

import time

import pytest


@pytest.mark.unit
class TestCronTrigger:
  """Tests for the Cron trigger."""

  @pytest.fixture
  def croniter_available(self):
    """Skip if croniter is not installed."""
    pytest.importorskip("croniter", reason="croniter not installed")

  def test_valid_schedule(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("*/5 * * * *")
    assert c.schedule == "*/5 * * * *"
    assert c.timezone == "UTC"

  def test_custom_timezone(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("0 9 * * *", timezone="US/Eastern")
    assert c.timezone == "US/Eastern"

  def test_invalid_schedule_raises(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    with pytest.raises(ValueError, match="Invalid cron expression"):
      Cron("not a valid cron")

  def test_name_property(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("0 9 * * *")
    assert c.name == "cron(0 9 * * *)"

  def test_next_run_returns_future_timestamp(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("*/1 * * * *")  # every minute
    now = time.time()
    next_time = c.next_run(now)
    assert next_time > now

  def test_next_run_chaining(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("*/5 * * * *")  # every 5 minutes
    now = time.time()
    first = c.next_run(now)
    second = c.next_run(first)
    # Second should be ~5 minutes after first
    diff = second - first
    assert 250 < diff < 350  # roughly 300 seconds (5 min)

  def test_is_base_trigger_subclass(self, croniter_available):
    from definable.agent.trigger.base import BaseTrigger
    from definable.agent.trigger.cron import Cron

    c = Cron("* * * * *")
    assert isinstance(c, BaseTrigger)

  def test_handler_default_none(self, croniter_available):
    from definable.agent.trigger.cron import Cron

    c = Cron("* * * * *")
    assert c.handler is None
    assert c.agent is None

  def test_missing_croniter_raises_import_error(self):
    """If croniter is not installed, a helpful ImportError is raised."""
    # We can't easily test this without actually uninstalling croniter,
    # so we just verify the class exists and the error message format
    # is documented in the code. Skip if croniter IS installed.
    try:
      import croniter  # noqa: F401

      pytest.skip("croniter is installed; cannot test ImportError path")
    except ImportError:
      from definable.agent.trigger import cron

      with pytest.raises(ImportError, match="croniter is required"):
        cron.Cron("* * * * *")
