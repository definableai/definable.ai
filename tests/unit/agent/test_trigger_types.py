"""Tests for Interval, OneShot, and trigger serialization."""

import time

import pytest

from definable.agent.trigger.interval import Interval
from definable.agent.trigger.oneshot import OneShot


class TestInterval:
  def test_basic(self):
    trigger = Interval(seconds=60)
    assert trigger.seconds == 60
    assert trigger.name == "interval(60s)"

  def test_fractional_seconds(self):
    trigger = Interval(seconds=0.5)
    assert trigger.seconds == 0.5
    assert trigger.name == "interval(0.5s)"

  def test_next_run(self):
    trigger = Interval(seconds=30)
    base = 1000.0
    assert trigger.next_run(base) == 1030.0

  def test_next_run_repeats(self):
    trigger = Interval(seconds=10)
    t1 = trigger.next_run(100.0)
    t2 = trigger.next_run(t1)
    assert t1 == 110.0
    assert t2 == 120.0

  def test_zero_seconds_raises(self):
    with pytest.raises(ValueError, match="must be positive"):
      Interval(seconds=0)

  def test_negative_seconds_raises(self):
    with pytest.raises(ValueError, match="must be positive"):
      Interval(seconds=-5)

  def test_has_name_property(self):
    trigger = Interval(seconds=300)
    assert "interval" in trigger.name

  def test_handler_default_none(self):
    trigger = Interval(seconds=60)
    assert trigger.handler is None

  def test_agent_default_none(self):
    trigger = Interval(seconds=60)
    assert trigger.agent is None


class TestOneShot:
  def test_with_delay(self):
    before = time.time()
    trigger = OneShot(delay=10.0)
    assert trigger.fire_at >= before + 10.0
    assert trigger.fire_at <= time.time() + 10.0 + 0.1

  def test_with_fire_at(self):
    target = time.time() + 3600
    trigger = OneShot(fire_at=target)
    assert trigger.fire_at == target

  def test_fire_at_overrides_delay(self):
    target = 999999999.0
    trigger = OneShot(delay=10.0, fire_at=target)
    assert trigger.fire_at == target

  def test_neither_raises(self):
    with pytest.raises(ValueError, match="requires either"):
      OneShot()

  def test_zero_delay_raises(self):
    with pytest.raises(ValueError, match="requires either"):
      OneShot(delay=0.0)

  def test_negative_delay_raises(self):
    with pytest.raises(ValueError, match="requires either"):
      OneShot(delay=-5.0)

  def test_not_fired_initially(self):
    trigger = OneShot(delay=10.0)
    assert trigger.fired is False

  def test_mark_fired(self):
    trigger = OneShot(delay=10.0)
    trigger.mark_fired()
    assert trigger.fired is True

  def test_next_run_returns_fire_time(self):
    trigger = OneShot(fire_at=12345.0)
    assert trigger.next_run(0) == 12345.0

  def test_next_run_returns_infinity_after_fired(self):
    trigger = OneShot(fire_at=12345.0)
    trigger.mark_fired()
    assert trigger.next_run(0) == float("inf")

  def test_name_contains_oneshot(self):
    trigger = OneShot(delay=10.0)
    assert "oneshot" in trigger.name

  def test_handler_default_none(self):
    trigger = OneShot(delay=60)
    assert trigger.handler is None
