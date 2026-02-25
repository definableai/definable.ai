"""Tests for KeyPool — multi-key rotation with cooldown tracking."""

import time

import pytest

from definable.model.resilience.key_pool import KeyHealth, KeyPool, SelectionStrategy


class TestKeyPoolInit:
  def test_single_key(self):
    pool = KeyPool(keys=["sk-key1"])
    assert pool.size == 1
    assert pool.strategy == SelectionStrategy.ROUND_ROBIN

  def test_multiple_keys(self):
    pool = KeyPool(keys=["sk-1", "sk-2", "sk-3"])
    assert pool.size == 3

  def test_custom_strategy(self):
    pool = KeyPool(keys=["sk-1"], strategy=SelectionStrategy.LEAST_RECENTLY_USED)
    assert pool.strategy == SelectionStrategy.LEAST_RECENTLY_USED

  def test_empty_keys_raises(self):
    with pytest.raises(ValueError, match="at least one key"):
      KeyPool(keys=[])

  def test_duplicate_keys_raises(self):
    with pytest.raises(ValueError, match="must be unique"):
      KeyPool(keys=["sk-1", "sk-1"])


class TestKeyPoolAcquire:
  def test_round_robin_cycles(self):
    pool = KeyPool(keys=["sk-a", "sk-b", "sk-c"])
    keys = [pool.acquire() for _ in range(6)]
    assert keys == ["sk-a", "sk-b", "sk-c", "sk-a", "sk-b", "sk-c"]

  def test_lru_selects_least_recent(self):
    pool = KeyPool(keys=["sk-a", "sk-b"], strategy=SelectionStrategy.LEAST_RECENTLY_USED)
    # First call picks sk-a (both have last_used=0, sorted by insertion order)
    k1 = pool.acquire()
    # Second call picks the other one
    k2 = pool.acquire()
    assert k1 != k2

  def test_acquire_skips_cooldown_keys(self):
    pool = KeyPool(keys=["sk-a", "sk-b", "sk-c"])
    # Put sk-a in cooldown
    pool.mark_rate_limited("sk-a")
    # Should skip sk-a
    k = pool.acquire()
    assert k == "sk-b"

  def test_all_keys_in_cooldown_raises(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_rate_limited("sk-a")
    with pytest.raises(RuntimeError, match="All keys are in cooldown"):
      pool.acquire()


class TestKeyPoolHealth:
  def test_mark_success(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_success("sk-a")
    pool.mark_success("sk-a")
    health = pool.get_health("sk-a")
    assert health is not None
    assert health.success_count == 2
    assert health.failure_count == 0
    assert health.consecutive_failures == 0

  def test_mark_failure(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_failure("sk-a")
    health = pool.get_health("sk-a")
    assert health is not None
    assert health.failure_count == 1
    assert health.consecutive_failures == 1

  def test_mark_rate_limited_sets_cooldown(self):
    pool = KeyPool(keys=["sk-a"], base_cooldown=10.0)
    pool.mark_rate_limited("sk-a")
    health = pool.get_health("sk-a")
    assert health is not None
    assert health.rate_limit_count == 1
    assert health.cooldown_until > time.time()

  def test_exponential_backoff_on_repeated_rate_limits(self):
    pool = KeyPool(keys=["sk-a"], base_cooldown=10.0, max_cooldown=100.0)
    pool.mark_rate_limited("sk-a")
    h1 = pool.get_health("sk-a")
    cooldown_1 = h1.cooldown_until  # type: ignore[union-attr]

    # Force cooldown to expire for next mark
    pool._health["sk-a"].cooldown_until = 0.0
    pool.mark_rate_limited("sk-a")
    h2 = pool.get_health("sk-a")
    cooldown_2 = h2.cooldown_until  # type: ignore[union-attr]

    # Second cooldown should be longer
    assert cooldown_2 > cooldown_1

  def test_success_resets_consecutive_failures(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_failure("sk-a")
    pool.mark_failure("sk-a")
    assert pool.get_health("sk-a").consecutive_failures == 2  # type: ignore[union-attr]
    pool.mark_success("sk-a")
    assert pool.get_health("sk-a").consecutive_failures == 0  # type: ignore[union-attr]

  def test_get_health_unknown_key(self):
    pool = KeyPool(keys=["sk-a"])
    assert pool.get_health("sk-unknown") is None

  def test_mark_unknown_key_is_noop(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_success("sk-unknown")  # Should not raise
    pool.mark_failure("sk-unknown")
    pool.mark_rate_limited("sk-unknown")

  def test_success_rate(self):
    pool = KeyPool(keys=["sk-a"])
    pool.mark_success("sk-a")
    pool.mark_success("sk-a")
    pool.mark_failure("sk-a")
    health = pool.get_health("sk-a")
    assert health is not None
    assert abs(health.success_rate - 2 / 3) < 0.01

  def test_success_rate_no_requests(self):
    pool = KeyPool(keys=["sk-a"])
    health = pool.get_health("sk-a")
    assert health is not None
    assert health.success_rate == 1.0  # Default when no requests


class TestKeyPoolUtilities:
  def test_all_health(self):
    pool = KeyPool(keys=["sk-a", "sk-b"])
    pool.mark_success("sk-a")
    healths = pool.all_health()
    assert len(healths) == 2

  def test_available_count(self):
    pool = KeyPool(keys=["sk-a", "sk-b", "sk-c"])
    assert pool.available_count() == 3
    pool.mark_rate_limited("sk-a")
    assert pool.available_count() == 2

  def test_reset_single_key(self):
    pool = KeyPool(keys=["sk-a", "sk-b"])
    pool.mark_failure("sk-a")
    pool.mark_failure("sk-b")
    pool.reset("sk-a")
    assert pool.get_health("sk-a").failure_count == 0  # type: ignore[union-attr]
    assert pool.get_health("sk-b").failure_count == 1  # type: ignore[union-attr]

  def test_reset_all_keys(self):
    pool = KeyPool(keys=["sk-a", "sk-b"])
    pool.mark_failure("sk-a")
    pool.mark_rate_limited("sk-b")
    pool.reset()
    for h in pool.all_health():
      assert h.failure_count == 0
      assert h.rate_limit_count == 0
      assert h.cooldown_until == 0.0

  def test_cooldown_cap(self):
    pool = KeyPool(keys=["sk-a"], base_cooldown=10.0, max_cooldown=50.0)
    # Mark rate limited many times
    for _ in range(20):
      pool._health["sk-a"].cooldown_until = 0.0
      pool.mark_rate_limited("sk-a")
    health = pool.get_health("sk-a")
    # Cooldown should not exceed max_cooldown + current time
    assert health.cooldown_until <= time.time() + 50.0 + 1.0  # type: ignore[union-attr]


class TestKeyHealth:
  def test_is_available_fresh(self):
    h = KeyHealth(key="sk-1")
    assert h.is_available is True

  def test_is_available_in_cooldown(self):
    h = KeyHealth(key="sk-1", cooldown_until=time.time() + 100)
    assert h.is_available is False

  def test_is_available_cooldown_expired(self):
    h = KeyHealth(key="sk-1", cooldown_until=time.time() - 1)
    assert h.is_available is True

  def test_total_requests(self):
    h = KeyHealth(key="sk-1", success_count=5, failure_count=3)
    assert h.total_requests == 8
