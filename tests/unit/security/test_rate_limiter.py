"""Tests for SlidingWindowRateLimiter and RateLimitHook."""

import pytest

from definable.agent.security.rate_limiter import (
  RateLimitConfig,
  RateLimitHook,
  SlidingWindowRateLimiter,
)


# ------------------------------------------------------------------
# SlidingWindowRateLimiter
# ------------------------------------------------------------------


class TestSlidingWindowRateLimiter:
  @pytest.mark.asyncio
  async def test_allows_within_limit(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=5, window_seconds=60))
    for _ in range(5):
      assert await limiter.check("user1") is True

  @pytest.mark.asyncio
  async def test_blocks_over_limit(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=3, window_seconds=60))
    for _ in range(3):
      assert await limiter.check("user1") is True
    assert await limiter.check("user1") is False

  @pytest.mark.asyncio
  async def test_separate_keys_independent(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=2, window_seconds=60))
    assert await limiter.check("user1") is True
    assert await limiter.check("user1") is True
    assert await limiter.check("user1") is False  # user1 exhausted
    assert await limiter.check("user2") is True  # user2 still fresh

  @pytest.mark.asyncio
  async def test_lockout_after_violations(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=1, window_seconds=60, lockout_threshold=2, lockout_duration_seconds=300))
    assert await limiter.check("user1") is True
    assert await limiter.check("user1") is False  # violation 1
    assert await limiter.check("user1") is False  # violation 2 → lockout
    assert await limiter.is_locked_out("user1") is True

  @pytest.mark.asyncio
  async def test_reset_clears_state(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=1, window_seconds=60))
    assert await limiter.check("user1") is True
    assert await limiter.check("user1") is False
    limiter.reset("user1")
    assert await limiter.check("user1") is True

  @pytest.mark.asyncio
  async def test_reset_all(self):
    limiter = SlidingWindowRateLimiter(RateLimitConfig(max_requests=1, window_seconds=60))
    assert await limiter.check("a") is True
    assert await limiter.check("b") is True
    assert await limiter.check("a") is False
    limiter.reset_all()
    assert await limiter.check("a") is True


# ------------------------------------------------------------------
# RateLimitHook
# ------------------------------------------------------------------


class TestRateLimitHook:
  @pytest.mark.asyncio
  async def test_allows_message_under_limit(self):
    hook = RateLimitHook(RateLimitConfig(max_requests=5, window_seconds=60))

    class FakeMsg:
      sender_id = "user1"

    result = await hook.on_message_received(FakeMsg())
    assert result is None  # None = pass through

  @pytest.mark.asyncio
  async def test_blocks_message_over_limit(self):
    hook = RateLimitHook(RateLimitConfig(max_requests=1, window_seconds=60))

    class FakeMsg:
      sender_id = "user1"

    await hook.on_message_received(FakeMsg())
    result = await hook.on_message_received(FakeMsg())
    assert result is False

  @pytest.mark.asyncio
  async def test_custom_key_fn(self):
    hook = RateLimitHook(
      RateLimitConfig(max_requests=1, window_seconds=60),
      key_fn=lambda msg: getattr(msg, "custom_id", None),
    )

    class FakeMsg:
      custom_id = "custom_key"

    await hook.on_message_received(FakeMsg())
    result = await hook.on_message_received(FakeMsg())
    assert result is False

  @pytest.mark.asyncio
  async def test_no_key_passes_through(self):
    hook = RateLimitHook(RateLimitConfig(max_requests=1, window_seconds=60))

    class FakeMsg:
      pass  # No sender_id attribute

    # Should pass through (can't rate limit without a key)
    result = await hook.on_message_received(FakeMsg())
    assert result is None

  def test_limiter_property(self):
    hook = RateLimitHook()
    assert isinstance(hook.limiter, SlidingWindowRateLimiter)
