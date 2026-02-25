"""Sliding-window rate limiter for interface message throttling.

Provides an in-memory rate limiter and an InterfaceHook adapter that
plugs into any BaseInterface via its hook pipeline.

Usage::

    from definable.agent.security import RateLimitConfig, RateLimitHook

    interface = TelegramInterface(
        agent=agent,
        bot_token="...",
        hooks=[RateLimitHook(RateLimitConfig(max_requests=10, window_seconds=60))],
    )
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

from definable.utils.log import log_debug, log_warning


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class RateLimitConfig:
  """Rate limiting configuration.

  Attributes:
    max_requests: Maximum requests allowed within the window.
    window_seconds: Duration of the sliding window in seconds.
    lockout_threshold: Number of violations before locking out the sender.
    lockout_duration_seconds: How long the lockout lasts in seconds.
    max_keys: Maximum unique keys to track (prevents memory exhaustion).
  """

  max_requests: int = 10
  window_seconds: int = 60
  lockout_threshold: int = 3
  lockout_duration_seconds: int = 300
  max_keys: int = 10_000


# ------------------------------------------------------------------
# Core Rate Limiter
# ------------------------------------------------------------------


class SlidingWindowRateLimiter:
  """In-memory sliding window rate limiter.

  Tracks request timestamps per key in a deque. Thread-safe via
  asyncio.Lock per key.
  """

  def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
    self._config = config or RateLimitConfig()
    self._windows: Dict[str, Deque[float]] = defaultdict(deque)
    self._violations: Dict[str, int] = defaultdict(int)
    self._lockouts: Dict[str, float] = {}
    self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

  async def check(self, key: str) -> bool:
    """Check if a request is allowed for the given key.

    Returns True if allowed, False if rate-limited.
    """
    async with self._locks[key]:
      now = time.monotonic()

      # Check lockout
      if key in self._lockouts:
        if now < self._lockouts[key]:
          log_debug(f"Rate limiter: key '{key}' is locked out until {self._lockouts[key]:.0f}")
          return False
        del self._lockouts[key]
        self._violations[key] = 0

      # Prune expired entries
      window = self._windows[key]
      cutoff = now - self._config.window_seconds
      while window and window[0] < cutoff:
        window.popleft()

      # Check limit
      if len(window) >= self._config.max_requests:
        self._violations[key] += 1
        if self._violations[key] >= self._config.lockout_threshold:
          self._lockouts[key] = now + self._config.lockout_duration_seconds
          log_warning(f"Rate limiter: locking out key '{key}' for {self._config.lockout_duration_seconds}s")
        return False

      # Allow and record
      window.append(now)
      self._enforce_max_keys()
      return True

  async def is_locked_out(self, key: str) -> bool:
    """Check if a key is currently locked out."""
    if key in self._lockouts:
      if time.monotonic() < self._lockouts[key]:
        return True
      del self._lockouts[key]
      self._violations[key] = 0
    return False

  def reset(self, key: str) -> None:
    """Reset rate limit state for a specific key."""
    self._windows.pop(key, None)
    self._violations.pop(key, None)
    self._lockouts.pop(key, None)
    self._locks.pop(key, None)

  def reset_all(self) -> None:
    """Reset all rate limit state."""
    self._windows.clear()
    self._violations.clear()
    self._lockouts.clear()
    self._locks.clear()

  def _enforce_max_keys(self) -> None:
    """Evict oldest keys if we exceed max_keys."""
    if len(self._windows) > self._config.max_keys:
      # Remove the key with the oldest last-request timestamp
      oldest_key = min(self._windows, key=lambda k: self._windows[k][-1] if self._windows[k] else 0)
      self.reset(oldest_key)


# ------------------------------------------------------------------
# InterfaceHook adapter
# ------------------------------------------------------------------


class RateLimitHook:
  """Hook that enforces rate limits on inbound interface messages.

  Attach to any interface via its ``hooks`` parameter. When the rate
  limit is exceeded, the hook vetoes the message and optionally sends
  a rejection reply.

  This class works as an InterfaceHook — it implements
  ``on_message_received`` which returns False to veto.
  """

  def __init__(
    self,
    config: Optional[RateLimitConfig] = None,
    *,
    key_fn: Optional[Callable] = None,
    rejection_message: str = "You're sending messages too quickly. Please wait a moment.",
  ) -> None:
    self._limiter = SlidingWindowRateLimiter(config)
    self._key_fn = key_fn
    self._rejection_message = rejection_message

  @property
  def limiter(self) -> SlidingWindowRateLimiter:
    """Access the underlying rate limiter for programmatic control."""
    return self._limiter

  async def on_message_received(self, message: object) -> Optional[bool]:
    """Called by BaseInterface before processing a message.

    Returns False to veto the message (rate limited), None to pass through.
    """
    # Extract key from message
    key = self._extract_key(message)
    if key is None:
      return None  # Can't rate-limit without a key

    allowed = await self._limiter.check(key)
    if not allowed:
      log_debug(f"RateLimitHook: throttling message from '{key}'")
      return False

    return None  # Allow

  def _extract_key(self, message: object) -> Optional[str]:
    """Extract rate limit key from a message object."""
    if self._key_fn:
      return self._key_fn(message)
    # Try common attributes
    for attr in ("sender_id", "user_id", "platform_user_id", "from_id"):
      val = getattr(message, attr, None)
      if val is not None:
        return str(val)
    return None
