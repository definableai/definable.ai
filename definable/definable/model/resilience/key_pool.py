"""KeyPool — thread-safe multi-key rotation with health tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SelectionStrategy(str, Enum):
  """Key selection strategy."""

  ROUND_ROBIN = "round_robin"
  LEAST_RECENTLY_USED = "lru"


@dataclass
class KeyHealth:
  """Health tracking for a single API key."""

  key: str
  success_count: int = 0
  failure_count: int = 0
  rate_limit_count: int = 0
  last_used: float = 0.0
  cooldown_until: float = 0.0
  consecutive_failures: int = 0

  @property
  def is_available(self) -> bool:
    """True if not in cooldown."""
    return time.time() >= self.cooldown_until

  @property
  def total_requests(self) -> int:
    """Total number of requests (success + failure + rate_limit)."""
    return self.success_count + self.failure_count + self.rate_limit_count

  @property
  def success_rate(self) -> float:
    """Success rate (0.0 to 1.0). Returns 1.0 when no requests."""
    total = self.total_requests
    if total == 0:
      return 1.0
    return self.success_count / total

  @property
  def error_rate(self) -> float:
    """Error rate (0.0 to 1.0)."""
    total = self.total_requests
    if total == 0:
      return 0.0
    return (self.failure_count + self.rate_limit_count) / total


class KeyPool:
  """Thread-safe pool of API keys with rotation and health tracking.

  Supports round-robin and LRU selection strategies. Keys that hit
  rate limits are placed in exponential backoff cooldown.

  Args:
    keys: List of API key strings (must be unique).
    strategy: Selection strategy (default: round_robin).
    base_cooldown: Base cooldown duration in seconds for rate-limited keys.
    max_cooldown: Maximum cooldown duration in seconds.

  Example::

    pool = KeyPool(keys=["sk-1", "sk-2", "sk-3"])
    key = pool.acquire()
    pool.mark_success(key)
  """

  def __init__(
    self,
    keys: List[str],
    *,
    strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
    base_cooldown: float = 60.0,
    max_cooldown: float = 300.0,
  ) -> None:
    if not keys:
      raise ValueError("KeyPool requires at least one key")
    if len(keys) != len(set(keys)):
      raise ValueError("Keys must be unique")
    self._keys = list(keys)
    self._strategy = strategy
    self._base_cooldown = base_cooldown
    self._max_cooldown = max_cooldown
    self._health: Dict[str, KeyHealth] = {k: KeyHealth(key=k) for k in keys}
    self._index = 0
    self._lock = threading.Lock()

  @property
  def size(self) -> int:
    """Number of keys in the pool."""
    return len(self._keys)

  @property
  def strategy(self) -> SelectionStrategy:
    """Current selection strategy."""
    return self._strategy

  def acquire(self) -> str:
    """Acquire the next available key.

    Returns:
      An API key string.

    Raises:
      RuntimeError: If all keys are in cooldown.
    """
    with self._lock:
      if self._strategy == SelectionStrategy.ROUND_ROBIN:
        return self._acquire_round_robin()
      return self._acquire_lru()

  def _acquire_round_robin(self) -> str:
    """Select via round-robin, skipping cooled-down keys."""
    n = len(self._keys)
    for _ in range(n):
      key = self._keys[self._index % n]
      self._index += 1
      health = self._health[key]
      if health.is_available:
        health.last_used = time.time()
        return key
    raise RuntimeError("All keys are in cooldown")

  def _acquire_lru(self) -> str:
    """Select the least recently used available key."""
    available = [h for h in self._health.values() if h.is_available]
    if not available:
      raise RuntimeError("All keys are in cooldown")
    chosen = min(available, key=lambda h: h.last_used)
    chosen.last_used = time.time()
    return chosen.key

  def mark_success(self, key: str) -> None:
    """Record a successful request."""
    with self._lock:
      health = self._health.get(key)
      if health:
        health.success_count += 1
        health.consecutive_failures = 0

  def mark_failure(self, key: str) -> None:
    """Record a failed request."""
    with self._lock:
      health = self._health.get(key)
      if health:
        health.failure_count += 1
        health.consecutive_failures += 1

  def mark_rate_limited(self, key: str) -> None:
    """Record a rate limit (429) and apply exponential backoff cooldown."""
    with self._lock:
      health = self._health.get(key)
      if health:
        health.rate_limit_count += 1
        health.consecutive_failures += 1
        backoff = min(
          self._base_cooldown * (2 ** (health.consecutive_failures - 1)),
          self._max_cooldown,
        )
        health.cooldown_until = time.time() + backoff

  def get_health(self, key: str) -> Optional[KeyHealth]:
    """Get health info for a specific key."""
    return self._health.get(key)

  def all_health(self) -> List[KeyHealth]:
    """Get health info for all keys."""
    return list(self._health.values())

  def available_count(self) -> int:
    """Number of keys currently available (not in cooldown)."""
    return sum(1 for h in self._health.values() if h.is_available)

  def reset(self, key: Optional[str] = None) -> None:
    """Reset health tracking for one or all keys."""
    with self._lock:
      if key is not None:
        if key in self._health:
          self._health[key] = KeyHealth(key=key)
      else:
        self._health = {k: KeyHealth(key=k) for k in self._keys}
        self._index = 0

  def __len__(self) -> int:
    return len(self._keys)

  def __repr__(self) -> str:
    return f"<KeyPool keys={len(self._keys)} available={self.available_count()} strategy={self._strategy.value}>"
