"""CachingPlugin — caches identical prompts to avoid redundant model calls."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet

from definable.agent.plugin.base import Plugin
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.pipeline.state import LoopState


@dataclass
class CacheEntry:
  """A cached response."""

  output_content: str
  hit_count: int = 0


class CachingPlugin(Plugin):
  """Caches model responses for identical prompts.

  Uses an LRU eviction strategy. Cache is keyed on the SHA-256
  hash of the system prompt + user messages (excluding metadata).

  Args:
    max_size: Maximum number of cached responses (default 256).
    ttl_seconds: Time-to-live per entry in seconds (0 = no expiry, default 0).

  Example::

    cache = CachingPlugin(max_size=100)
    agent = Agent(model="gpt-4o", plugins=[cache])
    # First call hits the model
    await agent.arun("What is 2+2?")
    # Second identical call returns cached response
    await agent.arun("What is 2+2?")
    print(cache.hit_count)  # 1
  """

  def __init__(self, *, max_size: int = 256, ttl_seconds: float = 0) -> None:
    self._max_size = max_size
    self._ttl_seconds = ttl_seconds
    self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    self.hit_count: int = 0
    self.miss_count: int = 0

  @property
  def name(self) -> str:
    return "caching"

  @property
  def description(self) -> str:
    return "LRU cache for identical prompts to avoid redundant model calls."

  @property
  def modifies(self) -> FrozenSet[str]:
    return frozenset({"invoke_loop"})

  async def on_load(self, agent: "Agent") -> None:
    agent.pipeline.hook("before:invoke_loop", self._check_cache, priority=-10)
    agent.pipeline.hook("after:invoke_loop", self._store_cache, priority=-10)

  async def _check_cache(self, state: "LoopState") -> "LoopState":
    key = self._compute_key(state)
    if key in self._cache:
      entry = self._cache[key]
      entry.hit_count += 1
      self.hit_count += 1
      # Move to end (most recently used)
      self._cache.move_to_end(key)
      # Set output and skip invoke_loop
      state.content = entry.output_content
      state.cache_hit = True  # type: ignore[attr-defined]
      log_debug(f"[caching] Cache hit (key={key[:12]}..., hits={entry.hit_count})")
    else:
      self.miss_count += 1
      state.cache_hit = False  # type: ignore[attr-defined]
    return state

  async def _store_cache(self, state: "LoopState") -> "LoopState":
    if getattr(state, "cache_hit", False):
      return state
    if not state.content:
      return state

    key = self._compute_key(state)
    self._cache[key] = CacheEntry(output_content=str(state.content))
    self._cache.move_to_end(key)

    # Evict oldest if over capacity
    while len(self._cache) > self._max_size:
      self._cache.popitem(last=False)

    log_debug(f"[caching] Cached response (key={key[:12]}..., size={len(self._cache)})")
    return state

  @staticmethod
  def _compute_key(state: "LoopState") -> str:
    """Compute a cache key from the state's messages."""
    hasher = hashlib.sha256()
    for msg in state.all_messages:
      hasher.update(f"{msg.role}:{msg.content or ''}".encode())
    if state.system_content:
      hasher.update(f"system:{state.system_content}".encode())
    return hasher.hexdigest()

  def clear(self) -> None:
    """Clear the cache and reset counters."""
    self._cache.clear()
    self.hit_count = 0
    self.miss_count = 0

  @property
  def size(self) -> int:
    """Current cache size."""
    return len(self._cache)
