"""ResilientModel — wraps a Model with key rotation and provider failover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from definable.model.resilience.events import KeyRotatedEvent, ProviderFailoverEvent
from definable.utils.log import log_info

if TYPE_CHECKING:
  from definable.model.base import Model
  from definable.model.resilience.failover import FailoverChain
  from definable.model.resilience.key_pool import KeyPool


@dataclass
class ResilientModel:
  """Wraps a Model with key rotation and provider failover.

  Not a Model subclass — acts as a wrapper that delegates to the inner
  model. Use ``agent.model = ResilientModel(...)`` for transparent
  integration.

  Args:
    inner: The primary model to wrap.
    key_pool: Optional KeyPool for key rotation.
    failover: Optional FailoverChain for provider failover.
    max_key_retries: Max key rotation attempts per call (default 3).
    on_key_rotated: Optional callback for key rotation events.
    on_failover: Optional callback for failover events.

  Example::

    from definable.model.resilience import ResilientModel, KeyPool
    from definable.model.openai import OpenAIChat

    model = ResilientModel(
      inner=OpenAIChat(id="gpt-4o"),
      key_pool=KeyPool(keys=["sk-1", "sk-2", "sk-3"]),
    )
    agent = Agent(model=model)
  """

  inner: Optional["Model"] = field(default=None, repr=False)
  key_pool: Optional["KeyPool"] = field(default=None, repr=False)
  failover: Optional["FailoverChain"] = field(default=None, repr=False)
  max_key_retries: int = 3
  on_key_rotated: Optional[Any] = field(default=None, repr=False)
  on_failover: Optional[Any] = field(default=None, repr=False)

  def __post_init__(self) -> None:
    if self.inner is None:
      raise ValueError("ResilientModel requires an 'inner' model")
    # Mirror identity from inner model
    self.id = self.inner.id
    self.name = getattr(self.inner, "name", None)
    self.provider = getattr(self.inner, "provider", None)
    # Inject initial key if pool provided
    self._current_key: Optional[str] = None
    if self.key_pool is not None:
      self._current_key = self.key_pool.acquire()
      self._inject_key(self.inner, self._current_key)

  def _inject_key(self, model: "Model", key: str) -> None:
    """Inject an API key into a model, clearing cached clients."""
    if hasattr(model, "api_key"):
      model.api_key = key  # type: ignore[attr-defined]
    # Invalidate cached HTTP client
    if hasattr(model, "_client"):
      model._client = None  # type: ignore[attr-defined]
    if hasattr(model, "_async_client"):
      model._async_client = None  # type: ignore[attr-defined]

  def _rotate_key(self) -> bool:
    """Rotate to the next key. Returns True if successful."""
    if self.key_pool is None or self._current_key is None:
      return False

    old_prefix = self._current_key[:8]
    self.key_pool.mark_rate_limited(self._current_key)

    try:
      self._current_key = self.key_pool.acquire()
    except RuntimeError:
      return False

    self._inject_key(self.inner, self._current_key)  # type: ignore[arg-type]
    new_prefix = self._current_key[:8]
    log_info(f"Key rotated: {old_prefix}... → {new_prefix}...")

    if self.on_key_rotated:
      self.on_key_rotated(KeyRotatedEvent(old_key_prefix=old_prefix, new_key_prefix=new_prefix))

    return True

  def _failover(self, error: Exception, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Attempt failover to the next provider in the chain. Returns result or raises."""
    if self.failover is None:
      from definable.exceptions import ModelProviderError

      raise ModelProviderError("All providers exhausted", status_code=502) from error

    old_id = self.id
    for entry in self.failover:
      try:
        result = getattr(entry.model, method_name)(*args, **kwargs)
        # Switch identity to the new provider
        self.inner = entry.model  # type: ignore[misc]
        self.id = entry.model.id
        self.name = getattr(entry.model, "name", None)
        self.provider = getattr(entry.model, "provider", None)
        if self.on_failover:
          self.on_failover(ProviderFailoverEvent(from_model_id=old_id, to_model_id=entry.model.id, reason=str(error)))
        return result
      except Exception:
        continue

    from definable.exceptions import ModelProviderError

    raise ModelProviderError("All providers exhausted", status_code=502) from error

  async def _afailover(self, error: Exception, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Async failover to the next provider in the chain."""
    if self.failover is None:
      from definable.exceptions import ModelProviderError

      raise ModelProviderError("All providers exhausted", status_code=502) from error

    old_id = self.id
    for entry in self.failover:
      try:
        result = await getattr(entry.model, method_name)(*args, **kwargs)
        self.inner = entry.model  # type: ignore[misc]
        self.id = entry.model.id
        self.name = getattr(entry.model, "name", None)
        self.provider = getattr(entry.model, "provider", None)
        if self.on_failover:
          self.on_failover(ProviderFailoverEvent(from_model_id=old_id, to_model_id=entry.model.id, reason=str(error)))
        return result
      except Exception:
        continue

    from definable.exceptions import ModelProviderError

    raise ModelProviderError("All providers exhausted", status_code=502) from error

  # --- Delegation ---

  def __getattr__(self, name: str) -> Any:
    """Proxy unknown attributes to the inner model."""
    # Only block truly internal attrs; allow _parse_provider_* delegation
    if name in ("inner", "key_pool", "failover", "max_key_retries", "on_key_rotated", "on_failover", "_current_key"):
      raise AttributeError(name)
    return getattr(self.inner, name)

  def invoke(self, *args: Any, **kwargs: Any) -> Any:
    """Synchronous invoke with key rotation and failover."""
    return self._call_with_resilience("invoke", *args, **kwargs)

  def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
    """Async invoke — must be awaited."""
    return self._acall_with_resilience("ainvoke", *args, **kwargs)

  def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
    """Synchronous streaming invoke with failover.

    Wraps the inner generator to catch errors during iteration
    and failover to the next provider.
    """
    from definable.exceptions import ModelProviderError

    def _gen() -> Any:
      try:
        yield from getattr(self.inner, "invoke_stream")(*args, **kwargs)
      except ModelProviderError as e:
        yield from self._failover(e, "invoke_stream", *args, **kwargs)

    return _gen()

  def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
    """Async streaming invoke with failover.

    Wraps the inner async generator to catch errors during iteration
    and failover to the next provider.
    """
    from definable.exceptions import ModelProviderError

    async def _agen() -> Any:
      try:
        async for chunk in getattr(self.inner, "ainvoke_stream")(*args, **kwargs):
          yield chunk
      except ModelProviderError as e:
        async for chunk in self._agen_failover(e, *args, **kwargs):
          yield chunk

    return _agen()

  async def _agen_failover(self, error: Exception, *args: Any, **kwargs: Any) -> Any:
    """Async generator failover to next provider."""
    if self.failover is None:
      from definable.exceptions import ModelProviderError

      raise ModelProviderError("All providers exhausted", status_code=502) from error

    old_id = self.id
    for entry in self.failover:
      try:
        async for chunk in entry.model.ainvoke_stream(*args, **kwargs):
          # Switch identity on first successful chunk
          self.inner = entry.model  # type: ignore[misc]
          self.id = entry.model.id
          self.name = getattr(entry.model, "name", None)
          self.provider = getattr(entry.model, "provider", None)
          if self.on_failover:
            self.on_failover(ProviderFailoverEvent(from_model_id=old_id, to_model_id=entry.model.id, reason=str(error)))
            self.on_failover = None  # Only emit once
          yield chunk
        return
      except Exception:
        continue

    from definable.exceptions import ModelProviderError

    raise ModelProviderError("All providers exhausted", status_code=502) from error

  def _call_with_resilience(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a sync method with key rotation on rate limits and failover on errors."""
    from definable.exceptions import ModelProviderError, ModelRateLimitError

    last_error: Optional[ModelProviderError] = None
    for _attempt in range(self.max_key_retries):
      try:
        result = getattr(self.inner, method_name)(*args, **kwargs)
        if self._current_key and self.key_pool:
          self.key_pool.mark_success(self._current_key)
        return result
      except ModelRateLimitError as e:
        last_error = e
        if not self._rotate_key():
          break
      except ModelProviderError as e:
        last_error = e
        if self._current_key and self.key_pool:
          self.key_pool.mark_failure(self._current_key)
        break

    # Attempt failover
    if last_error is not None:
      return self._failover(last_error, method_name, *args, **kwargs)
    raise last_error  # type: ignore[misc]

  async def _acall_with_resilience(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call an async method with key rotation on rate limits and failover on errors."""
    from definable.exceptions import ModelProviderError, ModelRateLimitError

    last_error: Optional[ModelProviderError] = None
    for _attempt in range(self.max_key_retries):
      try:
        result = await getattr(self.inner, method_name)(*args, **kwargs)
        if self._current_key and self.key_pool:
          self.key_pool.mark_success(self._current_key)
        return result
      except ModelRateLimitError as e:
        last_error = e
        if not self._rotate_key():
          break
      except ModelProviderError as e:
        last_error = e
        if self._current_key and self.key_pool:
          self.key_pool.mark_failure(self._current_key)
        break

    # Attempt failover
    if last_error is not None:
      return await self._afailover(last_error, method_name, *args, **kwargs)
    raise last_error  # type: ignore[misc]

  def __repr__(self) -> str:
    inner_id = getattr(self.inner, "id", "?")
    pool_size = len(self.key_pool) if self.key_pool else 0
    failover_size = len(self.failover) if self.failover else 0
    return f"<ResilientModel inner={inner_id} keys={pool_size} failover={failover_size}>"
