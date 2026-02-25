"""Fallback embedder — chains multiple embedding providers with auto-failover.

When the primary provider fails (rate limit, auth error, timeout), automatically
tries the next provider in the chain.

Usage::

    from definable.knowledge.embedder.fallback import FallbackEmbedder
    from definable.knowledge.embedder.openai import OpenAIEmbedder
    from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

    embedder = FallbackEmbedder(providers=[
        OpenAIEmbedder(),   # Primary
        VoyageAIEmbedder(), # Fallback
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from definable.knowledge.embedder.base import Embedder
from definable.utils.log import log_debug, log_warning


class EmbeddingErrorType(str, Enum):
  """Classification of embedding provider errors."""

  auth = "auth"
  rate_limit = "rate_limit"
  timeout = "timeout"
  network = "network"
  unknown = "unknown"


class EmbeddingError(Exception):
  """Raised when all embedding providers fail.

  Attributes:
    error_type: Classification of the last error.
    provider: Name of the provider that failed.
    original: The original exception.
  """

  def __init__(self, error_type: EmbeddingErrorType, provider: str, original: Exception) -> None:
    self.error_type = error_type
    self.provider = provider
    self.original = original
    super().__init__(f"Embedding failed ({error_type.value}) from {provider}: {original}")


@dataclass
class FallbackEmbedder(Embedder):
  """Embedder that chains multiple providers with automatic failover.

  Tries providers in order. On failure, classifies the error and
  rotates to the next provider. If all fail, raises EmbeddingError.

  Attributes:
    providers: Ordered list of embedder instances (primary first).
  """

  providers: List[Embedder] = field(default_factory=list)
  _active_index: int = field(default=0, repr=False)

  def __post_init__(self) -> None:
    if not self.providers:
      raise ValueError("FallbackEmbedder requires at least one provider.")
    # Inherit dimensions from the primary provider
    self.dimensions = self.providers[0].dimensions

  def get_embedding(self, text: str) -> List[float]:
    """Get embedding, falling back through providers on failure."""
    return self._try_providers(lambda p: p.get_embedding(text))

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Get embedding and usage, with fallback."""
    return self._try_providers(lambda p: p.get_embedding_and_usage(text))

  async def async_get_embedding(self, text: str) -> List[float]:
    """Async get embedding, with fallback."""
    return await self._try_providers_async(lambda p: p.async_get_embedding(text))

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Async get embedding and usage, with fallback."""
    return await self._try_providers_async(lambda p: p.async_get_embedding_and_usage(text))

  def reset(self) -> None:
    """Reset to the primary provider."""
    self._active_index = 0
    self.dimensions = self.providers[0].dimensions

  def _try_providers(self, fn: Any) -> Any:
    """Try calling fn on each provider in order."""
    last_error: Optional[Exception] = None

    for offset in range(len(self.providers)):
      idx = (self._active_index + offset) % len(self.providers)
      provider = self.providers[idx]
      try:
        result = fn(provider)
        # Success — update active index
        if idx != self._active_index:
          log_debug(f"FallbackEmbedder: switched to provider {idx} ({provider.__class__.__name__})")
          self._active_index = idx
          self.dimensions = provider.dimensions
        return result
      except Exception as exc:
        error_type = classify_error(exc)
        log_warning(f"FallbackEmbedder: provider {idx} ({provider.__class__.__name__}) failed ({error_type.value}): {exc}")
        last_error = exc
        continue

    # All providers failed
    raise EmbeddingError(
      error_type=classify_error(last_error) if last_error else EmbeddingErrorType.unknown,
      provider=self.providers[self._active_index].__class__.__name__,
      original=last_error or RuntimeError("No providers available"),
    )

  async def _try_providers_async(self, fn: Any) -> Any:
    """Async version of _try_providers."""
    last_error: Optional[Exception] = None

    for offset in range(len(self.providers)):
      idx = (self._active_index + offset) % len(self.providers)
      provider = self.providers[idx]
      try:
        result = await fn(provider)
        if idx != self._active_index:
          log_debug(f"FallbackEmbedder: switched to provider {idx} ({provider.__class__.__name__})")
          self._active_index = idx
          self.dimensions = provider.dimensions
        return result
      except Exception as exc:
        error_type = classify_error(exc)
        log_warning(f"FallbackEmbedder: provider {idx} ({provider.__class__.__name__}) failed ({error_type.value}): {exc}")
        last_error = exc
        continue

    raise EmbeddingError(
      error_type=classify_error(last_error) if last_error else EmbeddingErrorType.unknown,
      provider=self.providers[self._active_index].__class__.__name__,
      original=last_error or RuntimeError("No providers available"),
    )


def classify_error(exc: Optional[Exception]) -> EmbeddingErrorType:
  """Classify an exception into an EmbeddingErrorType.

  Uses duck typing on exception class names to avoid importing
  provider-specific SDKs.
  """
  if exc is None:
    return EmbeddingErrorType.unknown

  cls_name = type(exc).__name__
  msg = str(exc).lower()

  # Auth errors
  if "authentication" in cls_name.lower() or "auth" in cls_name.lower():
    return EmbeddingErrorType.auth
  if "401" in msg or "403" in msg or "invalid api key" in msg or "unauthorized" in msg:
    return EmbeddingErrorType.auth

  # Rate limit
  if "ratelimit" in cls_name.lower() or "rate_limit" in cls_name.lower():
    return EmbeddingErrorType.rate_limit
  if "429" in msg or "rate limit" in msg or "too many requests" in msg:
    return EmbeddingErrorType.rate_limit

  # Timeout
  if "timeout" in cls_name.lower():
    return EmbeddingErrorType.timeout
  if "timed out" in msg or "timeout" in msg:
    return EmbeddingErrorType.timeout

  # Network
  if "connect" in cls_name.lower() or "network" in cls_name.lower():
    return EmbeddingErrorType.network
  if "connection" in msg or "dns" in msg or "unreachable" in msg:
    return EmbeddingErrorType.network

  return EmbeddingErrorType.unknown
