"""Tests for FallbackEmbedder and error classification."""

import pytest

from definable.knowledge.embedder.base import Embedder
from definable.knowledge.embedder.fallback import (
  EmbeddingError,
  EmbeddingErrorType,
  FallbackEmbedder,
  classify_error,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Mock embedders
# ------------------------------------------------------------------


@dataclass
class MockEmbedder(Embedder):
  name: str = "mock"
  dimensions: int = 3

  def get_embedding(self, text: str) -> List[float]:
    return [0.1, 0.2, 0.3]

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    return [0.1, 0.2, 0.3], {"tokens": 5}

  async def async_get_embedding(self, text: str) -> List[float]:
    return [0.1, 0.2, 0.3]

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    return [0.1, 0.2, 0.3], {"tokens": 5}


@dataclass
class FailingEmbedder(Embedder):
  name: str = "failing"
  dimensions: int = 3
  error: Exception = field(default_factory=lambda: RuntimeError("embed failed"))

  def get_embedding(self, text: str) -> List[float]:
    raise self.error

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    raise self.error

  async def async_get_embedding(self, text: str) -> List[float]:
    raise self.error

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    raise self.error


# ------------------------------------------------------------------
# classify_error
# ------------------------------------------------------------------


class TestClassifyError:
  def test_none_returns_unknown(self):
    assert classify_error(None) == EmbeddingErrorType.unknown

  def test_auth_by_name(self):
    class AuthenticationError(Exception):
      pass

    assert classify_error(AuthenticationError("bad key")) == EmbeddingErrorType.auth

  def test_auth_by_message(self):
    assert classify_error(RuntimeError("401 Unauthorized")) == EmbeddingErrorType.auth
    assert classify_error(RuntimeError("invalid api key")) == EmbeddingErrorType.auth

  def test_rate_limit_by_name(self):
    class RateLimitError(Exception):
      pass

    assert classify_error(RateLimitError("too fast")) == EmbeddingErrorType.rate_limit

  def test_rate_limit_by_message(self):
    assert classify_error(RuntimeError("429 too many requests")) == EmbeddingErrorType.rate_limit

  def test_timeout_by_name(self):
    class CustomTimeoutError(Exception):
      pass

    assert classify_error(CustomTimeoutError("timed out")) == EmbeddingErrorType.timeout

  def test_network_by_message(self):
    assert classify_error(RuntimeError("connection refused")) == EmbeddingErrorType.network

  def test_unknown_for_generic(self):
    assert classify_error(RuntimeError("something else")) == EmbeddingErrorType.unknown


# ------------------------------------------------------------------
# FallbackEmbedder
# ------------------------------------------------------------------


class TestFallbackEmbedder:
  def test_requires_providers(self):
    with pytest.raises(ValueError, match="at least one provider"):
      FallbackEmbedder(providers=[])

  def test_uses_primary(self):
    primary = MockEmbedder(name="primary")
    fallback = MockEmbedder(name="fallback")
    embedder = FallbackEmbedder(providers=[primary, fallback])
    result = embedder.get_embedding("test")
    assert result == [0.1, 0.2, 0.3]

  def test_falls_back_on_failure(self):
    primary = FailingEmbedder(name="failing")
    fallback = MockEmbedder(name="working")
    embedder = FallbackEmbedder(providers=[primary, fallback])
    result = embedder.get_embedding("test")
    assert result == [0.1, 0.2, 0.3]
    assert embedder._active_index == 1

  def test_raises_when_all_fail(self):
    e1 = FailingEmbedder(name="fail1")
    e2 = FailingEmbedder(name="fail2")
    embedder = FallbackEmbedder(providers=[e1, e2])
    with pytest.raises(EmbeddingError):
      embedder.get_embedding("test")

  @pytest.mark.asyncio
  async def test_async_fallback(self):
    primary = FailingEmbedder(name="failing")
    fallback = MockEmbedder(name="working")
    embedder = FallbackEmbedder(providers=[primary, fallback])
    result = await embedder.async_get_embedding("test")
    assert result == [0.1, 0.2, 0.3]

  def test_reset(self):
    primary = FailingEmbedder(name="failing")
    fallback = MockEmbedder(name="working", dimensions=5)
    embedder = FallbackEmbedder(providers=[primary, fallback])
    embedder.get_embedding("test")
    assert embedder._active_index == 1
    assert embedder.dimensions == 5
    embedder.reset()
    assert embedder._active_index == 0
    assert embedder.dimensions == 3

  def test_inherits_primary_dimensions(self):
    primary = MockEmbedder(dimensions=768)
    fallback = MockEmbedder(dimensions=1536)
    embedder = FallbackEmbedder(providers=[primary, fallback])
    assert embedder.dimensions == 768

  def test_get_embedding_and_usage(self):
    primary = MockEmbedder()
    embedder = FallbackEmbedder(providers=[primary])
    embedding, usage = embedder.get_embedding_and_usage("test")
    assert embedding == [0.1, 0.2, 0.3]
    assert usage == {"tokens": 5}
