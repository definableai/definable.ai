"""Tests for ResilientModel — key rotation and provider failover."""

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any
import pytest

from definable.exceptions import ModelProviderError, ModelRateLimitError
from definable.model.base import Model
from definable.model.resilience.events import KeyRotatedEvent, ProviderFailoverEvent
from definable.model.resilience.failover import FailoverChain, FailoverEntry
from definable.model.resilience.key_pool import KeyPool
from definable.model.resilience.resilient import ResilientModel
from definable.model.response import ModelResponse


@dataclass
class FakeModel(Model):
  """Minimal concrete Model for testing."""

  id: str = "fake-model"
  name: str = "FakeModel"  # type: ignore[assignment]
  provider: str = "fake"  # type: ignore[assignment]
  api_key: str = ""
  client: Any = None
  async_client: Any = None

  _invoke_side_effect: Any = None
  _ainvoke_side_effect: Any = None
  _call_count: int = 0

  def invoke(self, *args, **kwargs) -> ModelResponse:
    self._call_count += 1
    if self._invoke_side_effect is not None:
      exc = self._invoke_side_effect
      if isinstance(exc, list):
        exc = exc.pop(0) if exc else None
      if isinstance(exc, Exception):
        raise exc
    return ModelResponse(content="ok")

  async def ainvoke(self, *args, **kwargs) -> ModelResponse:
    self._call_count += 1
    if self._ainvoke_side_effect is not None:
      exc = self._ainvoke_side_effect
      if isinstance(exc, list):
        exc = exc.pop(0) if exc else None
      if isinstance(exc, Exception):
        raise exc
    return ModelResponse(content="ok")

  def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
    if self._invoke_side_effect is not None:
      raise self._invoke_side_effect
    yield ModelResponse(content="chunk")

  async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
    if self._ainvoke_side_effect is not None:
      raise self._ainvoke_side_effect
    yield ModelResponse(content="chunk")

  def _parse_provider_response(self, response, **kwargs) -> ModelResponse:
    return ModelResponse(content="parsed")

  def _parse_provider_response_delta(self, response, **kwargs) -> ModelResponse:
    return ModelResponse(content="delta")


class TestResilientModelInit:
  def test_requires_inner(self):
    with pytest.raises(ValueError, match="requires an 'inner' model"):
      ResilientModel()

  def test_mirrors_inner_identity(self):
    inner = FakeModel(id="gpt-4o", name="OpenAIChat", provider="OpenAI")
    model = ResilientModel(inner=inner)
    assert model.id == "gpt-4o"
    assert model.name == "OpenAIChat"
    assert model.provider == "OpenAI"

  def test_with_key_pool(self):
    inner = FakeModel()
    pool = KeyPool(keys=["sk-1", "sk-2"])
    ResilientModel(inner=inner, key_pool=pool)
    # Key should be injected into inner
    assert inner.api_key in ("sk-1", "sk-2")

  def test_with_failover(self):
    inner = FakeModel(id="primary")
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=inner, failover=chain)
    assert model.failover is not None


class TestResilientModelInvoke:
  def test_successful_invoke(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    result = model.invoke()
    assert result.content == "ok"

  def test_key_rotation_on_rate_limit(self):
    inner = FakeModel()
    inner._invoke_side_effect = [
      ModelRateLimitError("rate limited"),
      None,  # Second call succeeds
    ]
    pool = KeyPool(keys=["sk-1", "sk-2"])
    model = ResilientModel(inner=inner, key_pool=pool)
    result = model.invoke()
    assert result.content == "ok"

  def test_key_rotated_event_emitted(self):
    events: list[Any] = []
    inner = FakeModel()
    inner._invoke_side_effect = [
      ModelRateLimitError("rate limited"),
      None,
    ]
    pool = KeyPool(keys=["sk-1", "sk-2"])
    model = ResilientModel(inner=inner, key_pool=pool, on_key_rotated=events.append)
    model.invoke()
    assert len(events) == 1
    assert isinstance(events[0], KeyRotatedEvent)
    assert events[0].reason == "rate_limited"

  def test_failover_on_provider_error(self):
    primary = FakeModel(id="primary")
    primary._invoke_side_effect = ModelProviderError("server error", status_code=500)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain)
    result = model.invoke()
    assert result.content == "ok"
    assert model.id == "fallback"

  def test_failover_event_emitted(self):
    events: list[Any] = []
    primary = FakeModel(id="primary")
    primary._invoke_side_effect = ModelProviderError("down", status_code=500)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain, on_failover=events.append)
    model.invoke()
    assert len(events) == 1
    assert isinstance(events[0], ProviderFailoverEvent)
    assert events[0].from_model_id == "primary"
    assert events[0].to_model_id == "fallback"

  def test_all_providers_exhausted_raises(self):
    primary = FakeModel(id="primary")
    primary._invoke_side_effect = ModelProviderError("down", status_code=500)
    model = ResilientModel(inner=primary)
    with pytest.raises(ModelProviderError, match="All providers exhausted"):
      model.invoke()

  def test_non_retryable_error_skips_to_failover(self):
    primary = FakeModel(id="primary")
    primary._invoke_side_effect = ModelProviderError("bad request", status_code=400)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain)
    model.invoke()
    assert model.id == "fallback"


class TestResilientModelAsync:
  @pytest.mark.asyncio
  async def test_successful_ainvoke(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    result = await model.ainvoke()
    assert result.content == "ok"

  @pytest.mark.asyncio
  async def test_key_rotation_ainvoke(self):
    inner = FakeModel()
    inner._ainvoke_side_effect = [
      ModelRateLimitError("rate limited"),
      None,
    ]
    pool = KeyPool(keys=["sk-1", "sk-2"])
    model = ResilientModel(inner=inner, key_pool=pool)
    result = await model.ainvoke()
    assert result.content == "ok"

  @pytest.mark.asyncio
  async def test_failover_ainvoke(self):
    primary = FakeModel(id="primary")
    primary._ainvoke_side_effect = ModelProviderError("down", status_code=500)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain)
    result = await model.ainvoke()
    assert result.content == "ok"

  @pytest.mark.asyncio
  async def test_all_exhausted_ainvoke(self):
    inner = FakeModel()
    inner._ainvoke_side_effect = ModelProviderError("down", status_code=500)
    model = ResilientModel(inner=inner)
    with pytest.raises(ModelProviderError):
      await model.ainvoke()


class TestResilientModelStream:
  def test_sync_stream(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    chunks = list(model.invoke_stream())
    assert len(chunks) == 1
    assert chunks[0].content == "chunk"

  def test_sync_stream_failover(self):
    primary = FakeModel(id="primary")
    primary._invoke_side_effect = ModelProviderError("down", status_code=500)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain)
    chunks = list(model.invoke_stream())
    assert len(chunks) == 1

  @pytest.mark.asyncio
  async def test_async_stream(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    chunks = []
    async for chunk in model.ainvoke_stream():
      chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].content == "chunk"

  @pytest.mark.asyncio
  async def test_async_stream_failover(self):
    primary = FakeModel(id="primary")
    primary._ainvoke_side_effect = ModelProviderError("down", status_code=500)
    fallback = FakeModel(id="fallback")
    chain = FailoverChain([FailoverEntry(model=fallback, priority=1)])
    model = ResilientModel(inner=primary, failover=chain)
    chunks = []
    async for chunk in model.ainvoke_stream():
      chunks.append(chunk)
    assert len(chunks) == 1


class TestResilientModelDelegation:
  def test_parse_provider_response_delegates(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    result = model._parse_provider_response({})
    assert result.content == "parsed"

  def test_parse_provider_response_delta_delegates(self):
    inner = FakeModel()
    model = ResilientModel(inner=inner)
    result = model._parse_provider_response_delta({})
    assert result.content == "delta"

  def test_getattr_proxies_to_inner(self):
    inner = FakeModel()
    inner.api_key = "sk-test"
    model = ResilientModel(inner=inner)
    assert model.api_key == "sk-test"


class TestFailoverChain:
  def test_requires_entries(self):
    with pytest.raises(ValueError, match="at least one entry"):
      FailoverChain([])

  def test_sorted_by_priority(self):
    e1 = FailoverEntry(model=FakeModel(id="low"), priority=2)
    e2 = FailoverEntry(model=FakeModel(id="high"), priority=0)
    chain = FailoverChain([e1, e2])
    assert chain.primary.model.id == "high"
    assert chain.entries[1].model.id == "low"

  def test_len_and_iter(self):
    entries = [FailoverEntry(model=FakeModel(id=f"m{i}"), priority=i) for i in range(3)]
    chain = FailoverChain(entries)
    assert len(chain) == 3
    ids = [e.model.id for e in chain]
    assert ids == ["m0", "m1", "m2"]
