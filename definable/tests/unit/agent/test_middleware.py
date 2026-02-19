"""
Unit tests for middleware protocol and built-in middleware classes.

Tests instantiation, protocol compliance, and basic behavior of:
  - LoggingMiddleware
  - RetryMiddleware
  - MetricsMiddleware
  - KnowledgeMiddleware
  - StreamingMiddleware protocol
  - Middleware __call__ protocol

No API calls.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.middleware import (
  KnowledgeMiddleware,
  LoggingMiddleware,
  MetricsMiddleware,
  Middleware,
  RetryMiddleware,
  StreamingMiddleware,
)


@pytest.mark.unit
class TestLoggingMiddleware:
  def test_can_be_instantiated(self):
    """LoggingMiddleware requires a logger."""
    logger = logging.getLogger("test")
    mw = LoggingMiddleware(logger)
    assert mw.logger is logger
    assert mw.level == logging.INFO

  def test_custom_log_level(self):
    """LoggingMiddleware accepts a custom log level."""
    logger = logging.getLogger("test")
    mw = LoggingMiddleware(logger, level=logging.DEBUG)
    assert mw.level == logging.DEBUG

  @pytest.mark.asyncio
  async def test_call_delegates_to_next_handler(self):
    """LoggingMiddleware calls next_handler and returns its result."""
    logger = logging.getLogger("test")
    mw = LoggingMiddleware(logger)

    mock_output = MagicMock()
    mock_output.status = "completed"
    next_handler = AsyncMock(return_value=mock_output)

    context = MagicMock()
    context.run_id = "test-run"
    context.session_id = "test-session"

    result = await mw(context, next_handler)
    assert result is mock_output
    next_handler.assert_awaited_once_with(context)

  @pytest.mark.asyncio
  async def test_call_reraises_on_error(self):
    """LoggingMiddleware re-raises exceptions from next_handler."""
    logger = logging.getLogger("test")
    mw = LoggingMiddleware(logger)

    next_handler = AsyncMock(side_effect=RuntimeError("boom"))
    context = MagicMock()
    context.run_id = "test-run"
    context.session_id = "test-session"

    with pytest.raises(RuntimeError, match="boom"):
      await mw(context, next_handler)


@pytest.mark.unit
class TestRetryMiddleware:
  def test_can_be_instantiated_with_defaults(self):
    """RetryMiddleware has sensible defaults."""
    mw = RetryMiddleware()
    assert mw.max_retries == 3
    assert mw.backoff_base == 1.0
    assert mw.backoff_max == 60.0

  def test_can_be_instantiated_with_max_retries(self):
    """RetryMiddleware accepts custom max_retries."""
    mw = RetryMiddleware(max_retries=5)
    assert mw.max_retries == 5

  def test_custom_backoff(self):
    """RetryMiddleware accepts custom backoff parameters."""
    mw = RetryMiddleware(backoff_base=0.5, backoff_max=30.0)
    assert mw.backoff_base == 0.5
    assert mw.backoff_max == 30.0

  @pytest.mark.asyncio
  async def test_returns_result_on_first_success(self):
    """RetryMiddleware returns immediately on success."""
    mw = RetryMiddleware(max_retries=3)

    mock_output = MagicMock()
    next_handler = AsyncMock(return_value=mock_output)
    context = MagicMock()

    result = await mw(context, next_handler)
    assert result is mock_output
    assert next_handler.await_count == 1

  @pytest.mark.asyncio
  async def test_non_transient_error_not_retried(self):
    """Non-transient errors are re-raised immediately without retry."""
    mw = RetryMiddleware(max_retries=3)

    next_handler = AsyncMock(side_effect=ValueError("bad input"))
    context = MagicMock()

    with pytest.raises(ValueError, match="bad input"):
      await mw(context, next_handler)
    assert next_handler.await_count == 1


@pytest.mark.unit
class TestMetricsMiddleware:
  def test_can_be_instantiated(self):
    """MetricsMiddleware has zero initial state."""
    mw = MetricsMiddleware()
    assert mw.run_count == 0
    assert mw.error_count == 0
    assert mw.average_latency_ms == 0.0

  @pytest.mark.asyncio
  async def test_increments_run_count(self):
    """Each call increments run_count."""
    mw = MetricsMiddleware()

    mock_output = MagicMock()
    next_handler = AsyncMock(return_value=mock_output)
    context = MagicMock()

    await mw(context, next_handler)
    assert mw.run_count == 1

    await mw(context, next_handler)
    assert mw.run_count == 2

  @pytest.mark.asyncio
  async def test_increments_error_count_on_failure(self):
    """Failed calls increment error_count."""
    mw = MetricsMiddleware()

    next_handler = AsyncMock(side_effect=RuntimeError("fail"))
    context = MagicMock()

    with pytest.raises(RuntimeError):
      await mw(context, next_handler)

    assert mw.error_count == 1
    assert mw.run_count == 1

  def test_reset_clears_metrics(self):
    """reset() zeroes all counters."""
    mw = MetricsMiddleware()
    mw._run_count = 5
    mw._error_count = 2
    mw._total_latency_ms = 1000.0

    mw.reset()
    assert mw.run_count == 0
    assert mw.error_count == 0
    assert mw.average_latency_ms == 0.0


@pytest.mark.unit
class TestKnowledgeMiddleware:
  def test_can_be_instantiated(self):
    """KnowledgeMiddleware requires a Knowledge instance."""
    knowledge = MagicMock()
    mw = KnowledgeMiddleware(knowledge)
    assert mw.knowledge is knowledge

  @pytest.mark.asyncio
  async def test_disabled_knowledge_skips_retrieval(self):
    """When knowledge.enabled is False, middleware passes through."""
    knowledge = MagicMock()
    knowledge.enabled = False
    mw = KnowledgeMiddleware(knowledge)

    mock_output = MagicMock()
    next_handler = AsyncMock(return_value=mock_output)
    context = MagicMock()

    result = await mw(context, next_handler)
    assert result is mock_output
    knowledge.asearch.assert_not_called()


@pytest.mark.unit
class TestStreamingMiddlewareProtocol:
  def test_protocol_exists(self):
    """StreamingMiddleware is a runtime-checkable Protocol."""
    assert hasattr(StreamingMiddleware, "__call__")

  def test_class_satisfying_protocol_is_recognized(self):
    """A class with the right __call__ signature satisfies StreamingMiddleware."""

    class MyStreamingMW:
      async def __call__(self, context, event_stream):
        async for event in event_stream:
          yield event

    mw = MyStreamingMW()
    assert isinstance(mw, StreamingMiddleware)


@pytest.mark.unit
class TestMiddlewareProtocol:
  def test_protocol_exists(self):
    """Middleware is a runtime-checkable Protocol."""
    assert hasattr(Middleware, "__call__")

  def test_class_satisfying_protocol_is_recognized(self):
    """A class with async __call__(context, next_handler) satisfies Middleware."""

    class MyMW:
      async def __call__(self, context, next_handler):
        return await next_handler(context)

    mw = MyMW()
    assert isinstance(mw, Middleware)

  def test_logging_middleware_satisfies_protocol(self):
    """LoggingMiddleware satisfies the Middleware protocol."""
    logger = logging.getLogger("test")
    mw = LoggingMiddleware(logger)
    assert isinstance(mw, Middleware)

  def test_retry_middleware_satisfies_protocol(self):
    """RetryMiddleware satisfies the Middleware protocol."""
    mw = RetryMiddleware()
    assert isinstance(mw, Middleware)

  def test_metrics_middleware_satisfies_protocol(self):
    """MetricsMiddleware satisfies the Middleware protocol."""
    mw = MetricsMiddleware()
    assert isinstance(mw, Middleware)
