"""
Behavioral tests: Middleware, retry logic, and EventBus integration.

Migrated from:
  - tests_e2e/behavioral/test_retry.py (retry transient errors)
  - tests_e2e/behavioral/test_event_callbacks.py (EventBus integration)

Strategy:
  - MockModel side_effect raises transient errors (ConnectionError) and
    non-transient errors (ValueError) to test retry logic
  - AgentConfig controls retry_transient_errors, max_retries, retry_backoff_base
  - EventBus handlers registered on agent.events
  - Assert on OUTCOMES: agent succeeds after transient failure, fails fast on
    non-transient, raises after exhausting retries, event handlers fire correctly

Covers:
  - Retry on ConnectionError succeeds when second call works
  - No retry on ValueError — raises immediately
  - Exhausted retries raises the last transient error
  - EventBus fires RunCompletedEvent on successful run
  - Handler that raises does not break the run
  - Multiple event types fire their registered handlers
"""

import pytest
from unittest.mock import MagicMock

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.agent.events import RunCompletedEvent, RunStartedEvent, RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_retry_config(**overrides):
  """Build an AgentConfig with fast retry settings for testing."""
  defaults = {
    "tracing": Tracing(enabled=False),
    "retry_transient_errors": True,
    "max_retries": 2,
    "retry_backoff_base": 0.01,  # Fast for tests
  }
  defaults.update(overrides)
  return AgentConfig(**defaults)  # type: ignore[arg-type]


def _make_success_response(content: str = "Success"):
  """Create a standard successful mock response."""
  response = MagicMock()
  response.content = content
  response.tool_calls = []
  response.response_usage = Metrics()
  response.reasoning_content = None
  response.citations = None
  response.images = None
  response.videos = None
  response.audios = None
  return response


# ---------------------------------------------------------------------------
# Tests: Retry transient errors
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestRetryTransientErrors:
  """Model call retry with transient errors in the agentic loop."""

  async def test_retry_on_connection_error(self):
    """Side effect raises ConnectionError on first call, succeeds on second."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ConnectionError("Connection refused")
      return _make_success_response("Recovered successfully")

    model = MockModel(side_effect=side_effect)
    config = _make_retry_config()
    agent = Agent(model=model, config=config)  # type: ignore[arg-type]

    output = await agent.arun("Do something")
    assert output.status == RunStatus.completed
    assert output.content is not None
    assert call_count == 2  # First call failed, second succeeded

  async def test_no_retry_on_non_transient(self):
    """Side effect raises ValueError — agent should raise immediately, not retry."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      raise ValueError("Invalid input — not transient")

    model = MockModel(side_effect=side_effect)
    config = _make_retry_config()
    agent = Agent(model=model, config=config)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid input"):
      await agent.arun("Do something")

    # Should have been called exactly once — no retries for non-transient errors
    assert call_count == 1

  async def test_exhausted_retries_raises(self):
    """Side effect always raises ConnectionError — after max_retries, agent raises."""
    call_count = 0

    def side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      raise ConnectionError("Persistent connection failure")

    model = MockModel(side_effect=side_effect)
    config = _make_retry_config(max_retries=2)
    agent = Agent(model=model, config=config)  # type: ignore[arg-type]

    with pytest.raises(ConnectionError, match="Persistent connection failure"):
      await agent.arun("Do something")

    # 1 initial + 2 retries = 3 total calls
    assert call_count == 3


# ---------------------------------------------------------------------------
# Tests: EventBus callbacks
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestEventCallbacks:
  """EventBus integration with Agent."""

  async def test_event_bus_fires_on_run_completed(self):
    """Register handler on RunCompletedEvent. Handler should be called after arun()."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    received_events = []

    def handler(event):
      received_events.append(event)

    agent.events.on(RunCompletedEvent, handler)

    output = await agent.arun("Say hello")
    assert output.status == RunStatus.completed
    assert len(received_events) == 1
    assert isinstance(received_events[0], RunCompletedEvent)

  async def test_event_bus_handler_error_does_not_break_run(self):
    """A handler that raises an exception should not break the agent run."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    def bad_handler(event):
      raise RuntimeError("Handler exploded!")

    agent.events.on(RunCompletedEvent, bad_handler)

    # The run should still complete despite the handler error
    output = await agent.arun("Say hello")
    assert output.status == RunStatus.completed
    assert output.content == "Hello!"

  async def test_multiple_event_types_fire(self):
    """Register handlers for multiple event types. Verify each fires."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    started_events = []
    completed_events = []

    def on_started(event):
      started_events.append(event)

    def on_completed(event):
      completed_events.append(event)

    agent.events.on(RunStartedEvent, on_started)
    agent.events.on(RunCompletedEvent, on_completed)

    output = await agent.arun("Say hello")
    assert output.status == RunStatus.completed

    # Both event types should have fired
    assert len(started_events) >= 1, f"Expected RunStartedEvent, got {len(started_events)}"
    assert len(completed_events) >= 1, f"Expected RunCompletedEvent, got {len(completed_events)}"
    assert isinstance(started_events[0], RunStartedEvent)
    assert isinstance(completed_events[0], RunCompletedEvent)
