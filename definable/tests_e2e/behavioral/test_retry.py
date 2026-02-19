"""
Behavioral tests: Model call retry with transient errors.

Strategy:
  - MockModel side_effect raises transient errors (ConnectionError) and
    non-transient errors (ValueError) to test retry logic
  - AgentConfig controls retry_transient_errors, max_retries, retry_backoff_base
  - Assert on OUTCOMES: agent succeeds after transient failure, fails fast on
    non-transient, and raises after exhausting retries

Covers:
  - Retry on ConnectionError succeeds when second call works
  - No retry on ValueError — raises immediately
  - Exhausted retries raises the last transient error
"""

import pytest
from unittest.mock import MagicMock

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.agent.events import RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Tests
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
