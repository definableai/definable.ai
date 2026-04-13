"""
Behavioral tests: Cooperative cancellation via CancellationToken.

Migrated from tests_e2e/behavioral/test_cancellation.py.

Strategy:
  - MockModel side_effect simulates multi-step tool-call cycles
  - CancellationToken is cancelled before or between loop iterations
  - Assert on OUTCOMES: RunStatus.cancelled is returned

Covers:
  - Cancel before run starts — returns cancelled status
  - Cancel between loop iterations — returns cancelled status
  - Cancelled run output has status == cancelled
  - Cancelled token stays cancelled — reuse returns cancelled immediately
"""

import pytest
from unittest.mock import MagicMock

from definable.agent import Agent
from definable.agent.cancellation import CancellationToken
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.agent.events import RunStatus
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def get_data() -> str:
  """Get some data."""
  return "data_result"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_multi_step_side_effect(token: CancellationToken, cancel_after_call: int = 1):
  """Return a side_effect that cancels the token after N calls.

  The loop checks cancellation at the top of each iteration, so cancelling
  during a model call means the NEXT iteration will see the cancellation.
  """
  call_count = 0

  def side_effect(messages, tools, **kwargs):
    nonlocal call_count
    call_count += 1
    response = MagicMock()
    response.response_usage = Metrics()
    response.reasoning_content = None
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None
    if call_count <= cancel_after_call:
      # Issue a tool call to keep the loop going
      response.content = ""
      response.tool_calls = [
        {"id": f"call_{call_count}", "type": "function", "function": {"name": "get_data", "arguments": "{}"}},
      ]
      # Cancel after this call completes — the next iteration will see it
      if call_count == cancel_after_call:
        token.cancel()
    else:
      response.content = "Final answer (should not reach here)"
      response.tool_calls = []
    return response

  return side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestCancellation:
  """Cooperative cancellation via CancellationToken."""

  async def test_cancel_before_run(self):
    """Cancelling before arun() returns output with cancelled status."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    token = CancellationToken()
    token.cancel()

    result = await agent.arun("Say hello", cancellation_token=token)
    assert result.status == RunStatus.cancelled

  async def test_cancel_between_iterations(self):
    """Cancelling between loop iterations returns cancelled status."""
    token = CancellationToken()
    model = MockModel(
      side_effect=_make_multi_step_side_effect(token, cancel_after_call=1),
    )
    agent = Agent(model=model, tools=[get_data], config=NO_TRACE)  # type: ignore[arg-type]

    result = await agent.arun("Get some data", cancellation_token=token)
    assert result.status == RunStatus.cancelled

  async def test_cancelled_output_has_no_content(self):
    """A cancelled run should have no final content."""
    token = CancellationToken()
    token.cancel()

    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    result = await agent.arun("Say hello", cancellation_token=token)
    assert result.status == RunStatus.cancelled
    # Cancelled runs typically have no content
    assert result.content is None

  async def test_cancellation_token_reuse(self):
    """A cancelled token stays cancelled — a new run with the same token returns cancelled."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True

    # First run
    result1 = await agent.arun("First", cancellation_token=token)
    assert result1.status == RunStatus.cancelled

    # Second run with the same token
    result2 = await agent.arun("Second", cancellation_token=token)
    assert result2.status == RunStatus.cancelled
