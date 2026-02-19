"""
Behavioral tests: HITL (Human-in-the-Loop) pause/resume via requires_confirmation.

Migrated from: tests_e2e/behavioral/test_hitl_pause_resume.py

Strategy:
  - MockModel side_effect drives the tool call / final answer cycle
  - Assert on OUTCOMES: run pauses, requirements are present, confirm/reject works,
    continue_run resumes successfully
  - Tools use @tool(requires_confirmation=True) and @tool(requires_user_input=True)

Covers:
  - Tool with requires_confirmation pauses the run
  - Confirming a requirement and calling continue_run completes the run
  - Rejecting a requirement and calling continue_run sends rejection to model
  - continue_run raises ValueError when run is not paused
  - continue_run raises ValueError when requirements are unresolved
  - Tool with requires_user_input pauses the run
"""

import pytest
from unittest.mock import MagicMock

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.model.metrics import Metrics
from definable.agent.events import RunStatus
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool(requires_confirmation=True)
def dangerous_action() -> str:
  """Delete everything from the database."""
  return "deleted"


@tool(requires_user_input=True)
def collect_feedback(rating: int, comment: str) -> str:
  """Collect user feedback with a rating and comment."""
  return f"Feedback received: {rating}/5 - {comment}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


def _make_tool_call_side_effect(tool_name: str, tool_args: str = "{}"):
  """Return a side_effect that issues a tool call on call 1, then answers on call 2+."""
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
    if call_count == 1:
      response.content = ""
      response.tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": tool_args}},
      ]
    else:
      response.content = "Done — action completed."
      response.tool_calls = []
    return response

  return side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
class TestHITLPauseResume:
  """HITL pause/resume via requires_confirmation and requires_user_input."""

  async def test_requires_confirmation_pauses_run(self):
    """Tool with requires_confirmation=True pauses the run before execution."""
    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_action"))
    agent = Agent(model=model, tools=[dangerous_action], config=NO_TRACE)  # type: ignore[arg-type]

    output = await agent.arun("Delete the database")

    assert output.status == RunStatus.paused
    assert output.is_paused is True
    assert output.requirements is not None
    assert len(output.requirements) > 0
    # The requirement should need confirmation
    req = output.requirements[0]
    assert req.needs_confirmation is True

  async def test_confirm_and_continue(self):
    """Confirming a requirement and calling continue_run completes the run."""
    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_action"))
    agent = Agent(model=model, tools=[dangerous_action], config=NO_TRACE)  # type: ignore[arg-type]

    paused_output = await agent.arun("Delete the database")
    assert paused_output.is_paused

    # Confirm all requirements
    for req in paused_output.requirements:  # type: ignore[union-attr]
      req.confirm()

    # Resume
    completed_output = await agent.continue_run(run_output=paused_output)
    assert completed_output.status == RunStatus.completed
    assert completed_output.content is not None

  async def test_reject_and_continue(self):
    """Rejecting a requirement and calling continue_run sends rejection to model."""
    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_action"))
    agent = Agent(model=model, tools=[dangerous_action], config=NO_TRACE)  # type: ignore[arg-type]

    paused_output = await agent.arun("Delete the database")
    assert paused_output.is_paused

    # Reject
    for req in paused_output.requirements:  # type: ignore[union-attr]
      req.reject()

    completed_output = await agent.continue_run(run_output=paused_output)
    # The model should receive a rejection message and still complete
    assert completed_output.status == RunStatus.completed

  async def test_continue_run_raises_if_not_paused(self):
    """continue_run raises ValueError when the run is not paused."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, config=NO_TRACE)  # type: ignore[arg-type]

    output = await agent.arun("Say hi")
    assert output.status == RunStatus.completed

    with pytest.raises(ValueError, match="not paused"):
      await agent.continue_run(run_output=output)

  async def test_continue_run_raises_if_unresolved(self):
    """continue_run raises ValueError when requirements are still unresolved."""
    model = MockModel(side_effect=_make_tool_call_side_effect("dangerous_action"))
    agent = Agent(model=model, tools=[dangerous_action], config=NO_TRACE)  # type: ignore[arg-type]

    paused_output = await agent.arun("Delete the database")
    assert paused_output.is_paused
    assert len(paused_output.active_requirements) > 0

    # Do NOT resolve requirements — should raise
    with pytest.raises(ValueError, match="unresolved"):
      await agent.continue_run(run_output=paused_output)

  async def test_requires_user_input_pauses_run(self):
    """Tool with requires_user_input=True pauses the run."""
    model = MockModel(
      side_effect=_make_tool_call_side_effect(
        "collect_feedback",
        '{"rating": 5, "comment": "Great!"}',
      )
    )
    agent = Agent(model=model, tools=[collect_feedback], config=NO_TRACE)  # type: ignore[arg-type]

    output = await agent.arun("Please collect my feedback")

    assert output.status == RunStatus.paused
    assert output.is_paused is True
    assert output.requirements is not None
    assert len(output.requirements) > 0
