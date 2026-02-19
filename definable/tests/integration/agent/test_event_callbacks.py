"""
Behavioral tests: EventBus integration with Agent.

Migrated from: tests_e2e/behavioral/test_event_callbacks.py

Strategy:
  - Register handlers on the agent's EventBus (agent.events)
  - Run the agent with MockModel
  - Assert on OUTCOMES: handlers were called, errors don't break the run,
    multiple event types all fire

Covers:
  - EventBus fires RunCompletedEvent on successful run
  - Handler that raises does not break the run
  - Multiple event types fire their registered handlers
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.agent.events import RunCompletedEvent, RunStartedEvent, RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NO_TRACE = AgentConfig(tracing=Tracing(enabled=False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
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
