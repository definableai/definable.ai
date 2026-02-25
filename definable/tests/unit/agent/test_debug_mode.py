"""
Unit tests for Agent debug mode.

Tests the new ModelCall events, DebugExporter, and debug=True wiring.
"""

import asyncio

import pytest

from definable.agent.agent import Agent
from definable.agent.run.agent import (
  ModelCallCompletedEvent,
  ModelCallStartedEvent,
  RunCompletedEvent,
  RunStartedEvent,
)
from definable.agent.testing import MockModel
from definable.agent.tracing.base import NoOpExporter, Tracing
from definable.agent.tracing.debug import DebugExporter
from definable.model.message import Message


@pytest.mark.unit
class TestModelCallEventFields:
  """Test ModelCallStartedEvent and ModelCallCompletedEvent dataclass fields."""

  def test_model_call_started_defaults(self):
    event = ModelCallStartedEvent()
    assert event.turn == 0
    assert event.messages is None
    assert event.tool_definitions is None
    assert event.response_format is None
    assert event.model_id == ""
    assert event.model_provider == ""
    assert event.event == "ModelCallStarted"

  def test_model_call_completed_defaults(self):
    event = ModelCallCompletedEvent()
    assert event.turn == 0
    assert event.content is None
    assert event.tool_calls is None
    assert event.metrics is None
    assert event.model_id == ""
    assert event.event == "ModelCallCompleted"

  def test_model_call_started_with_values(self):
    msgs = [Message(role="user", content="hi")]
    event = ModelCallStartedEvent(
      turn=1,
      messages=msgs,
      model_id="gpt-4o-mini",
      model_provider="openai",
    )
    assert event.turn == 1
    assert event.messages is msgs
    assert event.model_id == "gpt-4o-mini"
    assert event.model_provider == "openai"

  def test_model_call_completed_with_values(self):
    event = ModelCallCompletedEvent(
      turn=2,
      content="Hello!",
      tool_calls=[{"function": {"name": "add", "arguments": "{}"}}],
      model_id="gpt-4o",
    )
    assert event.turn == 2
    assert event.content == "Hello!"
    assert event.tool_calls is not None
    assert len(event.tool_calls) == 1
    assert event.model_id == "gpt-4o"

  def test_model_call_started_to_dict(self):
    event = ModelCallStartedEvent(turn=1, model_id="gpt-4o")
    d = event.to_dict()
    assert d["turn"] == 1
    assert d["model_id"] == "gpt-4o"
    assert d["event"] == "ModelCallStarted"

  def test_model_call_completed_to_dict(self):
    event = ModelCallCompletedEvent(turn=1, content="hi", model_id="gpt-4o")
    d = event.to_dict()
    assert d["turn"] == 1
    assert d["content"] == "hi"
    assert d["event"] == "ModelCallCompleted"


@pytest.mark.unit
class TestDebugExporter:
  """Test DebugExporter handles all event types without crashing."""

  def test_handles_run_started(self):
    exporter = DebugExporter()
    event = RunStartedEvent(model="gpt-4o", model_provider="openai", run_id="r1")
    exporter.export(event)  # Should not raise

  def test_handles_model_call_started(self):
    exporter = DebugExporter()
    event = ModelCallStartedEvent(
      turn=1,
      messages=[Message(role="user", content="hello")],
      model_id="gpt-4o",
      model_provider="openai",
    )
    exporter.export(event)

  def test_handles_model_call_completed(self):
    exporter = DebugExporter()
    event = ModelCallCompletedEvent(
      turn=1,
      content="Hello! How can I help?",
      model_id="gpt-4o",
    )
    exporter.export(event)

  def test_handles_run_completed(self):
    exporter = DebugExporter()
    event = RunCompletedEvent(content="done")
    exporter.export(event)

  def test_handles_unknown_event_gracefully(self):
    """DebugExporter silently ignores unrecognized event types."""
    from definable.agent.run.agent import CustomEvent

    exporter = DebugExporter()
    event = CustomEvent(event="CustomEvent")
    exporter.export(event)  # Should not raise

  def test_flush_and_shutdown(self):
    exporter = DebugExporter()
    exporter.flush()  # No console initialized — should not raise
    exporter.shutdown()

  def test_truncation(self):
    from definable.agent.tracing.debug import _truncate

    assert _truncate("short", 100) == "short"
    assert _truncate("a" * 200, 50) == "a" * 50 + "..."
    assert _truncate("exact", 5) == "exact"


@pytest.mark.unit
class TestAgentDebugWiring:
  """Test that debug=True properly wires DebugExporter into tracing."""

  def test_debug_creates_tracing(self):
    agent = Agent(model=MockModel(), debug=True)  # type: ignore[arg-type]
    assert agent._trace_writer is not None
    assert agent._trace_writer.exporter_count >= 1
    # At least one exporter should be DebugExporter
    exporters = agent._trace_writer._exporters
    assert any(isinstance(e, DebugExporter) for e in exporters)

  def test_debug_merges_with_existing_tracing(self):
    noop = NoOpExporter()
    agent = Agent(
      model=MockModel(),  # type: ignore[arg-type]
      debug=True,
      tracing=Tracing(exporters=[noop]),
    )
    assert agent._trace_writer is not None
    exporters = agent._trace_writer._exporters
    assert len(exporters) == 2
    assert any(isinstance(e, NoOpExporter) for e in exporters)
    assert any(isinstance(e, DebugExporter) for e in exporters)

  def test_debug_false_no_overhead(self):
    agent = Agent(model=MockModel(), debug=False)  # type: ignore[arg-type]
    # Without explicit tracing, no trace writer should exist
    assert agent._trace_writer is None


@pytest.mark.unit
class TestModelCallEventsInRun:
  """Test that ModelCallStarted/Completed events are emitted during agent runs."""

  def test_model_call_events_emitted(self):
    """Run agent with MockModel, verify ModelCall events are received via EventBus."""
    model = MockModel(responses=["Hello!"])
    agent = Agent(model=model, debug=True)  # type: ignore[arg-type]

    received_events = []

    @agent.events.on(ModelCallStartedEvent)
    def on_started(event):
      received_events.append(("started", event))

    @agent.events.on(ModelCallCompletedEvent)
    def on_completed(event):
      received_events.append(("completed", event))

    output = asyncio.run(agent.arun("Hi"))
    assert output.content == "Hello!"

    # Should have exactly 1 started + 1 completed
    started = [e for tag, e in received_events if tag == "started"]
    completed = [e for tag, e in received_events if tag == "completed"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].turn == 1
    assert completed[0].turn == 1
    assert completed[0].content == "Hello!"

  def test_model_call_events_multi_turn(self):
    """Agent with tools should emit events for each turn."""
    from unittest.mock import MagicMock

    from definable.model.metrics import Metrics
    from definable.tool.decorator import tool

    @tool
    def add(a: int, b: int) -> int:
      """Add two numbers."""
      return a + b

    # First response has tool call, second is final answer
    call_count = [0]

    def side_effect(messages, tools, **kwargs):
      resp = MagicMock()
      resp.response_usage = Metrics()
      resp.reasoning_content = None
      resp.citations = None
      resp.images = None
      resp.videos = None
      resp.audios = None
      if call_count[0] == 0:
        resp.content = ""
        resp.tool_calls = [{"id": "tc_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'}}]
      else:
        resp.content = "The answer is 42."
        resp.tool_calls = []
      call_count[0] += 1
      return resp

    model = MockModel(side_effect=side_effect)
    agent = Agent(model=model, tools=[add], debug=True)  # type: ignore[arg-type]

    turns = []

    @agent.events.on(ModelCallStartedEvent)
    def on_started(event):
      turns.append(("started", event.turn))

    @agent.events.on(ModelCallCompletedEvent)
    def on_completed(event):
      turns.append(("completed", event.turn))

    asyncio.run(agent.arun("What is 17 + 25?"))

    # Should have 2 turns (turn 1: tool call, turn 2: final answer)
    started_turns = [t for tag, t in turns if tag == "started"]
    completed_turns = [t for tag, t in turns if tag == "completed"]
    assert started_turns == [1, 2]
    assert completed_turns == [1, 2]

  def test_model_call_started_has_messages(self):
    """ModelCallStartedEvent should contain the message snapshot."""
    model = MockModel(responses=["hi"])
    agent = Agent(model=model, debug=True)  # type: ignore[arg-type]

    captured = []

    @agent.events.on(ModelCallStartedEvent)
    def on_started(event):
      captured.append(event)

    asyncio.run(agent.arun("Hello"))

    assert len(captured) == 1
    evt = captured[0]
    assert evt.messages is not None
    # Should contain at least the user message
    user_msgs = [m for m in evt.messages if m.role == "user"]
    assert len(user_msgs) >= 1
    assert evt.model_id == "mock-model"


@pytest.mark.unit
class TestNoDoubleEmit:
  """Verify each event is exported exactly once — no double-emit."""

  def test_no_double_emit_simple_run(self):
    """DebugExporter should receive each ModelCall event exactly once."""
    export_log: list[str] = []

    class CountingExporter:
      """Exporter that records every export() call."""

      def export(self, event):
        export_log.append(event.event)

      def flush(self):
        pass

      def shutdown(self):
        pass

    model = MockModel(responses=["Hello!"])
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tracing=Tracing(exporters=[CountingExporter()]),
    )

    asyncio.run(agent.arun("Hi"))

    started_count = export_log.count("ModelCallStarted")
    completed_count = export_log.count("ModelCallCompleted")
    run_completed_count = export_log.count("RunCompleted")
    assert started_count == 1, f"ModelCallStarted emitted {started_count} times, expected 1"
    assert completed_count == 1, f"ModelCallCompleted emitted {completed_count} times, expected 1"
    assert run_completed_count == 1, f"RunCompleted emitted {run_completed_count} times, expected 1"

  def test_no_double_emit_with_tools(self):
    """Tool call events should also be emitted exactly once per tool call."""
    from unittest.mock import MagicMock

    from definable.model.metrics import Metrics
    from definable.tool.decorator import tool

    @tool
    def greet(name: str) -> str:
      """Greet someone."""
      return f"Hello, {name}!"

    export_log: list[str] = []

    class CountingExporter:
      def export(self, event):
        export_log.append(event.event)

      def flush(self):
        pass

      def shutdown(self):
        pass

    call_count = [0]

    def side_effect(messages, tools, **kwargs):
      resp = MagicMock()
      resp.response_usage = Metrics()
      resp.reasoning_content = None
      resp.citations = None
      resp.images = None
      resp.videos = None
      resp.audios = None
      if call_count[0] == 0:
        resp.content = ""
        resp.tool_calls = [{"id": "tc_1", "type": "function", "function": {"name": "greet", "arguments": '{"name": "World"}'}}]
      else:
        resp.content = "Done greeting."
        resp.tool_calls = []
      call_count[0] += 1
      return resp

    model = MockModel(side_effect=side_effect)
    agent = Agent(
      model=model,  # type: ignore[arg-type]
      tools=[greet],
      tracing=Tracing(exporters=[CountingExporter()]),
    )

    asyncio.run(agent.arun("Greet World"))

    # 2 model turns → 2 started + 2 completed
    assert export_log.count("ModelCallStarted") == 2, f"got {export_log.count('ModelCallStarted')}"
    assert export_log.count("ModelCallCompleted") == 2, f"got {export_log.count('ModelCallCompleted')}"
    # 1 tool call → 1 started + 1 completed
    assert export_log.count("ToolCallStarted") == 1, f"got {export_log.count('ToolCallStarted')}"
    assert export_log.count("ToolCallCompleted") == 1, f"got {export_log.count('ToolCallCompleted')}"
    assert export_log.count("RunCompleted") == 1
