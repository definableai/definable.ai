"""E2E tests for the agent thinking layer."""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from definable.agents import Agent, AgentConfig, ThinkingConfig, TracingConfig
from definable.agents.testing import MockModel
from definable.models.metrics import Metrics
from definable.run.agent import ReasoningContentDeltaEvent, ReasoningStepEvent

# --- Fixtures ---

MOCK_THINKING_JSON = json.dumps({
  "analysis": "The user wants to know about microservices vs monoliths.",
  "approach": "I will outline the key tradeoffs covering scalability, complexity, and team structure.",
  "tool_plan": None,
})

MOCK_THINKING_WITH_TOOLS_JSON = json.dumps({
  "analysis": "The user wants current data about a topic.",
  "approach": "Search for information then synthesize results.",
  "tool_plan": ["web_search", "summarize"],
})


def _make_thinking_model(structured_json: str = MOCK_THINKING_JSON):
  """Create a MockModel configured for thinking (structured output)."""
  return MockModel(
    responses=["Final answer after thinking."],
    structured_responses=[structured_json],
  )


def _make_side_effect_model():
  """Create a model that tracks structured vs normal calls via side_effect."""
  calls: List[Dict[str, Any]] = []

  def side_effect(messages, tools, **kwargs):
    response = MagicMock()
    response.tool_executions = []
    response.tool_calls = []
    response.response_usage = Metrics()
    response.reasoning_content = None
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None

    response_format = kwargs.get("response_format")
    if response_format is not None:
      # This is the thinking call — return structured JSON
      response.content = MOCK_THINKING_JSON
      calls.append({"type": "thinking", "messages": messages})
    else:
      # This is the main call
      response.content = "Final answer after thinking."
      calls.append({"type": "main", "messages": messages})

    return response

  model = MockModel(side_effect=side_effect)
  model._tracked_calls = calls  # type: ignore[attr-defined]
  return model


# --- Tests ---


@pytest.mark.e2e
class TestAgentThinking:
  """Tests for the agent thinking layer."""

  @pytest.mark.asyncio
  async def test_thinking_disabled_by_default(self):
    """Agent without thinking=True should not make extra model calls."""
    model = MockModel(responses=["Test response"])
    agent = Agent(
      model=model,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Hello")
    assert output.content == "Test response"
    assert output.reasoning_steps is None
    assert model.call_count == 1

  @pytest.mark.asyncio
  async def test_thinking_enabled_makes_two_calls(self):
    """Agent with thinking=True makes two model calls (think + answer)."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("What are the pros and cons of microservices?")
    # Two calls: one for thinking (structured), one for main answer
    assert model.call_count == 2
    assert output.content == "Final answer after thinking."

  @pytest.mark.asyncio
  async def test_thinking_reasoning_steps_in_output(self):
    """RunOutput.reasoning_steps and reasoning_content are populated."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Explain microservices")

    assert output.reasoning_steps is not None
    # ThinkingOutput without tool_plan maps to 1 ReasoningStep
    assert len(output.reasoning_steps) == 1
    assert output.reasoning_steps[0].title == "Analysis"
    assert "microservices" in output.reasoning_steps[0].reasoning.lower()

    # reasoning_content should contain XML-formatted reasoning (legacy format for observability)
    assert output.reasoning_content is not None
    assert "<reasoning>" in output.reasoning_content
    assert "<step" in output.reasoning_content

  @pytest.mark.asyncio
  async def test_thinking_reasoning_messages_in_output(self):
    """RunOutput.reasoning_messages contains the thinking conversation."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Hello")

    assert output.reasoning_messages is not None
    assert len(output.reasoning_messages) >= 2  # system + assistant at minimum
    # First message should be the system thinking prompt
    assert output.reasoning_messages[0].role == "system"
    # Last message should be the assistant response (structured JSON)
    assert output.reasoning_messages[-1].role == "assistant"

  @pytest.mark.asyncio
  async def test_thinking_injects_analysis_into_system_prompt(self):
    """Thinking phase injects <analysis> tag into the system prompt for the main call."""
    model = _make_side_effect_model()
    agent = Agent(
      model=model,
      thinking=True,
      instructions="You are a helpful assistant.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Explain microservices")

    # The main call should have a system message that includes the analysis injection
    tracked_calls = model._tracked_calls  # type: ignore[attr-defined]
    assert len(tracked_calls) == 2
    assert tracked_calls[0]["type"] == "thinking"
    assert tracked_calls[1]["type"] == "main"

    # Check that the system message in the main call includes <analysis>
    main_messages = tracked_calls[1]["messages"]
    system_msg = next((m for m in main_messages if m.role == "system"), None)
    assert system_msg is not None
    assert "<analysis>" in system_msg.content
    assert "tradeoffs" in system_msg.content.lower()

  @pytest.mark.asyncio
  async def test_thinking_with_custom_model(self):
    """ThinkingConfig.model uses a different model for thinking."""
    main_model = MockModel(responses=["Main model answer"])
    thinking_model = MockModel(structured_responses=[MOCK_THINKING_JSON])

    agent = Agent(
      model=main_model,
      thinking=ThinkingConfig(model=thinking_model),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Complex question")

    # Thinking model called once for reasoning
    assert thinking_model.call_count == 1
    # Main model called once for the answer
    assert main_model.call_count == 1
    assert output.content == "Main model answer"
    assert output.reasoning_steps is not None
    assert len(output.reasoning_steps) == 1  # No tool_plan → 1 step

  @pytest.mark.asyncio
  async def test_thinking_with_custom_instructions(self):
    """ThinkingConfig.instructions overrides the default thinking prompt."""
    model = _make_side_effect_model()
    custom_prompt = "Think carefully about the problem before answering."

    agent = Agent(
      model=model,
      thinking=ThinkingConfig(instructions=custom_prompt),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Hello")

    # The thinking call should use the custom prompt
    tracked_calls = model._tracked_calls  # type: ignore[attr-defined]
    thinking_messages = tracked_calls[0]["messages"]
    system_msg = thinking_messages[0]
    assert system_msg.content == custom_prompt

  @pytest.mark.asyncio
  async def test_thinking_emits_reasoning_events(self):
    """Thinking phase emits ReasoningStarted, ReasoningStep, ReasoningCompleted events."""
    from definable.agents.tracing import NoOpExporter

    emitted_events: List[Any] = []

    class EventCapture(NoOpExporter):
      def export(self, event):
        emitted_events.append(event)

    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(
        tracing=TracingConfig(
          enabled=True,
          exporters=[EventCapture()],
        )
      ),
    )
    await agent.arun("Complex question")

    event_types = [type(e).__name__ for e in emitted_events]
    assert "ReasoningStartedEvent" in event_types
    assert "ReasoningStepEvent" in event_types
    assert "ReasoningCompletedEvent" in event_types

    # Should have 1 reasoning step event (no tool_plan → 1 mapped step)
    step_events = [e for e in emitted_events if isinstance(e, ReasoningStepEvent)]
    assert len(step_events) == 1
    assert "microservices" in step_events[0].reasoning_content.lower()

  @pytest.mark.asyncio
  async def test_thinking_with_bool_true(self):
    """thinking=True creates default ThinkingConfig."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._thinking is not None
    assert agent._thinking.enabled is True
    assert agent._thinking.model is None  # Uses agent's model
    assert agent._thinking.instructions is None  # Uses default

  @pytest.mark.asyncio
  async def test_thinking_with_none(self):
    """thinking=None (default) disables thinking."""
    model = MockModel(responses=["Response"])
    agent = Agent(
      model=model,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    assert agent._thinking is None

  @pytest.mark.asyncio
  async def test_thinking_with_config_disabled(self):
    """ThinkingConfig(enabled=False) disables thinking even when provided."""
    model = MockModel(responses=["Response"])
    agent = Agent(
      model=model,
      thinking=ThinkingConfig(enabled=False),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Hello")

    # Should only make one call (no thinking)
    assert model.call_count == 1
    assert output.reasoning_steps is None

  @pytest.mark.asyncio
  async def test_thinking_with_empty_reasoning_response(self):
    """Gracefully handles model returning empty/invalid ThinkingOutput JSON."""
    model = MockModel(
      responses=["Answer despite failed thinking."],
      structured_responses=["{}"],
    )
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Hello")

    # Should still produce output — the fallback ThinkingOutput has approach="Respond directly"
    assert output.content == "Answer despite failed thinking."

  @pytest.mark.asyncio
  async def test_thinking_with_malformed_json(self):
    """Gracefully handles model returning malformed JSON."""
    model = MockModel(
      responses=["Answer despite malformed thinking."],
      structured_responses=["not valid json at all"],
    )
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Hello")

    # Should still produce output — fallback produces a valid ThinkingOutput
    assert output.content == "Answer despite malformed thinking."
    # Fallback generates steps from ThinkingOutput(analysis="Could not parse", approach="Respond directly")
    assert output.reasoning_steps is not None
    assert output.reasoning_steps[0].title == "Analysis"

  @pytest.mark.asyncio
  async def test_thinking_streaming_yields_reasoning_events(self):
    """arun_stream yields full reasoning lifecycle: Started, Deltas, Steps, Completed."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    events = []
    async for event in agent.arun_stream("Complex question"):
      events.append(event)

    event_types = [type(e).__name__ for e in events]

    # All reasoning lifecycle events should be present
    assert "ReasoningStartedEvent" in event_types
    assert "ReasoningContentDeltaEvent" in event_types
    assert "ReasoningStepEvent" in event_types
    assert "ReasoningCompletedEvent" in event_types

    # Verify ordering: Started < Deltas < Steps < Completed < RunContent
    started_idx = event_types.index("ReasoningStartedEvent")
    first_delta_idx = event_types.index("ReasoningContentDeltaEvent")
    step_indices = [i for i, t in enumerate(event_types) if t == "ReasoningStepEvent"]
    completed_idx = event_types.index("ReasoningCompletedEvent")
    content_indices = [i for i, t in enumerate(event_types) if t == "RunContentEvent"]

    assert started_idx < first_delta_idx
    assert len(step_indices) == 1  # ThinkingOutput without tool_plan → 1 mapped step
    for si in step_indices:
      assert first_delta_idx < si  # Steps come after deltas
      assert si < completed_idx  # Steps come before completed
    if content_indices:
      assert completed_idx < content_indices[0]  # Reasoning completes before content

  @pytest.mark.asyncio
  async def test_thinking_streaming_yields_content_deltas(self):
    """ReasoningContentDeltaEvent chunks reconstruct the original structured JSON."""
    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    deltas = []
    async for event in agent.arun_stream("Complex question"):
      if isinstance(event, ReasoningContentDeltaEvent):
        deltas.append(event.reasoning_content)

    # Should have deltas (one per character in the mock)
    assert len(deltas) > 0

    # Concatenated deltas should reconstruct the original JSON
    reconstructed = "".join(deltas)
    assert reconstructed == MOCK_THINKING_JSON

  @pytest.mark.asyncio
  async def test_thinking_preserves_agent_instructions(self):
    """Thinking doesn't lose the agent's original instructions."""
    model = _make_side_effect_model()
    agent = Agent(
      model=model,
      thinking=True,
      instructions="You are a pirate. Always respond in pirate speak.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Hello")

    tracked_calls = model._tracked_calls  # type: ignore[attr-defined]
    main_messages = tracked_calls[1]["messages"]
    system_msg = next((m for m in main_messages if m.role == "system"), None)
    assert system_msg is not None
    # Original instructions should be present
    assert "pirate" in system_msg.content.lower()
    # Compact analysis injection should also be present
    assert "<analysis>" in system_msg.content

  # --- New context-aware thinking tests ---

  @pytest.mark.asyncio
  async def test_thinking_prompt_includes_tool_names(self):
    """When agent has tools, the thinking prompt contains tool names."""
    from definable.tools.decorator import tool

    @tool
    def web_search(query: str) -> str:
      """Search the web for information."""
      return f"Results for {query}"

    calls: List[Dict[str, Any]] = []

    def side_effect(messages, tools, **kwargs):
      response = MagicMock()
      response.tool_executions = []
      response.tool_calls = []
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      response_format = kwargs.get("response_format")
      if response_format is not None:
        response.content = MOCK_THINKING_JSON
        calls.append({"type": "thinking", "messages": messages})
      else:
        response.content = "Final answer."
        calls.append({"type": "main", "messages": messages})

      return response

    model = MockModel(side_effect=side_effect)

    agent = Agent(
      model=model,
      tools=[web_search],
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Search for Python tutorials")

    # Check the thinking call's system prompt contains the tool name
    thinking_messages = calls[0]["messages"]
    system_msg = thinking_messages[0]
    assert "web_search" in system_msg.content

  @pytest.mark.asyncio
  async def test_thinking_prompt_includes_instructions_summary(self):
    """When agent has instructions, the thinking prompt includes a truncated summary."""
    calls: List[Dict[str, Any]] = []

    def side_effect(messages, tools, **kwargs):
      response = MagicMock()
      response.tool_executions = []
      response.tool_calls = []
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      response_format = kwargs.get("response_format")
      if response_format is not None:
        response.content = MOCK_THINKING_JSON
        calls.append({"type": "thinking", "messages": messages})
      else:
        response.content = "Final answer."
        calls.append({"type": "main", "messages": messages})

      return response

    model = MockModel(side_effect=side_effect)

    agent = Agent(
      model=model,
      instructions="You are an expert Python developer who specializes in async programming.",
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Help me with async")

    # The thinking prompt should include the agent's role
    thinking_messages = calls[0]["messages"]
    system_msg = thinking_messages[0]
    assert "Your role:" in system_msg.content
    assert "Python developer" in system_msg.content

  @pytest.mark.asyncio
  async def test_thinking_prompt_indicates_knowledge_availability(self):
    """When knowledge context is available, thinking prompt mentions it."""
    from definable.run.base import RunContext

    model = _make_thinking_model()
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # Build a context-aware prompt manually to test the method
    context = RunContext(run_id="test", session_id="test")
    context.knowledge_context = "<knowledge>Some retrieved docs</knowledge>"
    tools: Dict[str, Any] = {}

    prompt = agent._build_thinking_prompt(context, tools)
    assert "knowledge base context is available" in prompt

  @pytest.mark.asyncio
  async def test_thinking_injection_positioned_before_knowledge(self):
    """System prompt has: instructions → <analysis> → knowledge (not knowledge → <analysis>)."""
    calls: List[Dict[str, Any]] = []

    def side_effect(messages, tools, **kwargs):
      response = MagicMock()
      response.tool_executions = []
      response.tool_calls = []
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      response_format = kwargs.get("response_format")
      if response_format is not None:
        response.content = MOCK_THINKING_JSON
        calls.append({"type": "thinking", "messages": messages})
      else:
        response.content = "Final answer."
        calls.append({"type": "main", "messages": messages})

      return response

    model = MockModel(side_effect=side_effect)

    # We need knowledge context to be injected. Set it up via RunContext manipulation.
    # The simplest way: use config with knowledge that returns content.
    # However, since we use side_effect model and knowledge requires real setup,
    # we'll test the ordering by checking _execute_run behavior directly.
    agent = Agent(
      model=model,
      thinking=True,
      instructions="You are a helpful assistant.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # Manually set knowledge_context before calling _execute_run to simulate
    from definable.run.agent import RunInput
    from definable.run.base import RunContext

    ctx = RunContext(
      run_id="test-ordering",
      session_id="test",
      metadata={"_messages": [], "_knowledge_position": "system"},
    )
    ctx.knowledge_context = "<knowledge>Retrieved documents here</knowledge>"

    from definable.models.message import Message

    test_messages = [Message(role="user", content="Test")]

    await agent._execute_run(ctx, test_messages, RunInput(input_content="Test"))

    # The main call system message should have <analysis> BEFORE <knowledge>
    main_messages = calls[-1]["messages"]
    system_msg = next((m for m in main_messages if m.role == "system"), None)
    assert system_msg is not None

    analysis_pos = system_msg.content.find("<analysis>")
    knowledge_pos = system_msg.content.find("<knowledge>")
    assert analysis_pos != -1, "Should contain <analysis> tag"
    assert knowledge_pos != -1, "Should contain <knowledge> tag"
    assert analysis_pos < knowledge_pos, "<analysis> should come before <knowledge>"

  @pytest.mark.asyncio
  async def test_thinking_with_tool_plan_maps_to_two_steps(self):
    """ThinkingOutput with tool_plan maps to 2 ReasoningStep objects."""
    model = _make_thinking_model(structured_json=MOCK_THINKING_WITH_TOOLS_JSON)
    agent = Agent(
      model=model,
      thinking=True,
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Find current data")

    assert output.reasoning_steps is not None
    assert len(output.reasoning_steps) == 2
    assert output.reasoning_steps[0].title == "Analysis"
    assert output.reasoning_steps[1].title == "Tool Plan"
    assert "web_search" in output.reasoning_steps[1].reasoning

  @pytest.mark.asyncio
  async def test_thinking_custom_instructions_bypass_context_aware(self):
    """ThinkingConfig(instructions='custom') bypasses context-aware prompt building."""
    calls: List[Dict[str, Any]] = []

    def side_effect(messages, tools, **kwargs):
      response = MagicMock()
      response.tool_executions = []
      response.tool_calls = []
      response.response_usage = Metrics()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None

      response_format = kwargs.get("response_format")
      if response_format is not None:
        response.content = MOCK_THINKING_JSON
        calls.append({"type": "thinking", "messages": messages})
      else:
        response.content = "Final answer."
        calls.append({"type": "main", "messages": messages})

      return response

    model = MockModel(side_effect=side_effect)
    custom_prompt = "Just analyze briefly."

    agent = Agent(
      model=model,
      instructions="You are an expert.",
      thinking=ThinkingConfig(instructions=custom_prompt),
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    await agent.arun("Hello")

    # The thinking call should use EXACTLY the custom prompt, not the built one
    thinking_messages = calls[0]["messages"]
    system_msg = thinking_messages[0]
    assert system_msg.content == custom_prompt
    # Should NOT contain context-aware elements
    assert "Your role:" not in system_msg.content
    assert "Available tools:" not in system_msg.content
