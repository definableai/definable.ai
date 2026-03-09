"""Tests for unified thinking — native thinking auto-detection and event unification.

Tests the unified thinking contract:
  - Agent(thinking=True) auto-detects native thinking support
  - Both native and Definable paths emit the same 3 events:
    ReasoningStarted → ReasoningContentDelta → ReasoningCompleted
  - Thinking.should_use_native() correctly routes based on model capabilities
  - Thinking.resolve_budget_tokens() maps effort to budget
  - Native thinking configures model params correctly
  - Definable fallback still works for non-thinking models
"""

import pytest

from definable.agent.reasoning.thinking import EFFORT_BUDGET_MAP, Thinking


# ═══════════════════════════════════════════════════════════════════════
# Thinking dataclass — new fields
# ═══════════════════════════════════════════════════════════════════════


class TestThinkingNewFields:
  def test_budget_tokens_default_none(self):
    t = Thinking()
    assert t.budget_tokens is None

  def test_budget_tokens_explicit(self):
    t = Thinking(budget_tokens=20000)
    assert t.budget_tokens == 20000

  def test_mode_default_none(self):
    t = Thinking()
    assert t.mode is None

  def test_mode_native(self):
    t = Thinking(mode="native")
    assert t.mode == "native"

  def test_mode_definable(self):
    t = Thinking(mode="definable")
    assert t.mode == "definable"

  def test_resolve_budget_tokens_from_effort(self):
    assert Thinking(effort="low").resolve_budget_tokens() == EFFORT_BUDGET_MAP["low"]
    assert Thinking(effort="medium").resolve_budget_tokens() == EFFORT_BUDGET_MAP["medium"]
    assert Thinking(effort="high").resolve_budget_tokens() == EFFORT_BUDGET_MAP["high"]

  def test_resolve_budget_tokens_explicit_overrides_effort(self):
    t = Thinking(effort="low", budget_tokens=50000)
    assert t.resolve_budget_tokens() == 50000

  def test_effort_budget_map_values(self):
    assert EFFORT_BUDGET_MAP["low"] == 4096
    assert EFFORT_BUDGET_MAP["medium"] == 10000
    assert EFFORT_BUDGET_MAP["high"] == 32000


# ═══════════════════════════════════════════════════════════════════════
# Thinking.should_use_native()
# ═══════════════════════════════════════════════════════════════════════


class _FakeModel:
  """Minimal model stub for testing should_use_native."""

  def __init__(self, *, supports_native_thinking: bool = False, model_id: str = "fake"):
    self.supports_native_thinking = supports_native_thinking
    self.id = model_id


class TestShouldUseNative:
  def test_auto_detect_native_model(self):
    model = _FakeModel(supports_native_thinking=True)
    t = Thinking()
    assert t.should_use_native(model) is True  # type: ignore[arg-type]

  def test_auto_detect_non_native_model(self):
    model = _FakeModel(supports_native_thinking=False)
    t = Thinking()
    assert t.should_use_native(model) is False  # type: ignore[arg-type]

  def test_mode_native_with_support(self):
    model = _FakeModel(supports_native_thinking=True)
    t = Thinking(mode="native")
    assert t.should_use_native(model) is True  # type: ignore[arg-type]

  def test_mode_native_without_support_raises(self):
    model = _FakeModel(supports_native_thinking=False, model_id="no-think-model")
    t = Thinking(mode="native")
    with pytest.raises(ValueError, match="does not support native thinking"):
      t.should_use_native(model)  # type: ignore[arg-type]

  def test_mode_definable_even_with_native_support(self):
    model = _FakeModel(supports_native_thinking=True)
    t = Thinking(mode="definable")
    assert t.should_use_native(model) is False  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Model supports_native_thinking flag
# ═══════════════════════════════════════════════════════════════════════


class TestModelNativeThinkingFlag:
  def test_base_model_default_false(self):
    from definable.model.base import Model

    assert Model.supports_native_thinking is False

  def test_claude_thinking_models_have_flag(self):
    from definable.model.anthropic.claude import Claude

    model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
    assert model.supports_native_thinking is True

  def test_claude_non_thinking_model_no_flag(self):
    from definable.model.anthropic.claude import Claude

    model = Claude(id="claude-3-5-haiku-20241022", api_key="test")
    assert model.supports_native_thinking is False

  def test_mock_model_default_false(self):
    from definable.agent.testing import MockModel

    model = MockModel()
    assert model.supports_native_thinking is False


# ═══════════════════════════════════════════════════════════════════════
# Event unification — Definable fallback path (via _emit / trace writer)
# ═══════════════════════════════════════════════════════════════════════


class TestDefinableFallbackEvents:
  """Definable's thinking layer emits ReasoningContentDelta via _emit (trace writer)."""

  @pytest.mark.asyncio
  async def test_fallback_emits_unified_events_via_emit(self):
    """When model doesn't support native thinking, Definable layer emits 3 events via _emit."""
    import json

    from definable.agent import Agent
    from definable.agent.testing import MockModel

    thinking_response = json.dumps({
      "analysis": "The user wants weather info.",
      "approach": "Use the search tool.",
      "tool_plan": ["search"],
    })

    model = MockModel(
      responses=["_unused_", "It's sunny in Tokyo!"],
      structured_responses=[thinking_response],
    )

    emitted_events = []
    agent = Agent(model=model, thinking=True, instructions="Be helpful.")  # type: ignore[arg-type]
    # Subscribe to _emit to capture trace events
    original_emit = agent._emit

    def capture_emit(event):
      emitted_events.append(event)
      original_emit(event)

    agent._emit = capture_emit  # type: ignore[method-assign]

    await agent.arun("What's the weather?")

    event_types = [e.event for e in emitted_events if hasattr(e, "event")]
    assert "ReasoningStarted" in event_types
    assert "ReasoningContentDelta" in event_types
    assert "ReasoningCompleted" in event_types
    # ReasoningStep should NOT be emitted
    assert "ReasoningStep" not in event_types

  @pytest.mark.asyncio
  async def test_fallback_delta_contains_analysis(self):
    """The ReasoningContentDelta from Definable layer contains flattened analysis text."""
    import json

    from definable.agent import Agent
    from definable.agent.testing import MockModel

    thinking_response = json.dumps({
      "analysis": "Complex query about weather.",
      "approach": "Look it up.",
      "tool_plan": None,
    })

    model = MockModel(
      responses=["_unused_", "It's sunny!"],
      structured_responses=[thinking_response],
    )

    emitted_events = []
    agent = Agent(model=model, thinking=True, instructions="Be helpful.")  # type: ignore[arg-type]
    original_emit = agent._emit

    def capture_emit(event):
      emitted_events.append(event)
      original_emit(event)

    agent._emit = capture_emit  # type: ignore[method-assign]

    await agent.arun("Weather?")

    delta_events = [e for e in emitted_events if hasattr(e, "event") and e.event == "ReasoningContentDelta"]
    assert len(delta_events) == 1
    assert "Complex query about weather" in delta_events[0].reasoning_content
    assert "Look it up" in delta_events[0].reasoning_content


# ═══════════════════════════════════════════════════════════════════════
# _enable_native_thinking — model configuration
# ═══════════════════════════════════════════════════════════════════════


class TestEnableNativeThinking:
  def test_claude_gets_thinking_dict(self):
    from definable.agent import Agent
    from definable.model.anthropic.claude import Claude

    model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
    agent = Agent(model=model, thinking=Thinking(effort="high"))

    assert model.thinking is None
    agent._enable_native_thinking()
    assert model.thinking == {"type": "enabled", "budget_tokens": 32000}

  def test_claude_existing_thinking_not_overwritten(self):
    from definable.agent import Agent
    from definable.model.anthropic.claude import Claude

    model = Claude(
      id="claude-sonnet-4-5-20250929",
      api_key="test",
      thinking={"type": "enabled", "budget_tokens": 5000},
    )
    agent = Agent(model=model, thinking=Thinking(effort="high"))
    agent._enable_native_thinking()
    # Should NOT overwrite existing thinking config
    assert model.thinking == {"type": "enabled", "budget_tokens": 5000}

  def test_effort_budget_mapping(self):
    from definable.agent import Agent
    from definable.model.anthropic.claude import Claude

    for effort, expected_budget in [("low", 4096), ("medium", 10000), ("high", 32000)]:
      model = Claude(id="claude-sonnet-4-5-20250929", api_key="test")
      agent = Agent(model=model, thinking=Thinking(effort=effort))  # type: ignore[arg-type]
      agent._enable_native_thinking()
      assert model.thinking["budget_tokens"] == expected_budget, f"Failed for effort={effort}"  # type: ignore[index]


# ═══════════════════════════════════════════════════════════════════════
# AgentLoop — native thinking events (non-streaming)
# ═══════════════════════════════════════════════════════════════════════


class TestAgentLoopNativeThinking:
  @pytest.mark.asyncio
  async def test_non_streaming_emits_reasoning_events(self):
    """Non-streaming loop with native thinking emits 3 reasoning events."""
    from unittest.mock import MagicMock

    from definable.agent.config import AgentConfig
    from definable.agent.events import RunContext
    from definable.agent.loop import AgentLoop
    from definable.model.message import Message
    from definable.model.metrics import Metrics

    # Build a mock model that returns reasoning_content
    mock_response = MagicMock()
    mock_response.content = "The answer is 42."
    mock_response.tool_calls = []
    mock_response.response_usage = Metrics()
    mock_response.reasoning_content = "Let me think about this deeply..."
    mock_response.redacted_reasoning_content = None
    mock_response.parsed = None

    async def mock_ainvoke(*, messages=None, assistant_message=None, tools=None, response_format=None, **kwargs):
      return mock_response

    model = MagicMock()
    model.id = "claude-test"
    model.provider = "Anthropic"
    model.ainvoke = mock_ainvoke

    context = RunContext(run_id="test", session_id="test")
    messages = [
      Message(role="system", content="You are helpful."),
      Message(role="user", content="What is the answer?"),
    ]

    emitted: list = []
    loop = AgentLoop(
      model=model,
      tools={},
      messages=messages,
      context=context,
      config=AgentConfig(),
      streaming=False,
      native_thinking=True,
      emit_fn=lambda e: emitted.append(e),
      agent_id="test",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    event_types = [e.event for e in events]
    assert "ReasoningStarted" in event_types
    assert "ReasoningContentDelta" in event_types
    assert "ReasoningCompleted" in event_types

    # Check reasoning content
    delta_events = [e for e in events if e.event == "ReasoningContentDelta"]
    assert len(delta_events) == 1
    assert delta_events[0].reasoning_content == "Let me think about this deeply..."  # type: ignore[union-attr]

    # Check that native_reasoning_content is captured
    assert loop.native_reasoning_content == "Let me think about this deeply..."

  @pytest.mark.asyncio
  async def test_non_streaming_no_events_without_native_thinking(self):
    """Without native_thinking=True, reasoning_content on response doesn't produce events."""
    from unittest.mock import MagicMock

    from definable.agent.config import AgentConfig
    from definable.agent.events import RunContext
    from definable.agent.loop import AgentLoop
    from definable.model.message import Message
    from definable.model.metrics import Metrics

    mock_response = MagicMock()
    mock_response.content = "The answer."
    mock_response.tool_calls = []
    mock_response.response_usage = Metrics()
    mock_response.reasoning_content = "Some reasoning..."
    mock_response.redacted_reasoning_content = None
    mock_response.parsed = None

    async def mock_ainvoke(*, messages=None, assistant_message=None, tools=None, response_format=None, **kwargs):
      return mock_response

    model = MagicMock()
    model.id = "test"
    model.provider = "test"
    model.ainvoke = mock_ainvoke

    context = RunContext(run_id="test", session_id="test")
    messages = [Message(role="user", content="Hi")]

    loop = AgentLoop(
      model=model,
      tools={},
      messages=messages,
      context=context,
      config=AgentConfig(),
      streaming=False,
      native_thinking=False,
      emit_fn=lambda e: None,
      agent_id="test",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    event_types = [e.event for e in events]
    assert "ReasoningStarted" not in event_types
    assert "ReasoningContentDelta" not in event_types

  @pytest.mark.asyncio
  async def test_streaming_emits_reasoning_events(self):
    """Streaming loop with native thinking emits reasoning events from chunks."""
    from unittest.mock import MagicMock

    from definable.agent.config import AgentConfig
    from definable.agent.events import RunContext
    from definable.agent.loop import AgentLoop
    from definable.model.message import Message
    from definable.model.metrics import Metrics

    # Build chunks: reasoning first, then content
    chunks = []
    for word in ["Let ", "me ", "think..."]:
      chunk = MagicMock()
      chunk.content = None
      chunk.reasoning_content = word
      chunk.tool_calls = None
      chunk.response_usage = None
      chunk.parsed = None
      chunks.append(chunk)

    for word in ["The ", "answer."]:
      chunk = MagicMock()
      chunk.content = word
      chunk.reasoning_content = None
      chunk.tool_calls = None
      chunk.response_usage = Metrics() if word == "answer." else None
      chunk.parsed = None
      chunks.append(chunk)

    async def mock_stream(*, messages=None, assistant_message=None, tools=None, response_format=None, **kwargs):
      for c in chunks:
        yield c

    model = MagicMock()
    model.id = "claude-test"
    model.provider = "Anthropic"
    model.ainvoke_stream = mock_stream

    context = RunContext(run_id="test", session_id="test")
    messages = [Message(role="user", content="What's the answer?")]

    loop = AgentLoop(
      model=model,
      tools={},
      messages=messages,
      context=context,
      config=AgentConfig(),
      streaming=True,
      native_thinking=True,
      emit_fn=lambda e: None,
      agent_id="test",
      agent_name="test",
    )

    events = []
    async for event in loop.run():
      events.append(event)

    event_types = [e.event for e in events]

    # Should have reasoning lifecycle events
    assert "ReasoningStarted" in event_types
    assert "ReasoningContentDelta" in event_types
    assert "ReasoningCompleted" in event_types
    assert "RunContent" in event_types

    # Reasoning events should come before content events
    first_reasoning = event_types.index("ReasoningStarted")
    first_content = event_types.index("RunContent")
    assert first_reasoning < first_content

    # Check accumulated reasoning content
    deltas = [e for e in events if e.event == "ReasoningContentDelta"]
    assert len(deltas) == 3
    full_reasoning = "".join(d.reasoning_content for d in deltas)  # type: ignore[union-attr, misc]
    assert full_reasoning == "Let me think..."

    assert loop.native_reasoning_content == "Let me think..."


# ═══════════════════════════════════════════════════════════════════════
# Pipeline ThinkPhase — native vs fallback routing
# ═══════════════════════════════════════════════════════════════════════


class TestThinkPhaseRouting:
  def test_should_run_false_when_no_thinking(self):
    from unittest.mock import MagicMock

    from definable.agent.pipeline.phases.think import ThinkPhase

    agent = MagicMock()
    agent._thinking = None
    phase = ThinkPhase(agent)
    state = MagicMock()
    assert phase.should_run(state) is False

  def test_should_run_true_when_thinking_set(self):
    from unittest.mock import MagicMock

    from definable.agent.pipeline.phases.think import ThinkPhase

    agent = MagicMock()
    agent._thinking = Thinking()
    phase = ThinkPhase(agent)
    state = MagicMock()
    assert phase.should_run(state) is True


# ═══════════════════════════════════════════════════════════════════════
# Integration: Agent(thinking=True) with non-native model
# ═══════════════════════════════════════════════════════════════════════


class TestAgentThinkingIntegration:
  @pytest.mark.asyncio
  async def test_thinking_true_with_non_native_model_uses_fallback(self):
    """Agent(thinking=True) with a model that doesn't support native thinking uses Definable layer."""
    import json

    from definable.agent import Agent
    from definable.agent.testing import MockModel

    thinking_response = json.dumps({
      "analysis": "Simple greeting.",
      "approach": "Respond warmly.",
      "tool_plan": None,
    })

    # responses[0] consumed by thinking (structured), responses[1] by main call
    model = MockModel(
      responses=["_unused_", "Hello there!"],
      structured_responses=[thinking_response],
    )
    assert model.supports_native_thinking is False

    agent = Agent(model=model, thinking=True)  # type: ignore[arg-type]
    result = await agent.arun("Hi!")

    # Should have made 2 calls: thinking (structured) + main response
    assert model.call_count == 2
    assert result.content == "Hello there!"

  @pytest.mark.asyncio
  async def test_thinking_mode_definable_forces_fallback(self):
    """thinking=Thinking(mode="definable") forces Definable's layer even with capable model."""
    import json

    from definable.agent import Agent
    from definable.agent.testing import MockModel

    thinking_response = json.dumps({
      "analysis": "Testing forced mode.",
      "approach": "Verify behavior.",
      "tool_plan": None,
    })

    model = MockModel(
      responses=["_unused_", "Result."],
      structured_responses=[thinking_response],
    )
    # Pretend this model supports native thinking
    model.supports_native_thinking = True

    agent = Agent(model=model, thinking=Thinking(mode="definable"))  # type: ignore[arg-type]
    await agent.arun("Test.")

    # Should use Definable layer (2 calls)
    assert model.call_count == 2

  @pytest.mark.asyncio
  async def test_reasoning_content_on_run_output(self):
    """RunOutput.reasoning_content should be populated from Definable layer."""
    import json

    from definable.agent import Agent
    from definable.agent.testing import MockModel

    thinking_response = json.dumps({
      "analysis": "User says hello.",
      "approach": "Greet back.",
      "tool_plan": None,
    })

    model = MockModel(
      responses=["_unused_", "Hello!"],
      structured_responses=[thinking_response],
    )

    agent = Agent(model=model, thinking=True)  # type: ignore[arg-type]
    result = await agent.arun("Hi!")

    assert result.reasoning_content is not None
    assert "User says hello" in result.reasoning_content


# ═══════════════════════════════════════════════════════════════════════
# Thinking.effort field (merged from test_thinking_effort.py)
# ═══════════════════════════════════════════════════════════════════════


class TestThinkingEffortField:
  def test_default_effort_is_medium(self):
    t = Thinking()
    assert t.effort == "medium"

  def test_effort_low(self):
    t = Thinking(effort="low")
    assert t.effort == "low"

  def test_effort_high(self):
    t = Thinking(effort="high")
    assert t.effort == "high"

  def test_effort_independent_of_trigger(self):
    """Effort and trigger are orthogonal settings."""
    t = Thinking(trigger="auto", effort="high")
    assert t.trigger == "auto"
    assert t.effort == "high"

    t2 = Thinking(trigger="never", effort="low")
    assert t2.trigger == "never"
    assert t2.effort == "low"


# ═══════════════════════════════════════════════════════════════════════
# ThinkingOutput — considerations field (merged from test_thinking_effort.py)
# ═══════════════════════════════════════════════════════════════════════


class TestThinkingOutputConsiderations:
  def test_considerations_default_none(self):
    from definable.agent.reasoning.step import ThinkingOutput

    output = ThinkingOutput(analysis="test", approach="test")  # type: ignore[call-arg]
    assert output.considerations is None

  def test_considerations_populated(self):
    from definable.agent.reasoning.step import ThinkingOutput

    output = ThinkingOutput(  # type: ignore[call-arg]
      analysis="Complex query",
      approach="Multi-step plan",
      considerations="Risk: rate limits may apply. Alternative: use caching.",
    )
    assert output.considerations == "Risk: rate limits may apply. Alternative: use caching."

  def test_considerations_with_tool_plan(self):
    from definable.agent.reasoning.step import ThinkingOutput

    output = ThinkingOutput(
      analysis="Need data",
      approach="Search and analyze",
      tool_plan=["search", "analyze"],
      considerations="Edge case: empty results.",
    )
    assert output.tool_plan == ["search", "analyze"]
    assert output.considerations is not None


# ═══════════════════════════════════════════════════════════════════════
# thinking_output_to_reasoning_steps — with considerations (merged from test_thinking_effort.py)
# ═══════════════════════════════════════════════════════════════════════


class TestReasoningStepsWithConsiderations:
  def test_no_considerations_no_extra_step(self):
    from definable.agent.reasoning.step import ThinkingOutput, thinking_output_to_reasoning_steps

    output = ThinkingOutput(analysis="Simple", approach="Direct answer")  # type: ignore[call-arg]
    steps = thinking_output_to_reasoning_steps(output)
    assert len(steps) == 1
    assert steps[0].title == "Analysis"

  def test_considerations_adds_third_step(self):
    from definable.agent.reasoning.step import NextAction, ThinkingOutput, thinking_output_to_reasoning_steps

    output = ThinkingOutput(
      analysis="Complex",
      approach="Multi-step",
      tool_plan=["search"],
      considerations="Watch for rate limits.",
    )
    steps = thinking_output_to_reasoning_steps(output)
    assert len(steps) == 3
    assert steps[0].title == "Analysis"
    assert steps[1].title == "Tool Plan"
    assert steps[2].title == "Considerations"
    assert steps[2].reasoning == "Watch for rate limits."
    assert steps[2].next_action == NextAction.FINAL_ANSWER

  def test_considerations_without_tools_adds_second_step(self):
    from definable.agent.reasoning.step import ThinkingOutput, thinking_output_to_reasoning_steps

    output = ThinkingOutput(  # type: ignore[call-arg]
      analysis="Complex",
      approach="Reason carefully",
      considerations="Multiple valid interpretations exist.",
    )
    steps = thinking_output_to_reasoning_steps(output)
    assert len(steps) == 2
    assert steps[0].title == "Analysis"
    assert steps[1].title == "Considerations"

  def test_tool_plan_next_action_continues_when_considerations_present(self):
    """When considerations follow, tool plan step should CONTINUE, not FINAL_ANSWER."""
    from definable.agent.reasoning.step import NextAction, ThinkingOutput, thinking_output_to_reasoning_steps

    output = ThinkingOutput(
      analysis="Need data",
      approach="Fetch and analyze",
      tool_plan=["fetch"],
      considerations="Data may be stale.",
    )
    steps = thinking_output_to_reasoning_steps(output)
    assert steps[1].title == "Tool Plan"
    assert steps[1].next_action == NextAction.CONTINUE
