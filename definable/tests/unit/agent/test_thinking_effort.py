"""Tests for the Thinking effort parameter."""

from definable.agent.reasoning.step import (
  NextAction,
  ThinkingOutput,
  thinking_output_to_reasoning_steps,
)
from definable.agent.reasoning.thinking import Thinking

# ═══════════════════════════════════════════════════════════════════════
# Thinking dataclass — effort field
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
# ThinkingOutput — considerations field
# ═══════════════════════════════════════════════════════════════════════


class TestThinkingOutputConsiderations:
  def test_considerations_default_none(self):
    output = ThinkingOutput(analysis="test", approach="test")
    assert output.considerations is None

  def test_considerations_populated(self):
    output = ThinkingOutput(
      analysis="Complex query",
      approach="Multi-step plan",
      considerations="Risk: rate limits may apply. Alternative: use caching.",
    )
    assert output.considerations == "Risk: rate limits may apply. Alternative: use caching."

  def test_considerations_with_tool_plan(self):
    output = ThinkingOutput(
      analysis="Need data",
      approach="Search and analyze",
      tool_plan=["search", "analyze"],
      considerations="Edge case: empty results.",
    )
    assert output.tool_plan == ["search", "analyze"]
    assert output.considerations is not None


# ═══════════════════════════════════════════════════════════════════════
# thinking_output_to_reasoning_steps — with considerations
# ═══════════════════════════════════════════════════════════════════════


class TestReasoningStepsWithConsiderations:
  def test_no_considerations_no_extra_step(self):
    output = ThinkingOutput(analysis="Simple", approach="Direct answer")
    steps = thinking_output_to_reasoning_steps(output)
    assert len(steps) == 1
    assert steps[0].title == "Analysis"

  def test_considerations_adds_third_step(self):
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
    output = ThinkingOutput(
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
    output = ThinkingOutput(
      analysis="Need data",
      approach="Fetch and analyze",
      tool_plan=["fetch"],
      considerations="Data may be stale.",
    )
    steps = thinking_output_to_reasoning_steps(output)
    assert steps[1].title == "Tool Plan"
    assert steps[1].next_action == NextAction.CONTINUE


# ═══════════════════════════════════════════════════════════════════════
# Agent._EFFORT_PROMPTS
# ═══════════════════════════════════════════════════════════════════════


class TestEffortPrompts:
  def test_effort_prompts_exist_for_all_levels(self):
    from definable.agent.agent import Agent

    assert "low" in Agent._EFFORT_PROMPTS
    assert "medium" in Agent._EFFORT_PROMPTS
    assert "high" in Agent._EFFORT_PROMPTS

  def test_medium_is_empty(self):
    from definable.agent.agent import Agent

    assert Agent._EFFORT_PROMPTS["medium"] == ""

  def test_low_is_brief(self):
    from definable.agent.agent import Agent

    assert "brief" in Agent._EFFORT_PROMPTS["low"].lower()

  def test_high_mentions_considerations(self):
    from definable.agent.agent import Agent

    assert "considerations" in Agent._EFFORT_PROMPTS["high"].lower()


# ═══════════════════════════════════════════════════════════════════════
# Agent._format_thinking_injection — effort-aware formatting
# ═══════════════════════════════════════════════════════════════════════


class TestFormatThinkingInjection:
  def _make_output(self, **kwargs):
    defaults = {"analysis": "User needs X", "approach": "Do Y"}
    defaults.update(kwargs)
    return ThinkingOutput(**defaults)

  def test_low_compact_format(self):
    from definable.agent.agent import Agent

    output = self._make_output()
    result = Agent._format_thinking_injection(output, effort="low")
    assert result.startswith("<analysis>")
    assert result.endswith("</analysis>")
    assert "\n" not in result  # compact, single line

  def test_medium_compact_format(self):
    from definable.agent.agent import Agent

    output = self._make_output(tool_plan=["search"])
    result = Agent._format_thinking_injection(output, effort="medium")
    assert "Tools: search" in result
    assert "\n" not in result

  def test_high_expanded_format(self):
    from definable.agent.agent import Agent

    output = self._make_output(
      tool_plan=["search", "summarize"],
      considerations="Risk: data may be stale.",
    )
    result = Agent._format_thinking_injection(output, effort="high")
    assert "Analysis: User needs X" in result
    assert "Approach: Do Y" in result
    assert "Tools: search, summarize" in result
    assert "Considerations: Risk: data may be stale." in result
    assert "\n" in result  # multi-line

  def test_high_without_considerations(self):
    from definable.agent.agent import Agent

    output = self._make_output()
    result = Agent._format_thinking_injection(output, effort="high")
    assert "Analysis:" in result
    assert "Approach:" in result
    assert "Considerations:" not in result

  def test_default_effort_is_medium(self):
    from definable.agent.agent import Agent

    output = self._make_output()
    default_result = Agent._format_thinking_injection(output)
    medium_result = Agent._format_thinking_injection(output, effort="medium")
    assert default_result == medium_result
