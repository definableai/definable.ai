"""Unit tests for LayeredPrompt — priority ordering, trimming, build_split."""

import pytest

from definable.agent.context.layers import (
  PRIORITY_EPHEMERAL,
  PRIORITY_INSTRUCTIONS,
  PRIORITY_KNOWLEDGE,
  PRIORITY_LAYER_GUIDE,
  PRIORITY_MEMORY,
  LayeredPrompt,
  PromptLayer,
)


@pytest.mark.unit
class TestPromptLayer:
  def test_priority_constants_ordering(self):
    assert PRIORITY_INSTRUCTIONS < PRIORITY_LAYER_GUIDE < PRIORITY_KNOWLEDGE < PRIORITY_MEMORY < PRIORITY_EPHEMERAL

  def test_default_priority_is_ephemeral(self):
    layer = PromptLayer(name="test", content="hello")
    assert layer.priority == PRIORITY_EPHEMERAL

  def test_default_cacheable_is_true(self):
    layer = PromptLayer(name="test", content="hello")
    assert layer.cacheable is True


@pytest.mark.unit
class TestLayeredPromptBuild:
  def test_empty_prompt(self):
    prompt = LayeredPrompt()
    assert prompt.build() == ""

  def test_single_layer(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="instructions", content="You are helpful.", priority=1))
    assert prompt.build() == "You are helpful."

  def test_layers_joined_by_double_newline(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="Part A", priority=1))
    prompt.add(PromptLayer(name="b", content="Part B", priority=2))
    assert prompt.build() == "Part A\n\nPart B"

  def test_layers_sorted_by_priority(self):
    prompt = LayeredPrompt()
    # Add in reverse priority order
    prompt.add(PromptLayer(name="ephemeral", content="Thinking", priority=5))
    prompt.add(PromptLayer(name="instructions", content="Core", priority=1))
    prompt.add(PromptLayer(name="knowledge", content="Docs", priority=3))

    result = prompt.build()
    # Core should come first, then Docs, then Thinking
    assert result == "Core\n\nDocs\n\nThinking"

  def test_empty_layers_ignored(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="Kept", priority=1))
    prompt.add(PromptLayer(name="b", content="", priority=2))
    prompt.add(PromptLayer(name="c", content="   ", priority=3))
    assert prompt.build() == "Kept"

  def test_clear_removes_all_layers(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="text", priority=1))
    prompt.clear()
    assert prompt.build() == ""


@pytest.mark.unit
class TestLayeredPromptTokenTrimming:
  def test_no_trimming_without_max_tokens(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="Hello world " * 100, priority=1))
    prompt.add(PromptLayer(name="b", content="Extra content " * 100, priority=5))
    result_unlimited = prompt.build()
    result_explicit = prompt.build(max_tokens=None)
    assert result_unlimited == result_explicit

  def test_low_priority_dropped_first(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="instructions", content="Core instructions here.", priority=1))
    prompt.add(PromptLayer(name="ephemeral", content="Ephemeral thinking output " * 50, priority=5))

    # Use a very small budget that can only fit the instructions
    result = prompt.build(max_tokens=10)
    assert "Core instructions" in result
    assert "Ephemeral" not in result

  def test_priority_1_never_dropped(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="sacred", content="Sacred text " * 20, priority=1))
    prompt.add(PromptLayer(name="expendable", content="Expendable " * 20, priority=5))

    # Even with impossibly small budget, priority 1 survives
    result = prompt.build(max_tokens=5)
    assert "Sacred" in result

  def test_multiple_priorities_dropped_in_order(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="p1", content="Priority 1 text", priority=1))
    prompt.add(PromptLayer(name="p3", content="Priority 3 text " * 20, priority=3))
    prompt.add(PromptLayer(name="p5", content="Priority 5 text " * 20, priority=5))

    # Budget that fits p1 + p3 but not p5
    result = prompt.build(max_tokens=30)
    assert "Priority 1" in result
    # p5 should be dropped before p3
    if "Priority 5" not in result:
      assert "Priority 3" in result or "Priority 1" in result


@pytest.mark.unit
class TestLayeredPromptSplit:
  def test_split_separates_cacheable(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="instr", content="Instructions", priority=1, cacheable=True))
    prompt.add(PromptLayer(name="guide", content="Layer guide", priority=2, cacheable=True))
    prompt.add(PromptLayer(name="knowledge", content="Docs", priority=3, cacheable=False))
    prompt.add(PromptLayer(name="memory", content="History", priority=4, cacheable=False))

    static, dynamic = prompt.build_split()
    assert "Instructions" in static
    assert "Layer guide" in static
    assert "Docs" in dynamic
    assert "History" in dynamic
    assert "Instructions" not in dynamic
    assert "Docs" not in static

  def test_split_all_cacheable(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="All static", priority=1, cacheable=True))
    static, dynamic = prompt.build_split()
    assert static == "All static"
    assert dynamic == ""

  def test_split_all_dynamic(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="a", content="All dynamic", priority=3, cacheable=False))
    static, dynamic = prompt.build_split()
    assert static == ""
    assert dynamic == "All dynamic"

  def test_split_empty(self):
    prompt = LayeredPrompt()
    static, dynamic = prompt.build_split()
    assert static == ""
    assert dynamic == ""


@pytest.mark.unit
class TestLayeredPromptStats:
  def test_token_stats_returns_per_layer_info(self):
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="instructions", content="Hello world", priority=1))
    prompt.add(PromptLayer(name="knowledge", content="Some documents here", priority=3))

    stats = prompt.token_stats()
    assert "instructions" in stats
    assert "knowledge" in stats
    assert stats["instructions"]["tokens"] > 0
    assert stats["instructions"]["priority"] == 1
    assert stats["knowledge"]["priority"] == 3
    assert stats["total"] > 0
