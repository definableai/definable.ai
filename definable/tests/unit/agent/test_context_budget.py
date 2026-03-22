"""Unit tests for token budget allocation and context window utilities."""

import pytest

from definable.agent.context import Context, TokenBudget
from definable.agent.context.budget import (
  MODEL_CONTEXT_WINDOWS,
  allocate_budget,
  get_context_window,
  resolve_allocation,
)
from definable.agent.context.manager import ContextManager


@pytest.mark.unit
class TestAllocateBudget:
  def test_basic_allocation(self):
    alloc = allocate_budget(context_window=128_000)
    assert alloc.context_window == 128_000
    assert alloc.output_reserve == 4096
    assert alloc.total_available == 128_000 - 4096
    assert alloc.system_prompt == int(alloc.total_available * 0.30)
    assert alloc.knowledge == int(alloc.total_available * 0.15)
    assert alloc.memory == int(alloc.total_available * 0.10)
    assert alloc.history == int(alloc.total_available * 0.40)

  def test_custom_percentages(self):
    alloc = allocate_budget(
      context_window=100_000,
      output_reserve=2000,
      system_prompt_pct=0.50,
      knowledge_pct=0.10,
      memory_pct=0.10,
      history_pct=0.20,
    )
    available = 98_000
    assert alloc.system_prompt == int(available * 0.50)
    assert alloc.knowledge == int(available * 0.10)
    assert alloc.history == int(available * 0.20)

  def test_zero_context_window(self):
    alloc = allocate_budget(context_window=0)
    assert alloc.total_available == 0
    assert alloc.system_prompt == 0

  def test_output_reserve_larger_than_window(self):
    alloc = allocate_budget(context_window=1000, output_reserve=5000)
    assert alloc.total_available == 0
    assert alloc.system_prompt == 0


@pytest.mark.unit
class TestGetContextWindow:
  def test_known_model_exact_match(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "gpt-4o"
    assert get_context_window(mock) == 128_000

  def test_known_model_prefix_match(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "gpt-4o-2024-08-06"
    assert get_context_window(mock) == 128_000

  def test_explicit_context_window_takes_priority(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "gpt-4o"
    mock.context_window = 256_000
    assert get_context_window(mock) == 256_000

  def test_unknown_model_returns_none(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "some-unknown-model-xyz"
    assert get_context_window(mock) is None

  def test_claude_models_in_lookup(self):
    assert "claude-sonnet-4-5-20250929" in MODEL_CONTEXT_WINDOWS
    assert "claude-opus-4-20250514" in MODEL_CONTEXT_WINDOWS


@pytest.mark.unit
class TestResolveAllocation:
  def test_with_explicit_context_window(self):
    budget = TokenBudget(context_window=128_000)
    alloc = resolve_allocation(budget)
    assert alloc is not None
    assert alloc.context_window == 128_000

  def test_with_model_auto_detection(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "gpt-4o"
    budget = TokenBudget()  # No explicit context_window
    alloc = resolve_allocation(budget, model=mock)
    assert alloc is not None
    assert alloc.context_window == 128_000

  def test_unknown_window_returns_none(self):
    from definable.agent.testing import MockModel

    mock = MockModel()
    mock.id = "unknown-model"
    budget = TokenBudget()
    alloc = resolve_allocation(budget, model=mock)
    assert alloc is None

  def test_non_budget_object_returns_none(self):
    alloc = resolve_allocation("not a budget")
    assert alloc is None


@pytest.mark.unit
class TestTokenBudgetValidation:
  def test_valid_budget(self):
    budget = TokenBudget(system_prompt_pct=0.3, knowledge_pct=0.2, memory_pct=0.1, history_pct=0.3)
    assert budget.output_reserve == 4096

  def test_budget_exceeding_100_percent(self):
    with pytest.raises(ValueError, match="must be <= 1.0"):
      TokenBudget(system_prompt_pct=0.5, knowledge_pct=0.3, memory_pct=0.2, history_pct=0.5)

  def test_budget_exactly_100_percent(self):
    budget = TokenBudget(system_prompt_pct=0.25, knowledge_pct=0.25, memory_pct=0.25, history_pct=0.25)
    assert budget is not None


@pytest.mark.unit
class TestContextManagerBudgetEnforcement:
  def test_knowledge_trimmed_to_budget(self):
    """When knowledge context exceeds its budget, it should be truncated."""
    budget = TokenBudget(context_window=1000, output_reserve=200, knowledge_pct=0.05)
    mgr = ContextManager(Context(token_budget=budget))

    # 0.05 * 800 = 40 tokens budget for knowledge
    huge_knowledge = "word " * 200  # ~200 tokens
    result = mgr.build_system_prompt(
      instructions="Be helpful.",
      knowledge_context=huge_knowledge,
    )
    # Knowledge should be present but truncated
    assert "Be helpful." in result
    # The full knowledge should NOT be present (it was truncated)
    assert len(result) < len(huge_knowledge)

  def test_memory_trimmed_to_budget(self):
    budget = TokenBudget(context_window=1000, output_reserve=200, memory_pct=0.05)
    mgr = ContextManager(Context(token_budget=budget))

    huge_memory = "message " * 200
    result = mgr.build_system_prompt(
      instructions="Core.",
      memory_context=huge_memory,
    )
    assert "Core." in result
    assert len(result) < len(huge_memory)

  def test_no_budget_no_trimming(self):
    mgr = ContextManager(Context())  # No token_budget
    large_content = "word " * 500
    result = mgr.build_system_prompt(knowledge_context=large_content)
    # Should contain all content since no budget limits apply
    assert "word" in result

  def test_allocation_property(self):
    budget = TokenBudget(context_window=128_000)
    mgr = ContextManager(Context(token_budget=budget))
    assert mgr.allocation is not None
    assert mgr.allocation.context_window == 128_000

  def test_stats_include_budget_info(self):
    budget = TokenBudget(context_window=128_000)
    mgr = ContextManager(Context(token_budget=budget))
    mgr.build_system_prompt(instructions="Hello")
    stats = mgr.last_stats
    assert stats is not None
    assert "budget" in stats
    assert stats["budget"]["context_window"] == 128_000


@pytest.mark.unit
class TestContextManagerHistoryTokenBudget:
  def test_history_trimmed_by_token_budget(self):
    """When history exceeds its token budget, oldest messages are dropped."""
    from definable.model.message import Message

    # Very small history budget: 0.05 * (1000 - 200) = 40 tokens
    budget = TokenBudget(context_window=1000, output_reserve=200, history_pct=0.05)
    mgr = ContextManager(Context(history_strategy="none", token_budget=budget))

    # Create many messages that exceed 40 tokens total
    msgs = [Message(role="user", content=f"This is message number {i} with lots of content") for i in range(20)]
    result = mgr.trim_history(msgs)
    # Should have fewer messages than input
    assert len(result) < len(msgs)
    # Should keep the most recent (last) messages
    assert result[-1].content == msgs[-1].content

  def test_history_under_budget_not_trimmed(self):
    from definable.model.message import Message

    budget = TokenBudget(context_window=128_000, history_pct=0.40)
    mgr = ContextManager(Context(history_strategy="none", token_budget=budget))

    msgs = [Message(role="user", content="short")]
    result = mgr.trim_history(msgs)
    assert len(result) == 1


@pytest.mark.unit
class TestCacheOptimizationSplit:
  def test_cache_split_separates_static_dynamic(self):
    mgr = ContextManager(Context(cache_optimization=True))
    static, dynamic = mgr.build_system_prompt_split(
      instructions="You are a helpful assistant.",
      layer_guide="## Capabilities\nMemory, Knowledge",
      knowledge_context="Retrieved doc A, doc B",
      memory_context="Previous conversation summary",
    )
    # Static: instructions + layer guide (cacheable)
    assert "helpful assistant" in static
    assert "Capabilities" in static
    # Dynamic: knowledge + memory (not cacheable)
    assert "doc A" in dynamic
    assert "conversation summary" in dynamic
    # Cross-check: no leakage
    assert "doc A" not in static
    assert "helpful assistant" not in dynamic

  def test_cache_disabled_returns_empty_static(self):
    mgr = ContextManager(Context(cache_optimization=False))
    static, dynamic = mgr.build_system_prompt_split(
      instructions="Core",
      knowledge_context="Docs",
    )
    assert static == ""
    assert "Core" in dynamic
    assert "Docs" in dynamic
