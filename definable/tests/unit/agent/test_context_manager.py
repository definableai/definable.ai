"""Unit tests for ContextManager — system prompt assembly + history trimming."""

import pytest

from definable.agent.context import Context, TokenBudget
from definable.agent.context.manager import ContextManager
from definable.model.message import Message


@pytest.mark.unit
class TestContextManagerBuildSystemPrompt:
  def test_basic_assembly(self):
    mgr = ContextManager(Context())
    result = mgr.build_system_prompt(
      instructions="You are helpful.",
      skill_instructions="Use search when needed.",
    )
    assert "You are helpful." in result
    assert "Use search when needed." in result

  def test_all_sections_included(self):
    mgr = ContextManager(Context())
    result = mgr.build_system_prompt(
      instructions="Instructions",
      skill_instructions="Skills",
      layer_guide="## Capabilities",
      thinking_injection="<execution_strategy>Plan</execution_strategy>",
      knowledge_context="<docs>Doc A</docs>",
      research_context="Research findings",
      memory_context="<conversation_history>Hello</conversation_history>",
    )
    for section in ["Instructions", "Skills", "Capabilities", "execution_strategy", "Doc A", "Research", "conversation_history"]:
      assert section in result

  def test_empty_sections_skipped(self):
    mgr = ContextManager(Context())
    result = mgr.build_system_prompt(
      instructions="Core",
      skill_instructions=None,
      layer_guide="",
      knowledge_context=None,
    )
    assert result == "Core"

  def test_priority_ordering(self):
    """Instructions should appear before knowledge, which appears before thinking."""
    mgr = ContextManager(Context())
    result = mgr.build_system_prompt(
      instructions="INSTRUCTIONS_MARKER",
      knowledge_context="KNOWLEDGE_MARKER",
      thinking_injection="THINKING_MARKER",
    )
    instr_pos = result.index("INSTRUCTIONS_MARKER")
    know_pos = result.index("KNOWLEDGE_MARKER")
    think_pos = result.index("THINKING_MARKER")
    assert instr_pos < know_pos < think_pos

  def test_stats_populated_after_build(self):
    mgr = ContextManager(Context())
    mgr.build_system_prompt(instructions="Hello world")
    stats = mgr.last_stats
    assert stats is not None
    assert "instructions" in stats
    assert stats["instructions"]["tokens"] > 0
    assert stats["total"] > 0


@pytest.mark.unit
class TestContextManagerTrimHistory:
  def test_trim_removes_system_messages(self):
    mgr = ContextManager(Context(history_strategy="tail", max_history_messages=100))
    msgs = [
      Message(role="system", content="system prompt"),
      Message(role="user", content="hello"),
      Message(role="assistant", content="hi"),
    ]
    result = mgr.trim_history(msgs)
    assert all(m.role != "system" for m in result)
    assert len(result) == 2

  def test_trim_tail_strategy(self):
    mgr = ContextManager(Context(history_strategy="tail", max_history_messages=3))
    msgs = [Message(role="user", content=f"q{i}") for i in range(10)]
    result = mgr.trim_history(msgs)
    assert len(result) == 3

  def test_no_trim_strategy(self):
    mgr = ContextManager(Context(history_strategy="none"))
    msgs = [Message(role="user", content=f"q{i}") for i in range(100)]
    result = mgr.trim_history(msgs)
    assert len(result) == 100

  def test_trim_preserves_tool_pairs(self):
    """Tool-call groups should never be split."""
    tool_calls = [{"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    mgr = ContextManager(Context(history_strategy="tail", max_history_messages=4))
    msgs = [
      Message(role="user", content="q1"),
      Message(role="assistant", content="searching", tool_calls=tool_calls),
      Message(role="tool", content="result", tool_call_id="tc_1"),
      Message(role="user", content="q2"),
      Message(role="assistant", content="final answer"),
    ]
    result = mgr.trim_history(msgs)
    roles = [m.role for m in result]
    # If tool is kept, its assistant must also be kept
    if "tool" in roles:
      idx = roles.index("tool")
      assert idx > 0
      assert result[idx - 1].role == "assistant"


@pytest.mark.unit
class TestContextManagerCacheSplit:
  def test_split_disabled_returns_empty_static(self):
    mgr = ContextManager(Context(cache_optimization=False))
    static, dynamic = mgr.build_system_prompt_split(
      instructions="Core",
      knowledge_context="Docs",
    )
    assert static == ""
    assert "Core" in dynamic
    assert "Docs" in dynamic

  def test_split_enabled_separates_static_dynamic(self):
    mgr = ContextManager(Context(cache_optimization=True))
    static, dynamic = mgr.build_system_prompt_split(
      instructions="Core instructions",
      layer_guide="Capabilities",
      knowledge_context="RAG docs",
      memory_context="History",
    )
    # Static: instructions + layer guide (cacheable=True)
    assert "Core instructions" in static
    assert "Capabilities" in static
    # Dynamic: knowledge + memory (cacheable=False)
    assert "RAG docs" in dynamic
    assert "History" in dynamic


@pytest.mark.unit
class TestContextConfig:
  def test_default_values(self):
    ctx = Context()
    assert ctx.history_strategy == "tail"
    assert ctx.max_history_messages == 50
    assert ctx.keep_first_messages == 4
    assert ctx.token_budget is None
    assert ctx.cache_optimization is False
    assert ctx.tool_filter is None

  def test_token_budget_validation(self):
    with pytest.raises(ValueError, match="must be <= 1.0"):
      TokenBudget(system_prompt_pct=0.5, knowledge_pct=0.3, memory_pct=0.2, history_pct=0.5)

  def test_token_budget_valid(self):
    budget = TokenBudget(system_prompt_pct=0.3, knowledge_pct=0.15, memory_pct=0.1, history_pct=0.4)
    assert budget.output_reserve == 4096


@pytest.mark.unit
class TestAgentContextIntegration:
  def test_agent_context_none_no_manager(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent._context_manager is None

  def test_agent_context_true_creates_manager(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(model=MockModel(), context=True)  # type: ignore[arg-type]
    assert agent._context_manager is not None
    assert agent._context_manager.config.history_strategy == "tail"
    assert agent._context_manager.config.max_history_messages == 50

  def test_agent_context_custom_config(self):
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    ctx = Context(history_strategy="head_and_tail", max_history_messages=30, keep_first_messages=5)
    agent = Agent(model=MockModel(), context=ctx)  # type: ignore[arg-type]
    assert agent._context_manager is not None
    assert agent._context_manager.config.history_strategy == "head_and_tail"
    assert agent._context_manager.config.max_history_messages == 30

  def test_agent_without_context_backward_compat(self):
    """Agent without context= should work exactly as before."""
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    agent = Agent(model=MockModel(), instructions="Be helpful.")  # type: ignore[arg-type]
    assert agent._context_manager is None
    assert agent.instructions == "Be helpful."
