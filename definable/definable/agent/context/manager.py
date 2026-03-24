"""ContextManager — orchestrates layered prompt building + history trimming.

Called from ``Agent._build_invoke_messages()`` when ``context`` is configured.
Replaces the naive concatenation with priority-aware, budget-aware assembly.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from definable.agent.context.budget import TokenAllocation, resolve_allocation
from definable.agent.context.history import HistoryTrimmer
from definable.agent.context.layers import (
  PRIORITY_EPHEMERAL,
  PRIORITY_INSTRUCTIONS,
  PRIORITY_KNOWLEDGE,
  PRIORITY_LAYER_GUIDE,
  PRIORITY_MEMORY,
  LayeredPrompt,
  PromptLayer,
)
from definable.tokens import count_text_tokens

if TYPE_CHECKING:
  from definable.agent.context import Context
  from definable.model.base import Model
  from definable.model.message import Message


class ContextManager:
  """Orchestrates system prompt assembly and history trimming.

  Ties together:
  - ``LayeredPrompt`` for structured, priority-based system prompt.
  - ``HistoryTrimmer`` for message history management.
  - (Phase 2) Token budgets for per-section limits.
  - (Phase 2) Cache-aware splitting for prompt caching.

  Example:
    from definable.agent.context import Context
    from definable.agent.context.manager import ContextManager

    mgr = ContextManager(Context(), model=model)
    system = mgr.build_system_prompt(
      instructions="You are helpful.",
      skill_instructions="",
      layer_guide="",
      knowledge_context="Doc A...",
    )
    trimmed = mgr.trim_history(messages)
  """

  def __init__(self, config: "Context", model: Optional["Model"] = None) -> None:
    self._config = config
    self._model = model
    self._model_id = model.id if model else "gpt-4o"
    self._trimmer = HistoryTrimmer(
      strategy=config.history_strategy,
      max_messages=config.max_history_messages,
      keep_first=config.keep_first_messages,
      model=model,
    )
    self._last_stats: Optional[Dict] = None
    # Resolve token allocation once (None if no budget or unknown window)
    self._allocation: Optional[TokenAllocation] = resolve_allocation(config.token_budget, model) if config.token_budget else None

  @property
  def config(self) -> "Context":
    return self._config

  @property
  def last_stats(self) -> Optional[Dict]:
    """Stats from the most recent build_system_prompt call."""
    return self._last_stats

  def build_system_prompt(
    self,
    *,
    instructions: Optional[str] = None,
    skill_instructions: Optional[str] = None,
    layer_guide: Optional[str] = None,
    thinking_injection: Optional[str] = None,
    knowledge_context: Optional[str] = None,
    research_context: Optional[str] = None,
    memory_context: Optional[str] = None,
  ) -> str:
    """Assemble the system prompt from priority-ordered layers.

    Args:
      instructions: Agent's base instructions.
      skill_instructions: Merged instructions from attached skills.
      layer_guide: Capabilities menu section.
      thinking_injection: Pre-computed thinking / execution strategy.
      knowledge_context: Retrieved RAG documents.
      research_context: Deep research findings.
      memory_context: Conversation history / memory recall.

    Returns:
      The assembled system prompt string.
    """
    prompt = LayeredPrompt(model_id=self._model_id)

    # Priority 1: Core instructions (sacred — never dropped)
    core = _merge_strings(instructions, skill_instructions)
    if core:
      prompt.add(PromptLayer(name="instructions", content=core, priority=PRIORITY_INSTRUCTIONS, cacheable=True))

    # Priority 2: Layer guide (rarely dropped)
    if layer_guide:
      prompt.add(PromptLayer(name="layer_guide", content=layer_guide, priority=PRIORITY_LAYER_GUIDE, cacheable=True))

    # Priority 3: Knowledge context — trim to budget if allocated
    if knowledge_context:
      knowledge_context = self._trim_section_to_budget(knowledge_context, "knowledge")
      if knowledge_context:
        prompt.add(PromptLayer(name="knowledge", content=knowledge_context, priority=PRIORITY_KNOWLEDGE, cacheable=False))

    # Priority 4: Memory context — trim to budget if allocated
    if memory_context:
      memory_context = self._trim_section_to_budget(memory_context, "memory")
      if memory_context:
        prompt.add(PromptLayer(name="memory", content=memory_context, priority=PRIORITY_MEMORY, cacheable=False))

    # Priority 5: Ephemeral (thinking, research — dropped first)
    if thinking_injection:
      prompt.add(PromptLayer(name="thinking", content=thinking_injection, priority=PRIORITY_EPHEMERAL, cacheable=False))
    if research_context:
      prompt.add(PromptLayer(name="research", content=research_context, priority=PRIORITY_EPHEMERAL, cacheable=False))

    # Build — apply total system prompt budget if allocated
    max_tokens = self._allocation.system_prompt if self._allocation else None
    result = prompt.build(max_tokens=max_tokens)

    # Record stats for observability
    self._last_stats = prompt.token_stats()
    if self._allocation:
      self._last_stats["budget"] = {
        "system_prompt": self._allocation.system_prompt,
        "knowledge": self._allocation.knowledge,
        "memory": self._allocation.memory,
        "history": self._allocation.history,
        "context_window": self._allocation.context_window,
      }

    return result

  def build_system_prompt_split(
    self,
    **kwargs: Optional[str],
  ) -> tuple:
    """Build system prompt and return (static_prefix, dynamic_suffix).

    For cache optimization: static part can be cached, dynamic part changes per turn.
    Falls back to ("", full_prompt) if cache_optimization is disabled.
    """
    if not self._config.cache_optimization:
      full = self.build_system_prompt(**kwargs)
      return "", full

    # Build with layers, then split by cacheable flag
    prompt = self._build_layered_prompt(**kwargs)
    return prompt.build_split()

  @property
  def allocation(self) -> Optional[TokenAllocation]:
    """The resolved token allocation, or None if no budget is configured."""
    return self._allocation

  def trim_history(self, messages: List["Message"]) -> List["Message"]:
    """Trim message history using the configured strategy (sync).

    For the ``summarize`` strategy, use ``atrim_history()`` instead.

    Args:
      messages: All conversation messages (may include system).

    Returns:
      Trimmed messages (system messages excluded).
    """
    non_system = [m for m in messages if m.role != "system"]
    trimmed = self._trimmer.trim(non_system)

    if self._allocation and trimmed:
      trimmed = self._trim_history_to_token_budget(trimmed, self._allocation.history)

    return trimmed

  async def atrim_history(self, messages: List["Message"]) -> List["Message"]:
    """Trim message history using the configured strategy (async).

    For the ``summarize`` strategy, dropped messages are summarized
    by an LLM and the summary is injected as the first message.

    Args:
      messages: All conversation messages (may include system).

    Returns:
      Trimmed messages (system messages excluded, summary injected).
    """
    non_system = [m for m in messages if m.role != "system"]
    trimmed = await self._trimmer.atrim(non_system)

    if self._allocation and trimmed:
      trimmed = self._trim_history_to_token_budget(trimmed, self._allocation.history)

    return trimmed

  def _trim_section_to_budget(self, content: str, section: str) -> str:
    """Trim a context section to its token budget.

    Args:
      content: The section content.
      section: Budget key ("knowledge" or "memory").

    Returns:
      Trimmed content, or original if no budget is active.
    """
    if not self._allocation:
      return content

    budget_tokens = getattr(self._allocation, section, None)
    if budget_tokens is None:
      return content

    actual_tokens = count_text_tokens(content, self._model_id)
    if actual_tokens <= budget_tokens:
      return content

    # Truncate from the end — binary search for the right length
    low, high = 0, len(content)
    while low < high:
      mid = (low + high + 1) // 2
      if count_text_tokens(content[:mid], self._model_id) <= budget_tokens:
        low = mid
      else:
        high = mid - 1

    return content[:low]

  def _trim_history_to_token_budget(self, messages: List["Message"], max_tokens: int) -> List["Message"]:
    """Drop oldest messages until total history tokens fit the budget.

    Respects tool-call pairs by operating on message groups.
    """
    from definable.agent.context.history import flatten_groups, group_messages
    from definable.tokens import count_tokens

    total = count_tokens(messages, model_id=self._model_id)
    if total <= max_tokens:
      return messages

    groups = group_messages(messages)
    # Drop groups from the front (oldest) until we fit
    while groups and count_tokens(flatten_groups(groups), model_id=self._model_id) > max_tokens:
      groups.pop(0)

    return flatten_groups(groups) if groups else messages[-1:]

  def _build_layered_prompt(self, **kwargs: Optional[str]) -> LayeredPrompt:
    """Build a LayeredPrompt from keyword arguments (for split builds)."""
    prompt = LayeredPrompt(model_id=self._model_id)

    core = _merge_strings(kwargs.get("instructions"), kwargs.get("skill_instructions"))
    if core:
      prompt.add(PromptLayer(name="instructions", content=core, priority=PRIORITY_INSTRUCTIONS, cacheable=True))
    layer_guide = kwargs.get("layer_guide")
    if layer_guide:
      prompt.add(PromptLayer(name="layer_guide", content=layer_guide, priority=PRIORITY_LAYER_GUIDE, cacheable=True))
    knowledge_ctx = kwargs.get("knowledge_context")
    if knowledge_ctx:
      prompt.add(PromptLayer(name="knowledge", content=knowledge_ctx, priority=PRIORITY_KNOWLEDGE, cacheable=False))
    memory_ctx = kwargs.get("memory_context")
    if memory_ctx:
      prompt.add(PromptLayer(name="memory", content=memory_ctx, priority=PRIORITY_MEMORY, cacheable=False))
    thinking_inj = kwargs.get("thinking_injection")
    if thinking_inj:
      prompt.add(PromptLayer(name="thinking", content=thinking_inj, priority=PRIORITY_EPHEMERAL, cacheable=False))
    research_ctx = kwargs.get("research_context")
    if research_ctx:
      prompt.add(PromptLayer(name="research", content=research_ctx, priority=PRIORITY_EPHEMERAL, cacheable=False))

    return prompt


def _merge_strings(*parts: Optional[str]) -> str:
  """Join non-empty strings with double newlines."""
  non_empty = [p for p in parts if p and p.strip()]
  return "\n\n".join(non_empty)
