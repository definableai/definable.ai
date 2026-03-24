"""Context management configuration for agents.

Controls system prompt layering, message history trimming,
and token budget allocation. Opt-in via ``Agent(context=True)``
or ``Agent(context=Context(...))``.

Example:
  from definable.agent import Agent
  from definable.agent.context import Context

  # Sensible defaults (tail trimming at 50 messages)
  agent = Agent(model="openai/gpt-4o", context=True)

  # Custom
  agent = Agent(
    model="openai/gpt-4o",
    context=Context(
      history_strategy="head_and_tail",
      max_history_messages=40,
      keep_first_messages=4,
    ),
  )
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

if TYPE_CHECKING:
  from definable.model.base import Model


@dataclass
class TokenBudget:
  """Token allocation across prompt sections.

  Divides the model's context window into zones. Each section
  is trimmed to its budget if it exceeds the allocation.

  Attributes:
    context_window: Model context window size. Auto-detected from model if None.
    output_reserve: Tokens reserved for the model's response.
    system_prompt_pct: Fraction of available tokens for the system prompt.
    knowledge_pct: Fraction for knowledge (RAG) context.
    memory_pct: Fraction for memory context.
    history_pct: Fraction for conversation history.
  """

  context_window: Optional[int] = None
  output_reserve: int = 4096
  system_prompt_pct: float = 0.30
  knowledge_pct: float = 0.15
  memory_pct: float = 0.10
  history_pct: float = 0.40

  def __post_init__(self) -> None:
    total = self.system_prompt_pct + self.knowledge_pct + self.memory_pct + self.history_pct
    if total > 1.0:
      raise ValueError(f"Token budget percentages sum to {total:.2f}, must be <= 1.0")


@dataclass
class Context:
  """User-facing context management configuration.

  Controls system prompt layering, message history trimming,
  and token budget allocation.

  Attributes:
    history_strategy: How to handle growing message history.
      - "none": No trimming (default behavior).
      - "tail": Keep the most recent max_history_messages.
      - "head_and_tail": Keep first keep_first_messages + last max_history_messages.
      - "summarize": Like tail, but summarize dropped messages first (Phase 3).
    max_history_messages: Maximum messages to keep (tail/head_and_tail).
    keep_first_messages: Messages to keep from the start (head_and_tail only).
    summarize_model: Model for summarizing dropped messages. Uses agent model if None.
    token_budget: Optional token budget allocation across sections.
    cache_optimization: Split system prompt into static+dynamic for prompt caching.
    deferred_tools: When True, tool schemas are NOT sent upfront. Instead, a compact
      catalog (names + descriptions) is injected into the system prompt, and a built-in
      ``load_tools`` tool lets the model request full schemas on demand.
    tool_filter: Optional function to filter tool schemas per call (ignored when
      deferred_tools is True).

  Example:
    from definable.agent import Agent
    from definable.agent.context import Context

    # Defaults
    agent = Agent(model="openai/gpt-4o", context=True)

    # Custom
    agent = Agent(
      model="openai/gpt-4o",
      context=Context(
        history_strategy="head_and_tail",
        max_history_messages=40,
        keep_first_messages=4,
      ),
    )
  """

  history_strategy: Literal["none", "tail", "head_and_tail", "summarize"] = "tail"
  max_history_messages: Optional[int] = 50
  keep_first_messages: int = 4
  summarize_model: Optional[Union[str, "Model"]] = None
  token_budget: Optional[TokenBudget] = None
  cache_optimization: bool = False
  deferred_tools: bool = False
  tool_filter: Optional[Callable[..., bool]] = field(default=None, repr=False)


__all__ = ["Context", "TokenBudget"]
