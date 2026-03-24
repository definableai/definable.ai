"""Token budget allocation and context window utilities.

Resolves ``TokenBudget`` percentages into absolute token counts
and provides a lookup table for well-known model context windows.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
  from definable.model.base import Model

# ── Context window lookup ─────────────────────────────────────
# Fallback when the model doesn't expose a ``context_window`` attribute.
# Values are the maximum input + output tokens for each model family.

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
  # OpenAI
  "gpt-4o": 128_000,
  "gpt-4o-mini": 128_000,
  "gpt-4.1": 1_047_576,
  "gpt-4.1-mini": 1_047_576,
  "gpt-4.1-nano": 1_047_576,
  "gpt-4-turbo": 128_000,
  "gpt-4": 8_192,
  "o1": 200_000,
  "o1-mini": 128_000,
  "o1-pro": 200_000,
  "o3": 200_000,
  "o3-mini": 200_000,
  "o4-mini": 200_000,
  # Anthropic
  "claude-opus-4-20250514": 200_000,
  "claude-sonnet-4-20250514": 200_000,
  "claude-sonnet-4-5-20250929": 200_000,
  "claude-3-5-sonnet-20241022": 200_000,
  "claude-3-5-haiku-20241022": 200_000,
  "claude-3-opus-20240229": 200_000,
  # Google
  "gemini-2.0-flash": 1_048_576,
  "gemini-2.0-flash-001": 1_048_576,
  "gemini-2.5-flash": 1_048_576,
  "gemini-2.5-pro": 1_048_576,
  "gemini-1.5-pro": 2_097_152,
  # DeepSeek
  "deepseek-chat": 64_000,
  "deepseek-reasoner": 64_000,
  # Mistral
  "mistral-large-latest": 128_000,
  "mistral-small-latest": 128_000,
  # xAI
  "grok-3": 131_072,
  "grok-3-mini": 131_072,
}


def get_context_window(model: "Model") -> Optional[int]:
  """Return the context window size for a model.

  Checks ``model.context_window`` first (explicit override),
  then falls back to the built-in lookup table.

  Args:
    model: A Model instance.

  Returns:
    Context window token count, or None if unknown.
  """
  # Explicit attribute takes priority
  explicit = getattr(model, "context_window", None)
  if explicit is not None:
    return int(explicit)

  # Fallback: exact ID match
  model_id = getattr(model, "id", "")
  if model_id in MODEL_CONTEXT_WINDOWS:
    return MODEL_CONTEXT_WINDOWS[model_id]

  # Fallback: prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
  for prefix, window in sorted(MODEL_CONTEXT_WINDOWS.items(), key=lambda kv: -len(kv[0])):
    if model_id.startswith(prefix):
      return window

  return None


@dataclass
class TokenAllocation:
  """Resolved token counts from budget percentages.

  Attributes:
    system_prompt: Tokens allocated for the system prompt.
    knowledge: Tokens allocated for knowledge context.
    memory: Tokens allocated for memory context.
    history: Tokens allocated for conversation history.
    output_reserve: Tokens reserved for model output.
    total_available: Total tokens available (context_window - output_reserve).
    context_window: The model's full context window.
  """

  system_prompt: int
  knowledge: int
  memory: int
  history: int
  output_reserve: int
  total_available: int
  context_window: int


def allocate_budget(
  context_window: int,
  output_reserve: int = 4096,
  system_prompt_pct: float = 0.30,
  knowledge_pct: float = 0.15,
  memory_pct: float = 0.10,
  history_pct: float = 0.40,
) -> TokenAllocation:
  """Convert budget percentages into absolute token counts.

  Args:
    context_window: Total context window size.
    output_reserve: Tokens reserved for the model's response.
    system_prompt_pct: Fraction for system prompt.
    knowledge_pct: Fraction for knowledge context.
    memory_pct: Fraction for memory context.
    history_pct: Fraction for conversation history.

  Returns:
    TokenAllocation with resolved counts.
  """
  available = max(0, context_window - output_reserve)

  return TokenAllocation(
    system_prompt=int(available * system_prompt_pct),
    knowledge=int(available * knowledge_pct),
    memory=int(available * memory_pct),
    history=int(available * history_pct),
    output_reserve=output_reserve,
    total_available=available,
    context_window=context_window,
  )


def resolve_allocation(
  budget: Any,
  model: Optional["Model"] = None,
) -> Optional[TokenAllocation]:
  """Resolve a TokenBudget into a TokenAllocation.

  Uses the budget's explicit context_window if set, otherwise
  auto-detects from the model. Returns None if the context
  window cannot be determined.

  Args:
    budget: A TokenBudget instance.
    model: Optional model for context window auto-detection.

  Returns:
    TokenAllocation, or None if context window is unknown.
  """
  if not hasattr(budget, "output_reserve"):
    return None

  context_window: Optional[int] = budget.context_window
  if context_window is None and model is not None:
    context_window = get_context_window(model)

  if context_window is None:
    return None

  return allocate_budget(
    context_window=context_window,
    output_reserve=budget.output_reserve,
    system_prompt_pct=budget.system_prompt_pct,
    knowledge_pct=budget.knowledge_pct,
    memory_pct=budget.memory_pct,
    history_pct=budget.history_pct,
  )
