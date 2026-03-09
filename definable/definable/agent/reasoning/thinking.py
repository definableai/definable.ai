"""Standalone Thinking block — a composable lego piece for agent reasoning.

Usage:
    from definable.agent.reasoning.thinking import Thinking
    from definable.agent import Agent

    agent = Agent(model=model, thinking=Thinking(trigger="auto"))
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
  from definable.model.base import Model

# Effort → budget_tokens mapping for native thinking
EFFORT_BUDGET_MAP: dict[str, int] = {
  "low": 4096,
  "medium": 10000,
  "high": 32000,
}


@dataclass
class Thinking:
  """Configuration for the agent's thinking/reasoning layer.

  When enabled, the agent performs reasoning before producing the final answer.
  If the model supports native thinking (e.g. Claude, DeepSeek), it uses the
  model's built-in extended thinking. Otherwise, it falls back to Definable's
  own thinking layer (a separate LLM call).

  Attributes:
    enabled: Whether thinking is active.
    model: Model to use for fallback thinking. If None, uses the agent's model.
        Only used when mode="definable" or auto-detected fallback.
    instructions: Custom thinking prompt for fallback mode. If None, uses default.
        Only used when mode="definable" or auto-detected fallback.
    trigger: When to activate thinking. "always" runs every call; "auto" does
      a lightweight model pre-check; "never" disables even if configured.
    effort: Depth of reasoning. "low" for quick decisions, "medium" (default)
      for standard analysis, "high" for thorough multi-step reasoning with
      considerations of edge cases and trade-offs. For native thinking, this
      maps to budget_tokens (low=4096, medium=10000, high=32000).
    budget_tokens: Explicit token budget for native thinking. Overrides the
        effort-based default when set. Ignored for fallback thinking.
    mode: Force a specific thinking path. "native" forces native thinking
        (raises ValueError if model doesn't support it). "definable" forces
        the fallback layer. None (default) auto-detects based on model
        capabilities.
    description: Description shown in the layer guide (system prompt).
  """

  enabled: bool = True
  model: Optional["Model"] = None
  instructions: Optional[str] = None
  trigger: Literal["always", "auto", "never"] = "always"
  effort: Literal["low", "medium", "high"] = "medium"
  budget_tokens: Optional[int] = None
  mode: Optional[Literal["native", "definable"]] = None
  description: Optional[str] = None

  def resolve_budget_tokens(self) -> int:
    """Return the token budget for native thinking."""
    if self.budget_tokens is not None:
      return self.budget_tokens
    return EFFORT_BUDGET_MAP[self.effort]

  def should_use_native(self, model: "Model") -> bool:
    """Determine whether to use native thinking for the given model.

    Args:
      model: The model to check for native thinking support.

    Returns:
      True if native thinking should be used, False for fallback.

    Raises:
      ValueError: If mode="native" but model doesn't support it.
    """
    if self.mode == "native":
      if not model.supports_native_thinking:
        raise ValueError(
          f"Model '{model.id}' does not support native thinking, "
          f"but thinking mode='native' was requested. "
          f"Use mode=None (auto-detect) or mode='definable' instead."
        )
      return True
    if self.mode == "definable":
      return False
    # Auto-detect: use native if the model supports it
    return model.supports_native_thinking
