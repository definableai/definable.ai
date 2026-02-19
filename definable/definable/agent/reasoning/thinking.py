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


@dataclass
class Thinking:
  """Configuration for the agent's thinking/reasoning layer.

  When enabled, the agent performs a separate reasoning step before
  producing the final answer, improving response quality for complex queries.

  Attributes:
    enabled: Whether thinking is active.
    model: Model to use for thinking. If None, uses the agent's model.
    instructions: Custom thinking prompt. If None, uses the default.
    trigger: When to activate thinking. "always" runs every call; "auto" does
      a lightweight model pre-check; "never" disables even if configured.
    description: Description shown in the layer guide (system prompt).
  """

  enabled: bool = True
  model: Optional["Model"] = None
  instructions: Optional[str] = None
  trigger: Literal["always", "auto", "never"] = "always"
  description: Optional[str] = None
