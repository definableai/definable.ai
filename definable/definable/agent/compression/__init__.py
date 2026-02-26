from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from definable.agent.compression.manager import CompressionManager

if TYPE_CHECKING:
  from definable.model.base import Model


@dataclass
class Compression:
  """User-facing compression configuration.

  Controls automatic compression of tool results to save context space
  while preserving critical information during agent execution.

  Attributes:
    model: Model instance or string shorthand for compression.
           If None, uses the agent's model.
    tool_results_limit: Compress after N uncompressed tool results.
    token_limit: Compress when context exceeds this token threshold.
    instructions: Custom compression prompt/instructions.

  Example:
    from definable.agent import Agent
    from definable.agent.compression import Compression

    # Defaults
    agent = Agent(model="openai/gpt-4o", compression=True)

    # Custom
    agent = Agent(
      model="openai/gpt-4o",
      compression=Compression(tool_results_limit=5, token_limit=80_000),
    )
  """

  model: Optional[Union[str, "Model"]] = None
  tool_results_limit: Optional[int] = 3
  token_limit: Optional[int] = None
  instructions: Optional[str] = None


__all__ = ["Compression", "CompressionManager"]
