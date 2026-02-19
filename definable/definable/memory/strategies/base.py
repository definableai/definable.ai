"""Base class for memory optimization strategies."""

from abc import ABC, abstractmethod
from typing import Any, List

from definable.memory.types import UserMemory


class MemoryOptimizationStrategy(ABC):
  """Abstract base class for memory optimization strategies.

  Strategies take a list of memories and produce an optimized list,
  typically with fewer entries and lower total token count.
  """

  @abstractmethod
  async def aoptimize(self, memories: List[UserMemory], model: Any) -> List[UserMemory]:
    """Optimize a list of memories using the provided model.

    Args:
      memories: The current list of user memories.
      model: The LLM model to use for optimization.

    Returns:
      A new list of optimized UserMemory objects.
    """
    ...

  def count_tokens(self, memories: List[UserMemory]) -> int:
    """Rough token count estimate (4 chars per token heuristic)."""
    total_chars = sum(len(m.memory) for m in memories)
    return total_chars // 4
