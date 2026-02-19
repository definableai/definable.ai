"""Base class for memory optimization strategies."""

from abc import ABC, abstractmethod
from typing import Any, List

from definable.memory.types import MemoryEntry


class MemoryStrategy(ABC):
  """Abstract base class for memory optimization strategies.

  Strategies take a list of entries and produce an optimized list,
  typically with fewer entries via summarization.
  """

  @abstractmethod
  async def optimize(self, entries: List[MemoryEntry], model: Any) -> List[MemoryEntry]:
    """Optimize a list of memory entries using the provided model.

    Args:
      entries: The current list of session entries.
      model: The LLM model to use for optimization.

    Returns:
      A new list of optimized MemoryEntry objects.
    """
    ...

  def count_tokens(self, entries: List[MemoryEntry]) -> int:
    """Rough token count estimate (4 chars per token heuristic)."""
    total_chars = sum(len(e.content) for e in entries)
    return total_chars // 4
