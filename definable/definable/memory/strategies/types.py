"""Strategy type enum and factory."""

from enum import Enum

from definable.memory.strategies.base import MemoryOptimizationStrategy


class StrategyType(str, Enum):
  """Available memory optimization strategies."""

  SUMMARIZE = "summarize"


class StrategyFactory:
  """Factory for creating memory optimization strategies."""

  @classmethod
  def create(cls, strategy_type: StrategyType) -> MemoryOptimizationStrategy:
    """Create a strategy instance by type.

    Args:
      strategy_type: The type of strategy to create.

    Returns:
      A MemoryOptimizationStrategy instance.
    """
    if strategy_type == StrategyType.SUMMARIZE:
      from definable.memory.strategies.summarize import SummarizeStrategy

      return SummarizeStrategy()
    raise ValueError(f"Unknown strategy type: {strategy_type}")
