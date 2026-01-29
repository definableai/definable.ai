"""Model pricing calculation module for computing API costs based on token usage."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

from definable.models.metrics import Metrics


@dataclass
class ModelPricing:
  """Pricing configuration for a specific model (rates per million tokens)."""
  input_per_million: float = 0.0
  output_per_million: float = 0.0
  cached_input_per_million: Optional[float] = None
  cache_write_per_million: Optional[float] = None
  audio_input_per_million: Optional[float] = None
  audio_output_per_million: Optional[float] = None
  reasoning_per_million: Optional[float] = None

  def calculate_cost(self, metrics: Metrics) -> float:
    """
    Calculate cost based on token usage.

    Args:
      metrics: Metrics object with token counts

    Returns:
      Calculated cost in USD
    """
    cost = 0.0

    # Input tokens (non-cached)
    non_cached_input = max(0, metrics.input_tokens - metrics.cache_read_tokens)
    cost += (non_cached_input / 1_000_000) * self.input_per_million

    # Cached input tokens (default to 50% of input rate if not specified)
    cached_rate = self.cached_input_per_million if self.cached_input_per_million is not None else self.input_per_million * 0.5
    cost += (metrics.cache_read_tokens / 1_000_000) * cached_rate

    # Cache write tokens
    if metrics.cache_write_tokens > 0 and self.cache_write_per_million is not None:
      cost += (metrics.cache_write_tokens / 1_000_000) * self.cache_write_per_million

    # Output tokens
    cost += (metrics.output_tokens / 1_000_000) * self.output_per_million

    # Audio input tokens
    if metrics.audio_input_tokens > 0:
      audio_rate = self.audio_input_per_million if self.audio_input_per_million is not None else self.input_per_million
      cost += (metrics.audio_input_tokens / 1_000_000) * audio_rate

    # Audio output tokens
    if metrics.audio_output_tokens > 0:
      audio_rate = self.audio_output_per_million if self.audio_output_per_million is not None else self.output_per_million
      cost += (metrics.audio_output_tokens / 1_000_000) * audio_rate

    # Reasoning tokens (models like o1)
    if metrics.reasoning_tokens > 0:
      reasoning_rate = self.reasoning_per_million if self.reasoning_per_million is not None else self.output_per_million
      cost += (metrics.reasoning_tokens / 1_000_000) * reasoning_rate

    return round(cost, 8)


class PricingRegistry:
  """Singleton registry for model pricing configurations."""
  _instance: Optional["PricingRegistry"] = None
  _pricing_data: dict[str, Any]

  def __new__(cls) -> "PricingRegistry":
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._pricing_data = {}
      cls._instance._load_pricing()
    return cls._instance

  def _load_pricing(self) -> None:
    """Load pricing config from model_pricing.json."""
    search_paths = [
      Path(__file__).parent.parent.parent / "model_pricing.json",
      Path.cwd() / "model_pricing.json",
    ]

    for path in search_paths:
      if path.exists():
        with open(path, encoding="utf-8") as f:
          self._pricing_data = json.load(f)
          return

  def get_pricing(self, provider: str, model_id: str) -> Optional[ModelPricing]:
    """
    Get pricing for a provider/model combination.

    Args:
      provider: Provider name (e.g., "OpenAI", "DeepSeek")
      model_id: Model identifier (e.g., "gpt-4o", "deepseek-chat")

    Returns:
      ModelPricing instance or None if not found
    """
    provider_key = provider.lower()
    provider_pricing = self._pricing_data.get(provider_key)
    if not provider_pricing:
      return None

    # Try exact match first
    if model_id in provider_pricing:
      return ModelPricing(**provider_pricing[model_id])

    # Try prefix matching for versioned models (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
    for key in provider_pricing:
      if model_id.startswith(key) or key.startswith(model_id):
        return ModelPricing(**provider_pricing[key])

    return None

  def reload(self) -> None:
    """Force reload pricing configuration from disk."""
    self._pricing_data = {}
    self._load_pricing()


def get_pricing(provider: str, model_id: str) -> Optional[ModelPricing]:
  """
  Convenience function to get pricing for a model.

  Args:
    provider: Provider name (e.g., "OpenAI", "DeepSeek")
    model_id: Model identifier (e.g., "gpt-4o", "deepseek-chat")

  Returns:
    ModelPricing instance or None if not found
  """
  return PricingRegistry().get_pricing(provider, model_id)


def calculate_cost(provider: str, model_id: str, metrics: Metrics) -> Optional[float]:
  """
  Calculate cost for the given provider, model, and usage metrics.

  Args:
    provider: Provider name (e.g., "OpenAI", "DeepSeek")
    model_id: Model identifier (e.g., "gpt-4o", "deepseek-chat")
    metrics: Metrics object with token counts

  Returns:
    Calculated cost in USD, or None if pricing not available
  """
  pricing = get_pricing(provider, model_id)
  if pricing is None:
    return None
  return pricing.calculate_cost(metrics)
