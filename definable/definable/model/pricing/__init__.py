"""Model spec sheets — pricing, capabilities, modalities and limits per model.

Each provider has a spec-sheet JSON in this package (``openai.json``,
``anthropic.json``, ...). A file looks like::

    {
      "provider": "OpenAI",
      "models": {
        "gpt-4o": {
          "pricing": {"input_per_million": 2.5, "output_per_million": 10.0},
          "capabilities": {"tool_use": true, "vision": true},
          "modalities": {"input": ["text", "image"], "output": ["text"]},
          "limits": {"context_window": 128000, "max_output_tokens": 16384}
        }
      }
    }

The registry loads every ``*.json`` in this directory, keyed by the lowercased
provider name (matching ``Model.provider``). ``pricing`` maps 1:1 to
:class:`ModelPricing`; the rest is exposed through :class:`ModelSpec`.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from definable.model.metrics import Metrics


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


@dataclass
class ModelSpec:
  """Full spec sheet for a model: pricing plus capabilities, modalities, limits.

  ``capabilities`` (e.g. ``tool_use``, ``vision``, ``structured_output``,
  ``reasoning``, ``prompt_caching``), ``modalities`` (``input``/``output``
  lists like ``text``/``image``/``audio``/``video``/``pdf``) and ``limits``
  (``context_window``, ``max_output_tokens``) are free-form so the spec sheets
  can grow without code changes.
  """

  model_id: str
  provider: str
  pricing: ModelPricing = field(default_factory=ModelPricing)
  capabilities: dict[str, Any] = field(default_factory=dict)
  modalities: dict[str, list[str]] = field(default_factory=dict)
  limits: dict[str, Any] = field(default_factory=dict)
  knowledge_cutoff: Optional[str] = None
  release_date: Optional[str] = None

  def supports(self, capability: str) -> bool:
    """True if the model advertises ``capability`` (e.g. ``"tool_use"``)."""
    return bool(self.capabilities.get(capability, False))

  def accepts(self, modality: str) -> bool:
    """True if ``modality`` (e.g. ``"image"``) is an accepted input."""
    return modality in self.modalities.get("input", [])


class PricingRegistry:
  """Singleton registry over the per-provider spec-sheet JSON files."""

  _instance: Optional["PricingRegistry"] = None
  _spec_data: dict[str, dict[str, Any]]

  def __new__(cls) -> "PricingRegistry":
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._spec_data = {}
      cls._instance._load()
    return cls._instance

  def _load(self) -> None:
    """Load every ``<provider>.json`` spec sheet in this package directory."""
    self._spec_data = {}
    for path in Path(__file__).parent.glob("*.json"):
      with open(path, encoding="utf-8") as f:
        data = json.load(f)
      provider_key = str(data.get("provider", path.stem)).lower()
      self._spec_data[provider_key] = data.get("models", {})

  def _match(self, provider: str, model_id: str) -> Optional[dict[str, Any]]:
    """Find a model entry by exact id, then by prefix in either direction."""
    models = self._spec_data.get(provider.lower())
    if not models:
      return None
    if model_id in models:
      return models[model_id]
    # Prefix match for versioned ids (e.g. "gpt-4o-2024-08-06" -> "gpt-4o")
    for key, entry in models.items():
      if model_id.startswith(key) or key.startswith(model_id):
        return entry
    return None

  def get_pricing(self, provider: str, model_id: str) -> Optional[ModelPricing]:
    """Get pricing for a provider/model, or None if unknown."""
    entry = self._match(provider, model_id)
    if entry is None:
      return None
    return ModelPricing(**entry.get("pricing", {}))

  def get_spec(self, provider: str, model_id: str) -> Optional[ModelSpec]:
    """Get the full spec sheet for a provider/model, or None if unknown."""
    entry = self._match(provider, model_id)
    if entry is None:
      return None
    return ModelSpec(
      model_id=model_id,
      provider=provider,
      pricing=ModelPricing(**entry.get("pricing", {})),
      capabilities=entry.get("capabilities", {}),
      modalities=entry.get("modalities", {}),
      limits=entry.get("limits", {}),
      knowledge_cutoff=entry.get("knowledge_cutoff"),
      release_date=entry.get("release_date"),
    )

  def reload(self) -> None:
    """Force reload all spec sheets from disk."""
    self._load()


def get_pricing(provider: str, model_id: str) -> Optional[ModelPricing]:
  """Convenience accessor for a model's pricing (see :meth:`PricingRegistry.get_pricing`)."""
  return PricingRegistry().get_pricing(provider, model_id)


def get_spec(provider: str, model_id: str) -> Optional[ModelSpec]:
  """Convenience accessor for a model's full spec sheet (see :meth:`PricingRegistry.get_spec`)."""
  return PricingRegistry().get_spec(provider, model_id)


def calculate_cost(provider: str, model_id: str, metrics: Metrics) -> Optional[float]:
  """Calculate cost for the given provider/model/usage, or None if pricing is unknown."""
  pricing = get_pricing(provider, model_id)
  if pricing is None:
    return None
  return pricing.calculate_cost(metrics)
