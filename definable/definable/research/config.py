"""Configuration for the deep research pipeline."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional

if TYPE_CHECKING:
  from definable.models.base import Model


@dataclass
class DeepResearchConfig:
  """Configuration for deep research.

  Controls how the research pipeline operates: search backend,
  compression model, depth, concurrency, and output format.

  Attributes:
    enabled: Whether deep research is active (always True when instantiated).
    depth: Research depth preset — "quick", "standard", or "deep".
    search_provider: Search backend name ("duckduckgo", "google", "serpapi").
    search_provider_config: Backend-specific config (API keys, CSE ID, etc.).
    search_fn: Custom async search callable. Overrides search_provider.
    compression_model: Model for CKU extraction (cheap/fast). If None, uses agent's model.
    max_sources: Maximum number of unique sources to read across all waves.
    max_waves: Maximum number of research waves.
    parallel_searches: Number of concurrent search queries per wave.
    parallel_reads: Number of concurrent page reads.
    min_relevance: Minimum relevance score for CKU inclusion.
    include_citations: Whether to include source citations in context.
    include_contradictions: Whether to surface contradictions.
    context_format: Format for the injected context ("xml" or "markdown").
    max_context_tokens: Approximate token budget for the context block.
    early_termination_threshold: Stop when novelty ratio drops below this.
    trigger: When to run research ("always", "auto", "tool").
  """

  enabled: bool = True
  depth: Literal["quick", "standard", "deep"] = "standard"
  search_provider: str = "duckduckgo"
  search_provider_config: Optional[Dict[str, Any]] = field(default=None, hash=False)
  search_fn: Optional[Callable] = field(default=None, hash=False)
  compression_model: Optional["Model"] = field(default=None, hash=False)
  max_sources: int = 15
  max_waves: int = 3
  parallel_searches: int = 5
  parallel_reads: int = 10
  min_relevance: float = 0.3
  include_citations: bool = True
  include_contradictions: bool = True
  context_format: Literal["xml", "markdown"] = "xml"
  max_context_tokens: int = 4000
  early_termination_threshold: float = 0.2
  trigger: Literal["auto", "always", "tool"] = "always"

  def with_depth_preset(self) -> "DeepResearchConfig":
    """Apply depth preset defaults (only overrides unset values)."""
    presets = {
      "quick": {"max_sources": 8, "max_waves": 1, "parallel_searches": 3},
      "standard": {"max_sources": 15, "max_waves": 3, "parallel_searches": 5},
      "deep": {"max_sources": 30, "max_waves": 5, "parallel_searches": 8},
    }
    preset = presets.get(self.depth, presets["standard"])
    # Only apply preset values if user hasn't customized them
    import dataclasses

    defaults = DeepResearchConfig()
    updates = {}
    for key, value in preset.items():
      if getattr(self, key) == getattr(defaults, key):
        updates[key] = value
    if updates:
      return dataclasses.replace(self, **updates)  # type: ignore[arg-type]
    return self
