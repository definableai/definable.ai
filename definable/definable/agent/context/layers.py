"""Layered system prompt with priority-based trimming.

Each section of the system prompt is wrapped in a PromptLayer with a
priority (1 = never drop, 5 = drop first).  When the total exceeds a
token budget, layers are dropped from the lowest priority upward.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from definable.tokens import count_text_tokens

# Priority constants — used by ContextManager to tag each section.
PRIORITY_INSTRUCTIONS = 1  # Core instructions + skills (sacred)
PRIORITY_LAYER_GUIDE = 2  # Capabilities menu
PRIORITY_KNOWLEDGE = 3  # RAG context
PRIORITY_MEMORY = 4  # Conversation memory
PRIORITY_EPHEMERAL = 5  # Thinking output, research (expendable)


@dataclass
class PromptLayer:
  """A single section of the system prompt.

  Attributes:
    name: Human-readable name (e.g. "instructions", "knowledge").
    content: The text content of this layer.
    priority: 1 = never drop, 5 = drop first.
    cacheable: True if this layer is static across turns (for cache opt).
  """

  name: str
  content: str
  priority: int = PRIORITY_EPHEMERAL
  cacheable: bool = True


@dataclass
class LayeredPrompt:
  """Assembles a system prompt from priority-ordered layers.

  Layers are added via ``add()``, then combined via ``build()``.
  If a token budget is provided, lower-priority layers are dropped
  (highest priority number first) until the total fits.

  Example:
    prompt = LayeredPrompt()
    prompt.add(PromptLayer(name="instructions", content="You are helpful.", priority=1))
    prompt.add(PromptLayer(name="knowledge", content="Doc A ...", priority=3))
    text = prompt.build()  # "You are helpful.\\n\\nDoc A ..."
  """

  _layers: List[PromptLayer] = field(default_factory=list, init=False, repr=False)
  model_id: str = "gpt-4o"

  def add(self, layer: PromptLayer) -> None:
    """Add a layer. Layers with empty content are silently ignored."""
    if layer.content and layer.content.strip():
      self._layers.append(layer)

  def clear(self) -> None:
    """Remove all layers."""
    self._layers.clear()

  @property
  def layers(self) -> List[PromptLayer]:
    """Return layers sorted by priority (lowest number = highest priority)."""
    return sorted(self._layers, key=lambda l: l.priority)

  def build(self, max_tokens: Optional[int] = None) -> str:
    """Combine layers into a single system prompt string.

    Args:
      max_tokens: If set, trim lower-priority layers to fit this budget.
        Priority 1 layers are never dropped.

    Returns:
      The assembled system prompt.
    """
    if not self._layers:
      return ""

    # Sort by priority — highest priority (1) first
    sorted_layers = sorted(self._layers, key=lambda l: l.priority)

    if max_tokens is None:
      return _join_layers(sorted_layers)

    # Token-aware assembly: include layers until budget exhausted.
    # Drop from lowest priority (highest number) first.
    included = list(sorted_layers)

    while included and _count_tokens_for_layers(included, self.model_id) > max_tokens:
      # Find the lowest-priority layer (highest priority number)
      worst = max(included, key=lambda l: l.priority)
      if worst.priority <= PRIORITY_INSTRUCTIONS:
        # Never drop sacred layers — stop trimming
        break
      included.remove(worst)

    # If still over budget after dropping all droppable layers,
    # truncate the lowest-priority remaining layer's content.
    if included and _count_tokens_for_layers(included, self.model_id) > max_tokens:
      # Find the least important remaining layer that can be truncated
      trimmable = [l for l in included if l.priority > PRIORITY_INSTRUCTIONS]
      if trimmable:
        target = max(trimmable, key=lambda l: l.priority)
        _truncate_layer_content(target, max_tokens, included, self.model_id)

    return _join_layers(sorted(included, key=lambda l: l.priority))

  def build_split(self) -> tuple:
    """Split layers into (static_prefix, dynamic_suffix) for cache optimization.

    Static = layers with cacheable=True (instructions, skills, layer guide).
    Dynamic = layers with cacheable=False (knowledge, memory, thinking).

    Returns:
      (static_prefix, dynamic_suffix) as strings.
    """
    sorted_layers = sorted(self._layers, key=lambda l: l.priority)
    static = [l for l in sorted_layers if l.cacheable]
    dynamic = [l for l in sorted_layers if not l.cacheable]
    return _join_layers(static), _join_layers(dynamic)

  def token_stats(self) -> dict:
    """Return per-layer token counts for observability."""
    stats: dict = {}
    for layer in self._layers:
      tokens = count_text_tokens(layer.content, self.model_id)
      stats[layer.name] = {"tokens": tokens, "priority": layer.priority, "cacheable": layer.cacheable}
    stats["total"] = sum(s["tokens"] for s in stats.values() if isinstance(s, dict))
    return stats


def _join_layers(layers: List[PromptLayer]) -> str:
  """Join non-empty layer contents with double newlines."""
  parts = [l.content for l in layers if l.content and l.content.strip()]
  return "\n\n".join(parts)


def _count_tokens_for_layers(layers: List[PromptLayer], model_id: str) -> int:
  """Count total tokens for a set of layers (joined text)."""
  text = _join_layers(layers)
  if not text:
    return 0
  return count_text_tokens(text, model_id)


def _truncate_layer_content(
  layer: PromptLayer,
  target_total_tokens: int,
  all_included: List[PromptLayer],
  model_id: str,
) -> None:
  """Truncate a single layer's content to bring the total under budget.

  Removes content from the end of the layer.
  """
  others_text = _join_layers([l for l in all_included if l is not layer])
  others_tokens = count_text_tokens(others_text, model_id) if others_text else 0
  # Account for the \n\n separator between this layer and others
  separator_tokens = 1 if others_text else 0
  available = target_total_tokens - others_tokens - separator_tokens

  if available <= 0:
    layer.content = ""
    return

  # Binary search for the right content length
  content = layer.content
  low, high = 0, len(content)
  while low < high:
    mid = (low + high + 1) // 2
    if count_text_tokens(content[:mid], model_id) <= available:
      low = mid
    else:
      high = mid - 1

  layer.content = content[:low]
