"""Retrieval result types and formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from definable.memory.cortex.record.types import MemoryRecord


@dataclass
class ScoredMemory:
  """A memory record with a relevance score."""

  record: MemoryRecord
  score: float = 0.0
  source_layer: str = ""  # Which retrieval layer produced this result

  @property
  def record_id(self) -> str:
    return self.record.record_id


@dataclass
class RetrievalResult:
  """Complete result from the retrieval engine."""

  query: str = ""
  memories: List[ScoredMemory] = field(default_factory=list)
  scratchpad_context: str = ""
  total_candidates: int = 0

  @property
  def top(self) -> Optional[ScoredMemory]:
    """Get the highest-scored memory."""
    return self.memories[0] if self.memories else None

  def format_for_prompt(self, max_memories: int = 10) -> str:
    """Format retrieval results as XML for system prompt injection.

    Produces a structured XML block that can be injected into the
    agent's system prompt for context.
    """
    parts: list[str] = ["<cortex_memory>"]

    if self.scratchpad_context:
      parts.append(self.scratchpad_context)

    memories_to_show = self.memories[:max_memories]
    if memories_to_show:
      parts.append("  <retrieved_memories>")
      for sm in memories_to_show:
        rec = sm.record
        parts.append(f'    <memory id="{rec.record_id[:8]}" score="{sm.score:.2f}" source="{sm.source_layer}">')

        if rec.narrative and rec.narrative.content:
          parts.append(f"      <narrative>{rec.narrative.content}</narrative>")

        if rec.facts:
          for fact in rec.facts:
            parts.append(f'      <fact confidence="{fact.confidence:.1f}">{fact.content}</fact>')

        if not rec.narrative and not rec.facts:
          parts.append(f"      <raw>{rec.raw_content[:500]}</raw>")

        if rec.tags:
          parts.append(f"      <tags>{', '.join(rec.tags)}</tags>")

        parts.append("    </memory>")
      parts.append("  </retrieved_memories>")

    parts.append("</cortex_memory>")
    return "\n".join(parts)
