"""Slow-path ingestion processor — LLM-powered multi-representation extraction.

Runs 3 parallel LLM calls (narrative, facts, tags) and handles embedding generation
and graph linking. Designed to run as a background task.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional

from definable.memory.cortex.record.types import Fact, NarrativeEpisode
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.cortex.ingestion.facts import FactExtractor
  from definable.memory.cortex.ingestion.narrative import NarrativeBuilder
  from definable.memory.cortex.ingestion.tags import TagGenerator
  from definable.model.base import Model


class SlowPathResult:
  """Result of slow-path processing."""

  __slots__ = ("narrative", "facts", "tags", "embedding")

  def __init__(
    self,
    narrative: Optional[NarrativeEpisode] = None,
    facts: Optional[List[Fact]] = None,
    tags: Optional[List[str]] = None,
    embedding: Optional[List[float]] = None,
  ):
    self.narrative = narrative
    self.facts = facts or []
    self.tags = tags or []
    self.embedding = embedding


class SlowPathProcessor:
  """LLM-powered multi-representation extraction.

  Runs narrative building, fact extraction, and tag generation
  in parallel via asyncio.gather(), then generates embeddings
  and creates graph links.
  """

  def __init__(
    self,
    model: Optional["Model"] = None,
    embedder: Optional["Embedder"] = None,
    narrative_builder: Optional["NarrativeBuilder"] = None,
    fact_extractor: Optional["FactExtractor"] = None,
    tag_generator: Optional["TagGenerator"] = None,
  ):
    self.model = model
    self.embedder = embedder

    if narrative_builder is not None:
      self._narrative_builder = narrative_builder
    else:
      from definable.memory.cortex.ingestion.narrative import NarrativeBuilder

      self._narrative_builder = NarrativeBuilder(model=model)

    if fact_extractor is not None:
      self._fact_extractor = fact_extractor
    else:
      from definable.memory.cortex.ingestion.facts import FactExtractor

      self._fact_extractor = FactExtractor(model=model)

    if tag_generator is not None:
      self._tag_generator = tag_generator
    else:
      from definable.memory.cortex.ingestion.tags import TagGenerator

      self._tag_generator = TagGenerator(model=model)

  async def process(self, text: str, turn_index: int = 0) -> SlowPathResult:
    """Run full slow-path extraction.

    Fires 3 parallel LLM calls + embedding generation.
    """
    # Run LLM extractions in parallel
    narrative_coro = self._narrative_builder.build(text)
    facts_coro = self._fact_extractor.extract(text, turn_index)
    tags_coro = self._tag_generator.generate(text)

    results = await asyncio.gather(
      narrative_coro,
      facts_coro,
      tags_coro,
      return_exceptions=True,
    )

    narrative = results[0] if not isinstance(results[0], Exception) else None
    facts = results[1] if not isinstance(results[1], Exception) else []
    tags = results[2] if not isinstance(results[2], Exception) else []

    # Generate embedding
    embedding = None
    if self.embedder is not None:
      try:
        embedding = await self.embedder.async_get_embedding(text)
      except Exception as exc:
        log_warning(f"SlowPath: embedding failed: {exc}")

    log_debug(
      f"SlowPath complete: narrative={'yes' if narrative else 'no'}, "
      f"facts={len(facts)}, tags={len(tags)}, "  # type: ignore[arg-type]
      f"embedding={'yes' if embedding else 'no'}",
      log_level=2,
    )

    return SlowPathResult(
      narrative=narrative,  # type: ignore[arg-type]
      facts=facts,  # type: ignore[arg-type]
      tags=tags,  # type: ignore[arg-type]
      embedding=embedding,
    )
