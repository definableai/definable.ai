"""Ingestion pipeline — orchestrates fast+slow path processing.

Fast path runs synchronously (zero LLM calls, <10ms).
Slow path fires as a background asyncio.Task.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from definable.memory.cortex.record.types import MemoryRecord, MemorySource, Edge, EdgeType
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.index.graph import GraphIndex
  from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex
  from definable.memory.cortex.index.tags import TagIndex
  from definable.memory.cortex.ingestion.slow import SlowPathProcessor
  from definable.memory.cortex.store import CortexStore
  from definable.model.base import Model


class IngestionPipeline:
  """Orchestrates fast and slow path ingestion of new memories.

  1. Fast path (sync): regex entities, binary signature, timestamp detection
  2. Record creation and immediate storage
  3. Slow path (async background): LLM narrative/facts/tags, embedding, graph linking
  """

  def __init__(
    self,
    store: "CortexStore",
    config: "CortexConfig",
    model: Optional["Model"] = None,
    embedder: Optional["Embedder"] = None,
    signature_builder: Optional["SignatureBuilder"] = None,
    signature_index: Optional["SignatureIndex"] = None,
    graph_index: Optional["GraphIndex"] = None,
    tag_index: Optional["TagIndex"] = None,
  ):
    self._store = store
    self._config = config
    self._model = model
    self._embedder = embedder
    self._sig_builder = signature_builder
    self._sig_index = signature_index
    self._graph_index = graph_index
    self._tag_index = tag_index
    self._turn_counter: int = 0
    self._background_tasks: list[asyncio.Task[None]] = []

    # Build processors
    from definable.memory.cortex.ingestion.fast import FastPathProcessor

    self._fast = FastPathProcessor(signature_builder=signature_builder)
    self._slow: Optional["SlowPathProcessor"] = None
    if config.slow_path_enabled and model is not None:
      from definable.memory.cortex.ingestion.slow import SlowPathProcessor

      self._slow = SlowPathProcessor(model=model, embedder=embedder)

  async def ingest(
    self,
    content: str,
    role: str = "user",
    source: MemorySource = MemorySource.CONVERSATION,
    session_id: str = "default",
    user_id: str = "default",
  ) -> MemoryRecord:
    """Ingest a new piece of content into Cortex.

    Fast path runs immediately. Slow path fires in background.

    Returns the created MemoryRecord (may be enriched later by slow path).
    """
    self._turn_counter += 1
    turn_index = self._turn_counter

    # Fast path
    fast_result = self._fast.process(content)

    # Create record with fast-path data
    record = MemoryRecord(
      session_id=session_id,
      user_id=user_id,
      source=source,
      raw_content=content,
      role=role,
      turn_index=turn_index,
      signature=fast_result.signature,
    )

    # Generate embedding eagerly (fast, no LLM needed) even when slow path is off
    if self._embedder:
      try:
        record.embedding = await self._embedder.async_get_embedding(content)
      except Exception as exc:
        log_warning(f"Embedding generation failed: {exc}")

    # Store immediately
    await self._store.add_record(record)

    # Index signature
    if self._sig_index and fast_result.signature:
      await self._sig_index.add(record.record_id, fast_result.signature)

    log_debug(
      f"Ingestion fast path: record={record.record_id[:8]}, entities={len(fast_result.entities)}, turn={turn_index}",
      log_level=2,
    )

    # Fire slow path in background
    if self._slow is not None and self._config.slow_path_enabled:
      task = asyncio.create_task(self._run_slow_path(record, turn_index))
      self._background_tasks.append(task)
      task.add_done_callback(lambda t: self._background_tasks.remove(t) if t in self._background_tasks else None)

    return record

  async def _run_slow_path(self, record: MemoryRecord, turn_index: int) -> None:
    """Background task: run slow path and update the record."""
    if self._slow is None:
      return

    try:
      result = await self._slow.process(record.raw_content, turn_index)

      # Update record with slow-path results
      record.narrative = result.narrative
      record.facts = result.facts
      record.tags = result.tags
      record.embedding = result.embedding
      record.updated_at = time.time()

      await self._store.update_record(record)

      # Index tags
      if self._tag_index and result.tags:
        await self._tag_index.add_tags(record.record_id, result.tags)

      # Create graph edges for entity co-occurrence
      if self._graph_index and result.facts:
        await self._create_entity_edges(record, result.facts)

      log_debug(f"Ingestion slow path complete: record={record.record_id[:8]}", log_level=2)
    except Exception as exc:
      log_warning(f"Ingestion slow path failed: {exc}")

  async def _create_entity_edges(self, record: MemoryRecord, facts: list) -> None:
    """Create ENTITY edges between records that share entities."""
    if not self._graph_index:
      return

    # Collect all entities from facts
    all_entities: set[str] = set()
    for fact in facts:
      all_entities.update(fact.entities)

    if not all_entities:
      return

    # Find other records with overlapping entities (via store scan — could be optimized)
    all_records = await self._store.get_all_records(user_id=record.user_id, active_only=True)
    for other in all_records:
      if other.record_id == record.record_id:
        continue
      other_entities = set()
      for f in other.facts:
        other_entities.update(f.entities)
      overlap = all_entities & other_entities
      if overlap:
        edge = Edge(
          source_id=record.record_id,
          target_id=other.record_id,
          edge_type=EdgeType.ENTITY,
          weight=len(overlap) / max(len(all_entities), 1),
          label=",".join(sorted(overlap)[:3]),
        )
        await self._graph_index.add_edge(edge)

  async def wait_for_background(self) -> None:
    """Wait for all background tasks to complete. Useful in tests."""
    if self._background_tasks:
      await asyncio.gather(*self._background_tasks, return_exceptions=True)
      self._background_tasks.clear()
