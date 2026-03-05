"""Background consolidator — merge, prune, and compress memory records.

Runs periodically to:
1. Detect duplicates (cosine similarity > threshold) and merge them
2. Prune stale records (staleness > 0.8)
3. Compress old, low-importance records
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from typing import TYPE_CHECKING, List, Optional

from definable.memory.cortex.record.types import MemoryRecord
from definable.utils.log import log_debug, log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.store import CortexStore


class BackgroundConsolidator:
  """Periodic background task that consolidates memory records.

  Operations:
    - Duplicate detection (cosine similarity > duplicate_threshold)
    - Staleness pruning (staleness > 0.8 → soft-delete)
    - No LLM calls — pure algorithmic consolidation
  """

  def __init__(
    self,
    store: "CortexStore",
    config: "CortexConfig",
    embedder: Optional["Embedder"] = None,
  ):
    self._store = store
    self._config = config
    self._embedder = embedder
    self._task: Optional[asyncio.Task[None]] = None
    self._running = False

  def start(self) -> None:
    """Start the background consolidation loop."""
    if self._task is not None:
      return
    self._running = True
    self._task = asyncio.create_task(self._loop())

  async def stop(self) -> None:
    """Stop the background consolidation loop."""
    self._running = False
    if self._task:
      self._task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._task
      self._task = None

  async def run_once(self, user_id: str = "default") -> ConsolidationReport:
    """Run a single consolidation pass."""
    report = ConsolidationReport()

    records = await self._store.get_all_records(user_id=user_id, active_only=True)
    if not records:
      return report

    # Pass 1: Duplicate detection
    duplicates_merged = await self._detect_and_merge_duplicates(records)
    report.duplicates_merged = duplicates_merged

    # Pass 2: Staleness pruning
    stale_pruned = await self._prune_stale(records)
    report.stale_pruned = stale_pruned

    report.total_records = len(records)
    report.active_after = report.total_records - duplicates_merged - stale_pruned

    log_debug(
      f"Consolidation: merged={duplicates_merged}, pruned={stale_pruned}, active={report.active_after}/{report.total_records}",
      log_level=2,
    )
    return report

  async def _detect_and_merge_duplicates(self, records: List[MemoryRecord]) -> int:
    """Find and merge duplicate records based on embedding similarity."""
    merged = 0
    threshold = self._config.duplicate_threshold

    # Only compare records that have embeddings
    with_embedding = [r for r in records if r.embedding and r.is_active]
    if len(with_embedding) < 2:
      return 0

    seen: set[str] = set()
    for i, r1 in enumerate(with_embedding):
      if r1.record_id in seen:
        continue
      for j in range(i + 1, len(with_embedding)):
        r2 = with_embedding[j]
        if r2.record_id in seen:
          continue
        sim = _cosine_similarity(r1.embedding, r2.embedding)  # type: ignore[arg-type]
        if sim >= threshold:
          # Keep the newer record, supersede the older one
          older, newer = (r1, r2) if r1.created_at <= r2.created_at else (r2, r1)
          older.superseded_by = newer.record_id
          older.updated_at = time.time()
          await self._store.update_record(older)
          seen.add(older.record_id)
          merged += 1

    return merged

  async def _prune_stale(self, records: List[MemoryRecord]) -> int:
    """Soft-delete records with high staleness."""
    pruned = 0
    for record in records:
      if record.is_active and record.staleness > 0.8:
        record.superseded_by = "stale_pruned"
        record.updated_at = time.time()
        await self._store.update_record(record)
        pruned += 1
    return pruned

  async def _loop(self) -> None:
    """Background loop that runs consolidation periodically."""
    while self._running:
      try:
        await asyncio.sleep(self._config.consolidation_interval_s)
        if self._running:
          await self.run_once()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        log_warning(f"Consolidator error: {exc}")


class ConsolidationReport:
  """Report from a consolidation pass."""

  __slots__ = ("duplicates_merged", "stale_pruned", "total_records", "active_after")

  def __init__(self) -> None:
    self.duplicates_merged = 0
    self.stale_pruned = 0
    self.total_records = 0
    self.active_after = 0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  if norm_a == 0.0 or norm_b == 0.0:
    return 0.0
  return dot / (norm_a * norm_b)
