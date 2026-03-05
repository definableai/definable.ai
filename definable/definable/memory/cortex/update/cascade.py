"""Cascade propagator — graph-based staleness propagation.

When a record is updated or deleted, related records may become stale.
This propagator walks the graph via BFS on CAUSAL+ENTITY edges and
increases staleness scores on connected records, decaying by 0.5 per hop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from definable.memory.cortex.record.types import EdgeType
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.index.graph import GraphIndex
  from definable.memory.cortex.store import CortexStore


class CascadePropagator:
  """Propagates staleness through the knowledge graph.

  When a record changes, connected records become slightly stale.
  Staleness decays by config.staleness_decay per hop (default 0.5).
  """

  def __init__(
    self,
    store: "CortexStore",
    graph_index: "GraphIndex",
    config: "CortexConfig",
  ):
    self._store = store
    self._graph = graph_index
    self._config = config

  async def propagate(self, source_id: str, initial_staleness: float = 1.0) -> int:
    """Propagate staleness from a changed record through the graph.

    Args:
      source_id: The record that was changed.
      initial_staleness: Starting staleness value at the source.

    Returns:
      Number of records whose staleness was increased.
    """
    affected = 0
    decay = self._config.staleness_decay
    max_hops = self._config.graph_max_hops

    # BFS on CAUSAL and ENTITY edges
    cascade_edges = [EdgeType.CAUSAL, EdgeType.ENTITY]
    neighbors = await self._graph.bfs(source_id, max_hops=max_hops, edge_types=cascade_edges)

    for neighbor_id, depth in neighbors:
      record = await self._store.get_record(neighbor_id)
      if record is None or not record.is_active:
        continue

      # Staleness decays exponentially with distance
      added_staleness = initial_staleness * (decay**depth)
      record.staleness = min(1.0, record.staleness + added_staleness)
      record.updated_at = time.time()
      await self._store.update_record(record)
      affected += 1

    if affected:
      log_debug(f"Cascade: {source_id[:8]} affected {affected} records", log_level=2)

    return affected
