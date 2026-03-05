"""Graph index — multi-relation knowledge graph with BFS traversal.

Stores typed edges between memory records in SQLite. Supports BFS traversal
with edge-type filtering for relationship-based retrieval.
"""

from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Set, Tuple

from definable.memory.cortex.record.types import Edge, EdgeType


class GraphIndex:
  """SQLite-backed directed graph for memory record relationships.

  Tables:
    cortex_edges: Stores directed edges with type, weight, and label.
  """

  def __init__(self, db: Any = None):
    self._db = db
    self._initialized = False

  async def initialize(self, db: Any) -> None:
    """Initialize with a shared aiosqlite connection."""
    self._db = db
    assert self._db is not None
    await self._db.executescript("""
      CREATE TABLE IF NOT EXISTS cortex_edges (
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        edge_type TEXT NOT NULL DEFAULT 'semantic',
        weight REAL NOT NULL DEFAULT 1.0,
        label TEXT,
        created_at REAL NOT NULL,
        PRIMARY KEY (source_id, target_id, edge_type)
      );

      CREATE INDEX IF NOT EXISTS idx_cortex_edges_source ON cortex_edges(source_id);
      CREATE INDEX IF NOT EXISTS idx_cortex_edges_target ON cortex_edges(target_id);
      CREATE INDEX IF NOT EXISTS idx_cortex_edges_type ON cortex_edges(edge_type);
    """)
    await self._db.commit()
    self._initialized = True

  async def add_edge(self, edge: Edge) -> None:
    """Add a directed edge."""
    assert self._db is not None
    await self._db.execute(
      """INSERT OR REPLACE INTO cortex_edges
         (source_id, target_id, edge_type, weight, label, created_at)
         VALUES (?, ?, ?, ?, ?, ?)""",
      (edge.source_id, edge.target_id, edge.edge_type.value, edge.weight, edge.label, edge.created_at),
    )
    await self._db.commit()

  async def remove_edge(self, source_id: str, target_id: str, edge_type: Optional[EdgeType] = None) -> None:
    """Remove edge(s) between source and target."""
    assert self._db is not None
    if edge_type is not None:
      await self._db.execute(
        "DELETE FROM cortex_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (source_id, target_id, edge_type.value),
      )
    else:
      await self._db.execute(
        "DELETE FROM cortex_edges WHERE source_id = ? AND target_id = ?",
        (source_id, target_id),
      )
    await self._db.commit()

  async def remove_node(self, record_id: str) -> None:
    """Remove all edges involving a record."""
    assert self._db is not None
    await self._db.execute("DELETE FROM cortex_edges WHERE source_id = ? OR target_id = ?", (record_id, record_id))
    await self._db.commit()

  async def get_neighbors(
    self,
    record_id: str,
    edge_types: Optional[List[EdgeType]] = None,
    direction: str = "outgoing",
  ) -> List[Edge]:
    """Get edges from/to a record, optionally filtered by type.

    Args:
      record_id: The record to find neighbors for.
      edge_types: Filter to these edge types. None = all.
      direction: "outgoing", "incoming", or "both".

    Returns:
      List of Edge objects.
    """
    assert self._db is not None
    edges: List[Edge] = []

    if direction in ("outgoing", "both"):
      query = "SELECT * FROM cortex_edges WHERE source_id = ?"
      params: list[Any] = [record_id]
      if edge_types:
        placeholders = ",".join("?" for _ in edge_types)
        query += f" AND edge_type IN ({placeholders})"
        params.extend(et.value for et in edge_types)
      cursor = await self._db.execute(query, params)
      for row in await cursor.fetchall():
        edges.append(self._row_to_edge(row))

    if direction in ("incoming", "both"):
      query = "SELECT * FROM cortex_edges WHERE target_id = ?"
      params = [record_id]
      if edge_types:
        placeholders = ",".join("?" for _ in edge_types)
        query += f" AND edge_type IN ({placeholders})"
        params.extend(et.value for et in edge_types)
      cursor = await self._db.execute(query, params)
      for row in await cursor.fetchall():
        edges.append(self._row_to_edge(row))

    return edges

  async def bfs(
    self,
    start_id: str,
    max_hops: int = 3,
    edge_types: Optional[List[EdgeType]] = None,
  ) -> List[Tuple[str, int]]:
    """Breadth-first traversal from a start node.

    Args:
      start_id: Starting record ID.
      max_hops: Maximum BFS depth.
      edge_types: Filter to these edge types. None = all.

    Returns:
      List of (record_id, depth) pairs found during traversal.
    """
    visited: Set[str] = {start_id}
    queue: deque[Tuple[str, int]] = deque([(start_id, 0)])
    results: List[Tuple[str, int]] = []

    while queue:
      current_id, depth = queue.popleft()
      if depth > 0:
        results.append((current_id, depth))
      if depth >= max_hops:
        continue

      neighbors = await self.get_neighbors(current_id, edge_types=edge_types, direction="both")
      for edge in neighbors:
        neighbor_id = edge.target_id if edge.source_id == current_id else edge.source_id
        if neighbor_id not in visited:
          visited.add(neighbor_id)
          queue.append((neighbor_id, depth + 1))

    return results

  async def count_edges(self) -> int:
    """Count total edges in the graph."""
    assert self._db is not None
    cursor = await self._db.execute("SELECT COUNT(*) FROM cortex_edges")
    row = await cursor.fetchone()
    return row[0] if row else 0

  @staticmethod
  def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    return Edge(
      source_id=row[0],
      target_id=row[1],
      edge_type=EdgeType(row[2]),
      weight=row[3],
      label=row[4],
      created_at=row[5],
    )

  async def close(self) -> None:
    """No-op — db lifecycle managed by CortexStore."""
    pass
