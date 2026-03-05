"""Tests for Cortex GraphIndex."""

import pytest
from definable.memory.cortex.index.graph import GraphIndex
from definable.memory.cortex.record.types import Edge, EdgeType


@pytest.fixture
async def graph(tmp_path):
  import aiosqlite

  db = await aiosqlite.connect(str(tmp_path / "graph.db"))
  g = GraphIndex()
  await g.initialize(db)
  yield g
  await db.close()


@pytest.mark.asyncio
class TestGraphIndex:
  async def test_add_and_get_neighbors(self, graph):
    e = Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC, weight=0.9)
    await graph.add_edge(e)
    neighbors = await graph.get_neighbors("a", direction="outgoing")
    assert len(neighbors) == 1
    assert neighbors[0].target_id == "b"

  async def test_direction_incoming(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.TEMPORAL))
    incoming = await graph.get_neighbors("b", direction="incoming")
    assert len(incoming) == 1
    assert incoming[0].source_id == "a"

  async def test_direction_both(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b"))
    await graph.add_edge(Edge(source_id="c", target_id="a"))
    both = await graph.get_neighbors("a", direction="both")
    assert len(both) == 2

  async def test_edge_type_filter(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC))
    await graph.add_edge(Edge(source_id="a", target_id="c", edge_type=EdgeType.CAUSAL))
    semantic_only = await graph.get_neighbors("a", edge_types=[EdgeType.SEMANTIC])
    assert len(semantic_only) == 1
    assert semantic_only[0].target_id == "b"

  async def test_remove_edge(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC))
    await graph.remove_edge("a", "b", EdgeType.SEMANTIC)
    neighbors = await graph.get_neighbors("a")
    assert len(neighbors) == 0

  async def test_remove_node(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b"))
    await graph.add_edge(Edge(source_id="c", target_id="a"))
    await graph.remove_node("a")
    assert await graph.count_edges() == 0

  async def test_bfs_simple(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b"))
    await graph.add_edge(Edge(source_id="b", target_id="c"))
    await graph.add_edge(Edge(source_id="c", target_id="d"))
    result = await graph.bfs("a", max_hops=3)
    ids = [r[0] for r in result]
    assert "b" in ids
    assert "c" in ids
    assert "d" in ids

  async def test_bfs_respects_max_hops(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b"))
    await graph.add_edge(Edge(source_id="b", target_id="c"))
    await graph.add_edge(Edge(source_id="c", target_id="d"))
    result = await graph.bfs("a", max_hops=1)
    ids = [r[0] for r in result]
    assert "b" in ids
    assert "c" not in ids

  async def test_bfs_with_edge_type_filter(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.CAUSAL))
    await graph.add_edge(Edge(source_id="a", target_id="c", edge_type=EdgeType.SEMANTIC))
    result = await graph.bfs("a", max_hops=1, edge_types=[EdgeType.CAUSAL])
    ids = [r[0] for r in result]
    assert "b" in ids
    assert "c" not in ids

  async def test_count_edges(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b"))
    await graph.add_edge(Edge(source_id="b", target_id="c"))
    assert await graph.count_edges() == 2

  async def test_upsert_edge(self, graph):
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC, weight=0.5))
    await graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC, weight=0.9))
    neighbors = await graph.get_neighbors("a")
    assert len(neighbors) == 1
    assert neighbors[0].weight == 0.9
