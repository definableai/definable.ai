"""Tests for Cortex update engine and cascade propagation."""

import pytest
from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.index.graph import GraphIndex
from definable.memory.cortex.index.tags import TagIndex
from definable.memory.cortex.record.types import Edge, EdgeType, MemoryRecord
from definable.memory.cortex.store import CortexStore
from definable.memory.cortex.update.cascade import CascadePropagator
from definable.memory.cortex.update.consolidator import BackgroundConsolidator
from definable.memory.cortex.update.engine import UpdateEngine


@pytest.fixture
async def update_fixtures(tmp_path):
  import aiosqlite

  store = CortexStore(db_path=str(tmp_path / "update.db"))
  await store.initialize()
  config = CortexConfig()

  db = await aiosqlite.connect(str(tmp_path / "update_idx.db"))
  graph = GraphIndex()
  await graph.initialize(db)
  tag_idx = TagIndex()
  await tag_idx.initialize(db)

  cascade = CascadePropagator(store=store, graph_index=graph, config=config)
  engine = UpdateEngine(store=store, config=config, graph_index=graph, tag_index=tag_idx, cascade=cascade)
  yield store, engine, graph, tag_idx, cascade, db
  await db.close()
  await store.close()


@pytest.mark.asyncio
class TestUpdateEngine:
  async def test_update_content(self, update_fixtures):
    store, engine, *_ = update_fixtures
    r = MemoryRecord(raw_content="original", session_id="s1")
    await store.add_record(r)
    new = await engine.update_content(r.record_id, "updated", reason="correction")
    assert new is not None
    assert new.raw_content == "updated"
    # Old record should be superseded
    old = await store.get_record(r.record_id)
    assert old is not None
    assert old.superseded_by == new.record_id

  async def test_update_nonexistent(self, update_fixtures):
    _, engine, *_ = update_fixtures
    result = await engine.update_content("nonexistent", "new")
    assert result is None

  async def test_forget(self, update_fixtures):
    store, engine, graph, tag_idx, *_ = update_fixtures
    r = MemoryRecord(raw_content="to forget", session_id="s1")
    await store.add_record(r)
    await tag_idx.add_tags(r.record_id, ["test"])
    success = await engine.forget(r.record_id, reason="requested")
    assert success
    record = await store.get_record(r.record_id)
    assert record is not None
    assert record.superseded_by == "forgotten"

  async def test_forget_nonexistent(self, update_fixtures):
    _, engine, *_ = update_fixtures
    assert not await engine.forget("nonexistent")

  async def test_belief_operations(self, update_fixtures):
    store, engine, *_ = update_fixtures
    await engine.set_belief("name", "Test", session_id="s1")
    sp = await engine.get_scratchpad(session_id="s1")
    assert sp.get_belief("name") == "Test"
    await engine.remove_belief("name", session_id="s1")
    sp = await engine.get_scratchpad(session_id="s1")
    assert sp.get_belief("name") is None


@pytest.mark.asyncio
class TestCascadePropagator:
  async def test_propagate_staleness(self, update_fixtures):
    store, _, graph, _, cascade, db = update_fixtures
    # Create chain: a → b → c
    r_a = MemoryRecord(raw_content="a", session_id="s1")
    r_b = MemoryRecord(raw_content="b", session_id="s1")
    r_c = MemoryRecord(raw_content="c", session_id="s1")
    for r in [r_a, r_b, r_c]:
      await store.add_record(r)
    await graph.add_edge(Edge(source_id=r_a.record_id, target_id=r_b.record_id, edge_type=EdgeType.CAUSAL))
    await graph.add_edge(Edge(source_id=r_b.record_id, target_id=r_c.record_id, edge_type=EdgeType.CAUSAL))

    affected = await cascade.propagate(r_a.record_id)
    assert affected == 2

    b = await store.get_record(r_b.record_id)
    c = await store.get_record(r_c.record_id)
    assert b is not None and b.staleness > 0
    assert c is not None and c.staleness > 0
    # b should be more stale than c (closer to source)
    assert b.staleness > c.staleness

  async def test_propagate_no_edges(self, update_fixtures):
    store, _, graph, _, cascade, _ = update_fixtures
    r = MemoryRecord(raw_content="isolated", session_id="s1")
    await store.add_record(r)
    affected = await cascade.propagate(r.record_id)
    assert affected == 0


@pytest.fixture
async def consolidator_fixtures(tmp_path):
  store = CortexStore(db_path=str(tmp_path / "consolidate.db"))
  await store.initialize()
  config = CortexConfig(duplicate_threshold=0.92)
  consolidator = BackgroundConsolidator(store=store, config=config)
  yield store, consolidator
  await store.close()


@pytest.mark.asyncio
class TestBackgroundConsolidator:
  async def test_detect_duplicates(self, consolidator_fixtures):
    store, consolidator = consolidator_fixtures
    # Create records with very similar embeddings
    r1 = MemoryRecord(raw_content="a", session_id="s1", embedding=[1.0, 0.0, 0.0])
    r2 = MemoryRecord(raw_content="b", session_id="s1", embedding=[0.99, 0.01, 0.0])  # very similar
    r3 = MemoryRecord(raw_content="c", session_id="s1", embedding=[0.0, 1.0, 0.0])  # very different
    for r in [r1, r2, r3]:
      await store.add_record(r)

    report = await consolidator.run_once()
    assert report.duplicates_merged == 1

  async def test_prune_stale(self, consolidator_fixtures):
    store, consolidator = consolidator_fixtures
    r = MemoryRecord(raw_content="stale", session_id="s1", staleness=0.9)
    await store.add_record(r)
    report = await consolidator.run_once()
    assert report.stale_pruned == 1
    record = await store.get_record(r.record_id)
    assert record is not None
    assert record.superseded_by == "stale_pruned"

  async def test_no_ops_on_empty(self, consolidator_fixtures):
    _, consolidator = consolidator_fixtures
    report = await consolidator.run_once()
    assert report.duplicates_merged == 0
    assert report.stale_pruned == 0
