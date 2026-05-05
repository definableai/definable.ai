"""Tests for Cortex retrieval engine."""

import pytest
from definable.memory.cortex.config import CortexConfig
from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex
from definable.memory.cortex.index.tags import TagIndex
from definable.memory.cortex.record.types import Fact, MemoryRecord, NarrativeEpisode
from definable.memory.cortex.retrieval.analyzer import QueryAnalyzer, QueryType
from definable.memory.cortex.retrieval.engine import RetrievalEngine
from definable.memory.cortex.retrieval.result import RetrievalResult, ScoredMemory
from definable.memory.cortex.store import CortexStore


class TestQueryAnalyzer:
  def setup_method(self):
    self.analyzer = QueryAnalyzer()

  def test_causal_query(self):
    plan = self.analyzer.analyze("Why did the tests fail?")
    assert plan.query_type == QueryType.CAUSAL
    assert plan.use_graph is True

  def test_temporal_query(self):
    plan = self.analyzer.analyze("When did we last discuss the memory system?")
    assert plan.query_type == QueryType.TEMPORAL

  def test_entity_query(self):
    plan = self.analyzer.analyze("Who was involved in the project?")
    assert plan.query_type == QueryType.ENTITY

  def test_preference_query(self):
    plan = self.analyzer.analyze("What coding style do I prefer?")
    assert plan.query_type == QueryType.PREFERENCE

  def test_recall_query(self):
    plan = self.analyzer.analyze("What did we discuss about testing?")
    assert plan.query_type == QueryType.RECALL

  def test_factual_query(self):
    plan = self.analyzer.analyze("What is CortexMemory?")
    assert plan.query_type == QueryType.FACTUAL

  def test_general_query(self):
    plan = self.analyzer.analyze("architecture patterns")
    assert plan.query_type == QueryType.GENERAL

  def test_extracts_keywords(self):
    plan = self.analyzer.analyze("Tell me about Python testing frameworks")
    assert "python" in plan.keywords
    assert "testing" in plan.keywords
    assert "frameworks" in plan.keywords

  def test_extracts_entities(self):
    plan = self.analyzer.analyze('Check the "CortexMemory" module')
    assert "CortexMemory" in plan.entities


class TestRetrievalResult:
  def test_format_empty(self):
    result = RetrievalResult(query="test")
    xml = result.format_for_prompt()
    assert "<cortex_memory>" in xml
    assert "</cortex_memory>" in xml

  def test_format_with_memories(self):
    rec = MemoryRecord(
      raw_content="Test content",
      narrative=NarrativeEpisode(content="A test narrative"),
      facts=[Fact(content="Test fact", confidence=0.9)],
      tags=["technical"],
    )
    result = RetrievalResult(
      query="test",
      memories=[ScoredMemory(record=rec, score=0.85, source_layer="embedding")],
    )
    xml = result.format_for_prompt()
    assert "<narrative>A test narrative</narrative>" in xml
    assert "Test fact" in xml
    assert "technical" in xml
    assert 'score="0.85"' in xml

  def test_top_property(self):
    rec = MemoryRecord(raw_content="top")
    result = RetrievalResult(
      memories=[ScoredMemory(record=rec, score=0.9)],
    )
    assert result.top is not None
    assert result.top.score == 0.9

  def test_top_empty(self):
    result = RetrievalResult()
    assert result.top is None


@pytest.fixture
async def retrieval_fixtures(tmp_path):
  import aiosqlite

  store = CortexStore(db_path=str(tmp_path / "retrieval.db"))
  await store.initialize()
  config = CortexConfig()

  db = await aiosqlite.connect(str(tmp_path / "retrieval_idx.db"))
  tag_idx = TagIndex()
  await tag_idx.initialize(db)
  sig_builder = SignatureBuilder(dims=512)
  sig_idx = SignatureIndex()
  await sig_idx.initialize(db)

  engine = RetrievalEngine(
    store=store,
    config=config,
    tag_index=tag_idx,
    signature_builder=sig_builder,
    signature_index=sig_idx,
  )

  yield store, engine, tag_idx, sig_builder, sig_idx, db
  await db.close()
  await store.close()


@pytest.mark.asyncio
class TestRetrievalEngine:
  async def test_recall_empty(self, retrieval_fixtures):
    store, engine, *_ = retrieval_fixtures
    result = await engine.recall("anything", session_id="s1")
    assert isinstance(result, RetrievalResult)
    assert len(result.memories) == 0

  async def test_recall_with_records(self, retrieval_fixtures):
    store, engine, tag_idx, sig_builder, sig_idx, db = retrieval_fixtures
    # Add records with signatures and tags
    r1 = MemoryRecord(raw_content="Python testing is important", session_id="s1")
    r1.signature = sig_builder.build(r1.raw_content)
    await store.add_record(r1)
    await sig_idx.add(r1.record_id, r1.signature)
    await tag_idx.add_tags(r1.record_id, ["technical/testing"])

    r2 = MemoryRecord(raw_content="Cooking pasta recipes", session_id="s1")
    r2.signature = sig_builder.build(r2.raw_content)
    await store.add_record(r2)
    await sig_idx.add(r2.record_id, r2.signature)
    await tag_idx.add_tags(r2.record_id, ["personal/cooking"])

    result = await engine.recall("testing", session_id="s1")
    assert len(result.memories) > 0

  async def test_recall_includes_scratchpad(self, retrieval_fixtures):
    store, engine, *_ = retrieval_fixtures
    from definable.memory.cortex.record.scratchpad import Scratchpad

    sp = Scratchpad(session_id="s1", beliefs={"pref": "direct"})
    await store.save_scratchpad(sp)
    result = await engine.recall("anything", session_id="s1")
    assert "scratchpad" in result.scratchpad_context.lower()

  async def test_recall_respects_top_k(self, retrieval_fixtures):
    store, engine, *_ = retrieval_fixtures
    for i in range(20):
      await store.add_record(MemoryRecord(raw_content=f"record-{i}", session_id="s1"))
    result = await engine.recall("record", session_id="s1", top_k=5)
    assert len(result.memories) <= 5
