"""Tests for Knowledge + new scoring/hybrid features integration."""

import pytest

from definable.knowledge.document import Document
from definable.knowledge.base import Knowledge
from definable.knowledge.scoring.temporal import TemporalDecay
from definable.knowledge.scoring.mmr import MMRConfig


class TestKnowledgeExports:
  def test_temporal_decay_importable(self):
    from definable.knowledge import TemporalDecay

    assert TemporalDecay is not None

  def test_mmr_config_importable(self):
    from definable.knowledge import MMRConfig

    assert MMRConfig is not None

  def test_fts_index_importable(self):
    from definable.knowledge import FTSIndex

    assert FTSIndex is not None

  def test_hybrid_config_importable(self):
    from definable.knowledge import HybridSearchConfig

    assert HybridSearchConfig is not None

  def test_fallback_embedder_importable(self):
    from definable.knowledge import FallbackEmbedder

    assert FallbackEmbedder is not None

  def test_fallback_embedder_from_embedder_pkg(self):
    from definable.knowledge.embedder import FallbackEmbedder

    assert FallbackEmbedder is not None


class TestKnowledgeWithScoring:
  def test_knowledge_accepts_temporal_decay(self):
    kb = Knowledge(temporal_decay=TemporalDecay(half_life_days=7.0))
    assert kb.temporal_decay is not None
    assert kb.temporal_decay.half_life_days == 7.0

  def test_knowledge_accepts_mmr(self):
    kb = Knowledge(mmr=MMRConfig(lambda_param=0.7))
    assert kb.mmr is not None
    assert kb.mmr.lambda_param == 0.7

  def test_knowledge_search_with_temporal_decay(self):
    """Temporal decay in sync search doesn't crash."""
    kb = Knowledge(temporal_decay=TemporalDecay(half_life_days=30.0))
    # Empty DB — just ensure no crash
    results = kb.search("test", top_k=5)
    assert results == []

  def test_knowledge_search_with_mmr(self):
    """MMR in sync search doesn't crash."""
    kb = Knowledge(mmr=MMRConfig(lambda_param=0.5))
    results = kb.search("test", top_k=5)
    assert results == []

  @pytest.mark.asyncio
  async def test_knowledge_asearch_with_fts(self):
    """FTS + hybrid search in async search."""
    from definable.knowledge.fts.index import FTSIndex

    fts = FTSIndex()
    await fts.initialize()

    kb = Knowledge(fts_index=fts)
    # Empty — just verifying pipeline doesn't crash
    results = await kb.asearch("test", top_k=5)
    assert results == []

    await fts.close()

  @pytest.mark.asyncio
  async def test_knowledge_aadd_populates_fts(self):
    """aadd() indexes documents into FTS when configured."""
    from definable.knowledge.fts.index import FTSIndex
    from definable.vectordb.memory import InMemoryVectorDB
    from unittest.mock import AsyncMock, MagicMock

    fts = FTSIndex()
    await fts.initialize()

    # Create a mock VectorDB that doesn't need real embeddings
    mock_db = MagicMock(spec=InMemoryVectorDB)
    mock_db.embedder = None
    mock_db.async_create = AsyncMock()
    mock_db.content_hash_exists = MagicMock(return_value=False)
    mock_db.upsert_available = MagicMock(return_value=False)
    mock_db.ainsert = AsyncMock()

    kb = Knowledge(vector_db=mock_db, fts_index=fts)

    docs = [Document(content="Machine learning is great")]
    await kb.aadd(docs, chunk=False)

    # FTS should now have the document
    count = await fts.count()
    assert count == 1

    await fts.close()
