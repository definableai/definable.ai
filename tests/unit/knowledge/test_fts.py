"""Tests for FTS index and hybrid search."""

import pytest

from definable.knowledge.document import Document
from definable.knowledge.fts.index import FTSIndex
from definable.knowledge.fts.hybrid import HybridSearchConfig, HybridSearcher
from definable.knowledge.fts.keywords import extract_keywords, keywords_to_fts5_query


# ------------------------------------------------------------------
# Keyword extraction
# ------------------------------------------------------------------


class TestExtractKeywords:
  def test_basic_extraction(self):
    keywords = extract_keywords("What is machine learning?")
    assert "machine" in keywords
    assert "learning" in keywords
    assert "what" not in keywords
    assert "is" not in keywords

  def test_stop_words_removed(self):
    keywords = extract_keywords("the quick brown fox jumps over the lazy dog")
    assert "the" not in keywords
    assert "over" not in keywords
    assert "quick" in keywords

  def test_max_keywords(self):
    keywords = extract_keywords("a b c d e f g h i j k l m n", max_keywords=3)
    assert len(keywords) <= 3

  def test_empty_query(self):
    assert extract_keywords("") == []

  def test_all_stop_words(self):
    keywords = extract_keywords("the a an is are was")
    assert keywords == []


class TestKeywordsToFts5Query:
  def test_basic(self):
    query = keywords_to_fts5_query(["machine", "learning"])
    assert "machine" in query
    assert "learning" in query
    assert "OR" in query

  def test_empty(self):
    assert keywords_to_fts5_query([]) == ""

  def test_hyphenated_terms_quoted(self):
    query = keywords_to_fts5_query(["real-time"])
    assert '"real-time"' in query


# ------------------------------------------------------------------
# FTSIndex
# ------------------------------------------------------------------


class TestFTSIndex:
  @pytest.mark.asyncio
  async def test_initialize(self):
    fts = FTSIndex()
    await fts.initialize()
    assert fts._initialized
    await fts.close()

  @pytest.mark.asyncio
  async def test_add_and_search(self):
    fts = FTSIndex()
    await fts.initialize()

    docs = [
      Document(id="doc1", content="Machine learning is a subset of artificial intelligence"),
      Document(id="doc2", content="Cooking pasta requires boiling water"),
      Document(id="doc3", content="Deep learning uses neural networks"),
    ]
    count = await fts.add("hash1", docs)
    assert count == 3

    results = await fts.search("machine learning")
    assert len(results) > 0
    doc_ids = [r[0] for r in results]
    assert "doc1" in doc_ids

    await fts.close()

  @pytest.mark.asyncio
  async def test_search_returns_relevant(self):
    fts = FTSIndex()
    await fts.initialize()

    docs = [
      Document(id="ml", content="Machine learning algorithms optimize predictions"),
      Document(id="cook", content="Baking bread requires yeast and flour"),
    ]
    await fts.add("h1", docs)

    results = await fts.search("machine learning")
    assert len(results) >= 1
    assert results[0][0] == "ml"

    await fts.close()

  @pytest.mark.asyncio
  async def test_search_documents(self):
    fts = FTSIndex()
    await fts.initialize()

    docs = [Document(id="d1", content="Python programming language")]
    await fts.add("h1", docs)

    results = await fts.search_documents("Python")
    assert len(results) >= 1
    assert isinstance(results[0], Document)
    assert results[0].reranking_score is not None

    await fts.close()

  @pytest.mark.asyncio
  async def test_delete(self):
    fts = FTSIndex()
    await fts.initialize()

    docs = [Document(content="test content")]
    await fts.add("hash1", docs)
    assert await fts.count() == 1

    await fts.delete("hash1")
    assert await fts.count() == 0

    await fts.close()

  @pytest.mark.asyncio
  async def test_clear(self):
    fts = FTSIndex()
    await fts.initialize()

    await fts.add("h1", [Document(content="a")])
    await fts.add("h2", [Document(content="b")])
    assert await fts.count() == 2

    await fts.clear()
    assert await fts.count() == 0

    await fts.close()

  @pytest.mark.asyncio
  async def test_empty_content_skipped(self):
    fts = FTSIndex()
    await fts.initialize()

    count = await fts.add("h1", [Document(content="")])
    assert count == 0

    await fts.close()


# ------------------------------------------------------------------
# HybridSearcher
# ------------------------------------------------------------------


class TestHybridSearcher:
  @pytest.mark.asyncio
  async def test_merge_rrf(self):
    fts = FTSIndex()
    await fts.initialize()

    docs = [
      Document(id="d1", content="Machine learning with Python"),
      Document(id="d2", content="Cooking Italian pasta"),
    ]
    await fts.add("h1", docs)

    vector_results = [
      Document(content="Machine learning with Python", reranking_score=0.9),
      Document(content="Deep learning frameworks", reranking_score=0.7),
    ]

    searcher = HybridSearcher(fts_index=fts)
    merged = await searcher.merge(vector_results, "machine learning", limit=5)
    assert len(merged) > 0

    await fts.close()

  @pytest.mark.asyncio
  async def test_merge_with_no_fts_results(self):
    fts = FTSIndex()
    await fts.initialize()
    # Empty FTS index

    vector_results = [Document(content="test", reranking_score=0.9)]
    searcher = HybridSearcher(fts_index=fts)
    merged = await searcher.merge(vector_results, "test", limit=5)
    assert len(merged) == 1

    await fts.close()

  @pytest.mark.asyncio
  async def test_merge_with_no_vector_results(self):
    fts = FTSIndex()
    await fts.initialize()
    await fts.add("h1", [Document(content="test document")])

    searcher = HybridSearcher(fts_index=fts)
    merged = await searcher.merge([], "test", limit=5)
    assert len(merged) >= 1

    await fts.close()

  @pytest.mark.asyncio
  async def test_weighted_merge(self):
    fts = FTSIndex()
    await fts.initialize()
    await fts.add("h1", [Document(content="machine learning")])

    vector_results = [Document(content="machine learning", reranking_score=0.9)]

    searcher = HybridSearcher(
      fts_index=fts,
      config=HybridSearchConfig(merge_strategy="weighted"),
    )
    merged = await searcher.merge(vector_results, "machine learning", limit=5)
    assert len(merged) >= 1

    await fts.close()
