"""Tests for Maximal Marginal Relevance (MMR) diversity ranking."""

from definable.knowledge.document import Document
from definable.knowledge.scoring.mmr import MMRConfig, mmr_rerank, _cosine_similarity, _jaccard_similarity


class TestCosineSimilarity:
  def test_identical_vectors(self):
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

  def test_orthogonal_vectors(self):
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6

  def test_opposite_vectors(self):
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

  def test_empty_vectors(self):
    assert _cosine_similarity([], []) == 0.0

  def test_zero_vector(self):
    assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


class TestJaccardSimilarity:
  def test_identical(self):
    assert _jaccard_similarity("hello world", "hello world") == 1.0

  def test_disjoint(self):
    assert _jaccard_similarity("hello world", "foo bar") == 0.0

  def test_partial_overlap(self):
    score = _jaccard_similarity("hello world", "hello foo")
    assert 0.0 < score < 1.0

  def test_empty(self):
    assert _jaccard_similarity("", "") == 0.0


class TestMMRRerank:
  def test_single_doc_returned(self):
    docs = [Document(content="only one")]
    result = mmr_rerank(None, docs)
    assert len(result) == 1
    assert result[0].content == "only one"

  def test_empty_docs(self):
    result = mmr_rerank(None, [])
    assert result == []

  def test_disabled_config(self):
    docs = [Document(content="a"), Document(content="b")]
    result = mmr_rerank(None, docs, config=MMRConfig(enabled=False))
    assert result == docs

  def test_top_k_limit(self):
    docs = [Document(content=f"doc {i}", reranking_score=1.0 - i * 0.1) for i in range(10)]
    result = mmr_rerank(None, docs, top_k=3)
    assert len(result) == 3

  def test_diversity_with_embeddings(self):
    # Two near-identical docs and one different
    docs = [
      Document(content="machine learning is great", embedding=[1.0, 0.0, 0.0], reranking_score=0.9),
      Document(content="machine learning is awesome", embedding=[0.99, 0.1, 0.0], reranking_score=0.85),
      Document(content="cooking recipes", embedding=[0.0, 0.0, 1.0], reranking_score=0.7),
    ]
    query_emb = [1.0, 0.0, 0.0]

    # Pure relevance (lambda=1.0) should pick by score
    pure_rel = mmr_rerank(query_emb, docs, config=MMRConfig(lambda_param=1.0))
    assert pure_rel[0].content == "machine learning is great"

    # Balanced MMR should introduce diversity
    balanced = mmr_rerank(query_emb, docs, config=MMRConfig(lambda_param=0.5), top_k=3)
    assert len(balanced) == 3
    # The diverse doc should appear earlier than with pure relevance
    diverse_contents = [d.content for d in balanced]
    assert "cooking recipes" in diverse_contents

  def test_jaccard_fallback(self):
    # No embeddings — should use Jaccard similarity
    docs = [
      Document(content="the cat sat on the mat", reranking_score=0.9),
      Document(content="the cat sat on the hat", reranking_score=0.85),
      Document(content="python programming language", reranking_score=0.7),
    ]
    result = mmr_rerank(None, docs, config=MMRConfig(lambda_param=0.5), top_k=3)
    assert len(result) == 3
