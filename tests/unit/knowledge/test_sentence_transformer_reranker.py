"""
Unit tests for SentenceTransformerReranker.

Tests reranking logic with mocked CrossEncoder.
No API calls. No model downloads.

Covers:
  - Default configuration
  - Rerank sorts by score descending
  - top_n limits results
  - Empty document list handled
  - Error handling returns original documents
  - Async via thread pool
  - Lazy model loading
  - reranking_score set on documents
"""

from unittest.mock import MagicMock, patch

import pytest

from definable.knowledge.document import Document
from definable.knowledge.reranker.sentence_transformer import SentenceTransformerReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_docs(n: int = 5) -> list:
  return [Document(content=f"Document {i} about topic {chr(65 + i)}.", meta_data={"idx": i}) for i in range(n)]


def make_reranker_with_mock(scores: list, **kwargs) -> SentenceTransformerReranker:
  """Create a reranker with a mocked cross-encoder returning given scores."""
  reranker = SentenceTransformerReranker(**kwargs)
  mock_encoder = MagicMock()
  mock_encoder.predict.return_value = scores
  object.__setattr__(reranker, "_cross_encoder", mock_encoder)
  return reranker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerConfig:
  """Defaults and configuration."""

  def test_default_model(self):
    r = SentenceTransformerReranker()
    assert r.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

  def test_default_top_n_none(self):
    r = SentenceTransformerReranker()
    assert r.top_n is None

  def test_default_batch_size(self):
    r = SentenceTransformerReranker()
    assert r.batch_size == 32

  def test_custom_model(self):
    r = SentenceTransformerReranker(model="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    assert r.model == "cross-encoder/ms-marco-TinyBERT-L-2-v2"

  def test_custom_top_n(self):
    r = SentenceTransformerReranker(top_n=3)
    assert r.top_n == 3

  def test_custom_device(self):
    r = SentenceTransformerReranker(device="cpu")
    assert r.device == "cpu"

  def test_cross_encoder_initially_none(self):
    r = SentenceTransformerReranker()
    assert r._cross_encoder is None


# ---------------------------------------------------------------------------
# Reranking logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerLogic:
  """Reranking sorts documents by cross-encoder score."""

  def test_rerank_sorts_by_score_descending(self):
    docs = make_docs(4)
    scores = [0.1, 0.9, 0.5, 0.3]
    reranker = make_reranker_with_mock(scores)

    result = reranker.rerank("query", docs)
    # Highest score first
    assert result[0].content == "Document 1 about topic B."  # score 0.9
    assert result[1].content == "Document 2 about topic C."  # score 0.5
    assert result[2].content == "Document 3 about topic D."  # score 0.3
    assert result[3].content == "Document 0 about topic A."  # score 0.1

  def test_reranking_score_set_on_documents(self):
    docs = make_docs(3)
    scores = [0.2, 0.8, 0.5]
    reranker = make_reranker_with_mock(scores)

    result = reranker.rerank("query", docs)
    for doc in result:
      assert doc.reranking_score is not None
      assert isinstance(doc.reranking_score, float)

  def test_top_n_limits_results(self):
    docs = make_docs(5)
    scores = [0.1, 0.9, 0.5, 0.3, 0.7]
    reranker = make_reranker_with_mock(scores, top_n=3)

    result = reranker.rerank("query", docs)
    assert len(result) == 3
    # Top 3 by score: 0.9, 0.7, 0.5
    assert result[0].reranking_score == pytest.approx(0.9)
    assert result[1].reranking_score == pytest.approx(0.7)
    assert result[2].reranking_score == pytest.approx(0.5)

  def test_top_n_larger_than_docs(self):
    docs = make_docs(3)
    scores = [0.5, 0.3, 0.8]
    reranker = make_reranker_with_mock(scores, top_n=10)

    result = reranker.rerank("query", docs)
    assert len(result) == 3  # Only 3 docs available

  def test_empty_documents_returns_empty(self):
    reranker = make_reranker_with_mock([])
    result = reranker.rerank("query", [])
    assert result == []

  def test_single_document(self):
    docs = [Document(content="Only one.")]
    reranker = make_reranker_with_mock([0.7])
    result = reranker.rerank("query", docs)
    assert len(result) == 1
    assert result[0].reranking_score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerErrors:
  """Error handling returns original documents."""

  def test_predict_error_returns_originals(self):
    docs = make_docs(3)
    reranker = SentenceTransformerReranker()
    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("model error")
    object.__setattr__(reranker, "_cross_encoder", mock_encoder)

    result = reranker.rerank("query", docs)
    assert len(result) == 3
    # Should return original documents unchanged
    assert result[0].content == "Document 0 about topic A."

  def test_import_error_gives_helpful_message(self):
    reranker = SentenceTransformerReranker()
    with patch.dict("sys.modules", {"sentence_transformers": None}):
      with pytest.raises(ImportError, match="sentence-transformers"):
        reranker.cross_encoder


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerAsync:
  """Async rerank via thread pool."""

  @pytest.mark.asyncio
  async def test_arerank_returns_sorted(self):
    docs = make_docs(3)
    scores = [0.1, 0.9, 0.5]
    reranker = make_reranker_with_mock(scores)

    result = await reranker.arerank("query", docs)
    assert len(result) == 3
    assert result[0].reranking_score == pytest.approx(0.9)

  @pytest.mark.asyncio
  async def test_arerank_error_returns_originals(self):
    docs = make_docs(3)
    reranker = SentenceTransformerReranker()
    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("error")
    object.__setattr__(reranker, "_cross_encoder", mock_encoder)

    result = await reranker.arerank("query", docs)
    assert len(result) == 3

  @pytest.mark.asyncio
  async def test_arerank_empty(self):
    reranker = make_reranker_with_mock([])
    result = await reranker.arerank("query", [])
    assert result == []


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerBatching:
  """Batch size is passed to predict."""

  def test_batch_size_passed_to_predict(self):
    docs = make_docs(3)
    reranker = make_reranker_with_mock([0.1, 0.2, 0.3], batch_size=16)
    reranker.rerank("query", docs)
    reranker.cross_encoder.predict.assert_called_once()
    _, kwargs = reranker.cross_encoder.predict.call_args
    assert kwargs["batch_size"] == 16


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSentenceTransformerRerankerImports:
  """Importable from convenience paths."""

  def test_import_from_knowledge_reranker(self):
    from definable.knowledge.reranker import SentenceTransformerReranker as STR

    assert STR is SentenceTransformerReranker

  def test_import_from_top_level(self):
    from definable.reranker import SentenceTransformerReranker as STR

    assert STR is SentenceTransformerReranker
