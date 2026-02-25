"""
Unit tests for SemanticChunker.

Tests pure logic: does the chunker split text on semantic boundaries?
Uses mock embedder — no API calls.

Covers:
  - Semantic boundary detection via cosine similarity
  - Fallback to size-based splitting when no embedder
  - similarity_threshold controls split aggressiveness
  - min_sentences enforced
  - sentence_window averaging
  - Chunk metadata preserved
  - Helper methods: cosine_similarity, average_embeddings
"""

import pytest

from definable.knowledge.chunker.semantic import SemanticChunker
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------


class MockEmbedder:
  """Returns fixed embeddings for testing semantic boundaries."""

  def __init__(self, embeddings=None):
    self._embeddings = embeddings or {}
    self._default_dim = 3
    self._call_count = 0

  def get_embedding(self, text: str) -> list:
    self._call_count += 1
    if text in self._embeddings:
      return self._embeddings[text]
    # Default: hash-based pseudo-embedding
    h = hash(text) % 1000
    return [float(h % 10) / 10, float((h // 10) % 10) / 10, float((h // 100) % 10) / 10]


class SimilarEmbedder:
  """Returns the same embedding for all text — everything is similar."""

  def get_embedding(self, text: str) -> list:
    return [1.0, 0.0, 0.0]


class DissimilarEmbedder:
  """Returns alternating orthogonal embeddings — everything is dissimilar."""

  def __init__(self):
    self._count = 0

  def get_embedding(self, text: str) -> list:
    self._count += 1
    if self._count % 2 == 0:
      return [1.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0]


class FailingEmbedder:
  """Raises an error on get_embedding."""

  def get_embedding(self, text: str) -> list:
    raise RuntimeError("Embedding service unavailable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(content: str, name: str = "test_doc") -> Document:
  return Document(content=content, name=name, source="test.txt", source_type="text")


def make_multisentence(n: int = 10) -> str:
  """Create N distinct sentences."""
  return " ".join([f"Sentence number {i} about topic {chr(65 + i % 26)}." for i in range(n)])


# ---------------------------------------------------------------------------
# Basic splitting with embedder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerWithEmbedder:
  """SemanticChunker with a mock embedder."""

  def test_empty_document_returns_empty(self):
    chunker = SemanticChunker(embedder=MockEmbedder())
    assert chunker.chunk(make_doc("")) == []

  def test_short_document_returns_single_chunk(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder(), min_sentences=2)
    doc = make_doc("One sentence.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1

  def test_similar_text_stays_together(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder(), similarity_threshold=0.5)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    # All similar → should stay as one chunk (or few based on min_sentences)
    assert len(chunks) <= 2

  def test_dissimilar_text_gets_split(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1

  def test_threshold_0_splits_nothing(self):
    """With threshold=0, nothing is below threshold → single chunk."""
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.0, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1

  def test_threshold_1_splits_everything(self):
    """With threshold=1.0, even similar text gets split."""
    chunker = SemanticChunker(embedder=MockEmbedder(), similarity_threshold=1.0, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1

  def test_embedder_failure_falls_back_to_size(self):
    chunker = SemanticChunker(chunk_size=50, embedder=FailingEmbedder(), min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    # Should not crash — falls back to size-based splitting
    assert len(chunks) >= 1

  def test_all_content_preserved(self):
    text = make_multisentence(6)
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(text)
    chunks = chunker.chunk(doc)
    combined = " ".join(c.content for c in chunks)
    for i in range(6):
      assert f"Sentence number {i}" in combined


# ---------------------------------------------------------------------------
# Fallback (no embedder)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerFallback:
  """SemanticChunker without embedder uses size-based splitting."""

  def test_no_embedder_splits_by_size(self):
    chunker = SemanticChunker(chunk_size=100, embedder=None, min_sentences=1)
    doc = make_doc(make_multisentence(20))
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1

  def test_no_embedder_empty_returns_empty(self):
    chunker = SemanticChunker(chunk_size=100, embedder=None)
    assert chunker.chunk(make_doc("")) == []

  def test_no_embedder_preserves_content(self):
    text = make_multisentence(8)
    chunker = SemanticChunker(chunk_size=100, embedder=None, min_sentences=1)
    doc = make_doc(text)
    chunks = chunker.chunk(doc)
    combined = " ".join(c.content for c in chunks)
    for i in range(8):
      assert f"Sentence number {i}" in combined


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerConfig:
  """Configurable parameters work correctly."""

  def test_min_sentences_enforced(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder(), min_sentences=5)
    doc = make_doc("Short. Very short.")  # 2 sentences < min_sentences
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1  # Returns doc as-is

  def test_sentence_window_larger(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, sentence_window=3, min_sentences=1)
    doc = make_doc(make_multisentence(15))
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1

  def test_chunk_many_works(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder())
    docs = [make_doc(make_multisentence(5)), make_doc(make_multisentence(5))]
    chunks = chunker.chunk_many(docs)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerMetadata:
  """Chunk metadata must be correct."""

  def test_chunk_indices_sequential(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert chunk.chunk_index == i

  def test_chunk_total_consistent(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    totals = {c.chunk_total for c in chunks}
    assert len(totals) == 1
    assert totals.pop() == len(chunks)

  def test_source_preserved(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = Document(content=make_multisentence(10), source="file.txt", source_type="text")
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.source == "file.txt"

  def test_parent_id_set(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    parent_ids = {c.parent_id for c in chunks}
    assert len(parent_ids) == 1
    assert parent_ids.pop() is not None

  def test_meta_data_has_chunker_field(self):
    chunker = SemanticChunker(embedder=DissimilarEmbedder(), similarity_threshold=0.5, min_sentences=1)
    doc = make_doc(make_multisentence(10))
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("chunker") == "semantic"

  def test_meta_data_inherited(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder())
    doc = Document(content=make_multisentence(5), meta_data={"tag": "test"})
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("tag") == "test"


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerHelpers:
  """Static helper methods work correctly."""

  def test_cosine_similarity_identical(self):
    sim = SemanticChunker._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert sim == pytest.approx(1.0)

  def test_cosine_similarity_orthogonal(self):
    sim = SemanticChunker._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert sim == pytest.approx(0.0)

  def test_cosine_similarity_opposite(self):
    sim = SemanticChunker._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert sim == pytest.approx(-1.0)

  def test_cosine_similarity_empty(self):
    assert SemanticChunker._cosine_similarity([], [1.0, 0.0]) == 0.0
    assert SemanticChunker._cosine_similarity([1.0], []) == 0.0

  def test_cosine_similarity_zero_vector(self):
    assert SemanticChunker._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

  def test_average_embeddings_single(self):
    avg = SemanticChunker._average_embeddings([[1.0, 2.0, 3.0]])
    assert avg == [1.0, 2.0, 3.0]

  def test_average_embeddings_two(self):
    avg = SemanticChunker._average_embeddings([[2.0, 0.0], [0.0, 4.0]])
    assert avg == pytest.approx([1.0, 2.0])

  def test_average_embeddings_empty(self):
    assert SemanticChunker._average_embeddings([]) == []

  def test_returns_list_of_documents(self):
    chunker = SemanticChunker(embedder=SimilarEmbedder())
    doc = make_doc(make_multisentence(5))
    result = chunker.chunk(doc)
    assert isinstance(result, list)
    assert all(isinstance(c, Document) for c in result)
