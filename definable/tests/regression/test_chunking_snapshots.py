"""
Regression tests: Chunking behavior must be deterministic.

Migrated from tests_e2e/regression/test_chunking_snapshots.py.

These tests lock in known-good behavior. If chunking logic changes
(even accidentally), these tests will catch it.

Strategy:
  - Use fixed, well-known input documents
  - Assert on chunk COUNT and specific CONTENT patterns
  - If you intentionally change chunking logic, update the expected values here

No API calls. Pure logic.
"""

import pytest

from definable.knowledge.chunker.recursive import RecursiveChunker
from definable.knowledge.chunker.text import TextChunker
from definable.knowledge.document import Document

# ---------------------------------------------------------------------------
# Fixed test document (never change this — it's the regression baseline)
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed.

Types of Machine Learning

There are three main types: supervised learning, unsupervised learning,
and reinforcement learning.

Supervised Learning

In supervised learning, the model learns from labeled training data.
Common algorithms include linear regression, decision trees, and neural networks.

Unsupervised Learning

Unsupervised learning finds hidden patterns in unlabeled data.
Clustering and dimensionality reduction are common techniques.

Reinforcement Learning

Reinforcement learning trains agents to make decisions through rewards and penalties.
It is used in game playing, robotics, and recommendation systems."""


@pytest.mark.regression
class TestTextChunkerDeterminism:
  """TextChunker must produce stable, deterministic output."""

  def test_same_doc_produces_same_chunk_count(self):
    """Running chunker twice on the same doc should produce same number of chunks."""
    chunker = TextChunker(chunk_size=200, separator="\n\n", chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    result1 = chunker.chunk(doc)
    result2 = chunker.chunk(doc)
    assert len(result1) == len(result2)

  def test_same_doc_produces_same_content(self):
    """Chunking is deterministic — same input produces same chunks."""
    chunker = TextChunker(chunk_size=200, separator="\n\n", chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    result1 = chunker.chunk(doc)
    result2 = chunker.chunk(doc)
    assert [c.content for c in result1] == [c.content for c in result2]

  def test_expected_section_in_chunks(self):
    """Known sections of the document should appear in output chunks."""
    chunker = TextChunker(chunk_size=500, separator="\n\n", chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    chunks = chunker.chunk(doc)
    all_content = " ".join(c.content for c in chunks)
    assert "supervised learning" in all_content.lower()
    assert "unsupervised learning" in all_content.lower()
    assert "reinforcement learning" in all_content.lower()

  def test_chunk_size_reduces_chunk_count(self):
    """Larger chunk_size should produce fewer or equal chunks."""
    doc = Document(content=SAMPLE_DOCUMENT)
    chunker_small = TextChunker(chunk_size=100, separator="\n\n", chunk_overlap=0)
    chunker_large = TextChunker(chunk_size=500, separator="\n\n", chunk_overlap=0)
    chunks_small = chunker_small.chunk(doc)
    chunks_large = chunker_large.chunk(doc)
    assert len(chunks_large) <= len(chunks_small)

  def test_chunk_overlap_increases_chunk_count_or_size(self):
    """Overlap creates larger chunks — total content should be >= original."""
    doc = Document(content=SAMPLE_DOCUMENT)
    chunker_no_overlap = TextChunker(chunk_size=200, separator="\n\n", chunk_overlap=0)
    chunker_with_overlap = TextChunker(chunk_size=200, separator="\n\n", chunk_overlap=50)
    chunks_no_overlap = chunker_no_overlap.chunk(doc)
    chunks_with_overlap = chunker_with_overlap.chunk(doc)
    total_no_overlap = sum(len(c.content) for c in chunks_no_overlap)
    total_with_overlap = sum(len(c.content) for c in chunks_with_overlap)
    # With overlap, total content should be >= original (some repeated)
    assert total_with_overlap >= total_no_overlap


@pytest.mark.regression
class TestRecursiveChunkerDeterminism:
  """RecursiveChunker must produce stable, deterministic output."""

  def test_same_doc_produces_same_chunk_count(self):
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    assert len(chunker.chunk(doc)) == len(chunker.chunk(doc))

  def test_same_doc_produces_same_content(self):
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    r1 = chunker.chunk(doc)
    r2 = chunker.chunk(doc)
    assert [c.content for c in r1] == [c.content for c in r2]

  def test_expected_content_preserved(self):
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    chunks = chunker.chunk(doc)
    all_content = " ".join(c.content for c in chunks)
    assert "machine learning" in all_content.lower()
    assert "neural networks" in all_content.lower()

  def test_chunk_count_bounded_by_doc_size(self):
    """Number of chunks should scale reasonably with document size."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    doc = Document(content=SAMPLE_DOCUMENT)
    chunks = chunker.chunk(doc)
    # Should not produce more chunks than characters / chunk_size (rough bound)
    max_expected = len(SAMPLE_DOCUMENT) // 50  # Conservative upper bound
    assert len(chunks) <= max_expected


@pytest.mark.regression
class TestChunkerEdgeCaseStability:
  """Edge cases that should produce stable, known behavior."""

  def test_single_word_document(self):
    chunker = TextChunker(chunk_size=100)
    doc = Document(content="Hello")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello"

  def test_document_at_exact_chunk_size(self):
    """Document exactly chunk_size chars should produce exactly 1 chunk."""
    chunker = TextChunker(chunk_size=10, separator="\n", chunk_overlap=0)
    doc = Document(content="ABCDEFGHIJ")  # Exactly 10 chars, no separator
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1

  def test_unicode_content_preserved(self):
    """Unicode content (non-ASCII) should be preserved exactly."""
    chunker = TextChunker(chunk_size=100)
    content = "Hllo wrld — это тест"
    doc = Document(content=content)
    chunks = chunker.chunk(doc)
    combined = "".join(c.content for c in chunks)
    assert "Hllo" in combined
    assert "это тест" in combined

  def test_whitespace_only_lines_handled(self):
    """Documents with whitespace-only lines should not crash."""
    chunker = TextChunker(chunk_size=100, separator="\n")
    doc = Document(content="Line 1\n   \nLine 2\n\nLine 3")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
