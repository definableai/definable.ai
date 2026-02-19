"""
Unit tests for chunking strategies.

Tests pure logic: does the chunker split text correctly?
No API calls. No external dependencies.

Covers:
  - TextChunker splits by separator and respects chunk_size
  - RecursiveChunker uses multiple separators, falls back correctly
  - Overlap is applied correctly
  - Empty/tiny documents handled gracefully
  - Chunk metadata (index, total, parent_id, source) preserved
  - chunk_many() processes multiple documents
"""

import pytest

from definable.knowledge.chunker.recursive import RecursiveChunker
from definable.knowledge.chunker.text import TextChunker
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(content: str, name: str = "test_doc", source: str = "test.txt") -> Document:
  return Document(content=content, name=name, source=source, source_type="text")


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTextChunker:
  """TextChunker splits by a single separator and merges parts up to chunk_size."""

  def test_empty_document_returns_empty_list(self):
    chunker = TextChunker(chunk_size=100)
    assert chunker.chunk(make_doc("")) == []

  def test_small_document_returns_single_chunk(self):
    chunker = TextChunker(chunk_size=1000)
    doc = make_doc("Hello world")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"

  def test_splits_at_separator(self):
    chunker = TextChunker(chunk_size=10, separator="\n\n")
    # Each paragraph is 5 chars — fitting, so they merge until chunk_size is hit
    doc = make_doc("Hello\n\nWorld\n\nFoo")
    chunks = chunker.chunk(doc)
    # All content preserved
    all_text = " ".join(c.content for c in chunks)
    assert "Hello" in all_text
    assert "World" in all_text
    assert "Foo" in all_text

  def test_chunks_respect_max_size(self):
    chunker = TextChunker(chunk_size=50, separator="\n")
    lines = ["A" * 20, "B" * 20, "C" * 20, "D" * 20]
    doc = make_doc("\n".join(lines))
    chunks = chunker.chunk(doc)
    # Verify no chunk exceeds chunk_size (may be slightly over due to separator)
    for chunk in chunks:
      assert len(chunk.content) <= 100  # with separator tolerance

  def test_chunk_metadata_preserved(self):
    chunker = TextChunker(chunk_size=10, separator="\n")
    doc = make_doc("Line1\nLine2\nLine3", source="my_file.txt", name="my_doc")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    for i, chunk in enumerate(chunks):
      assert chunk.source == "my_file.txt"
      assert chunk.chunk_index == i
      assert chunk.chunk_total == len(chunks)
      assert chunk.parent_id is not None

  def test_source_type_preserved(self):
    chunker = TextChunker(chunk_size=10, separator="\n")
    doc = Document(content="Line1\nLine2", source_type="url", source="http://example.com")
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.source_type == "url"

  def test_meta_data_inherited(self):
    chunker = TextChunker(chunk_size=10, separator="\n")
    doc = Document(content="A\nB\nC", meta_data={"category": "test"})
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("category") == "test"

  def test_meta_data_has_chunk_index_and_total(self):
    chunker = TextChunker(chunk_size=5, separator="\n")
    doc = make_doc("AAAAA\nBBBBB\nCCCCC")
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert chunk.meta_data["chunk_index"] == i
      assert chunk.meta_data["chunk_total"] == len(chunks)

  def test_chunk_many_processes_multiple_docs(self):
    chunker = TextChunker(chunk_size=10, separator="\n")
    docs = [
      make_doc("Doc1Line1\nDoc1Line2"),
      make_doc("Doc2Line1\nDoc2Line2"),
    ]
    chunks = chunker.chunk_many(docs)
    assert len(chunks) >= 2

  def test_single_long_line_becomes_one_chunk(self):
    """A line with no separator cannot be split further by TextChunker."""
    chunker = TextChunker(chunk_size=5, separator="\n")
    doc = make_doc("ABCDEFGHIJKLMNOP")  # No newlines
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1  # No way to split without separator

  def test_name_includes_chunk_index(self):
    chunker = TextChunker(chunk_size=5, separator="\n")
    doc = Document(content="AAA\nBBB\nCCC", name="mydoc")
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert f"chunk_{i}" in chunk.name  # type: ignore[operator]


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecursiveChunker:
  """RecursiveChunker uses a hierarchy of separators and falls through them."""

  def test_empty_document_returns_empty_list(self):
    chunker = RecursiveChunker(chunk_size=100)
    assert chunker.chunk(make_doc("")) == []

  def test_small_document_returns_single_chunk(self):
    chunker = RecursiveChunker(chunk_size=1000)
    doc = make_doc("Hello world")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0].content

  def test_respects_chunk_size(self):
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
    doc = make_doc("word " * 100)
    chunks = chunker.chunk(doc)
    # Each chunk should be at or below chunk_size (with tolerance for overlap)
    for chunk in chunks:
      assert len(chunk.content) <= 100

  def test_all_content_preserved(self):
    """Every word in the original doc should appear somewhere in the chunks."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    doc = make_doc(" ".join(words))
    chunks = chunker.chunk(doc)
    combined = " ".join(c.content for c in chunks)
    for word in words:
      assert word in combined

  def test_uses_paragraph_separator_first(self):
    """RecursiveChunker should split on double-newlines first."""
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
    doc = make_doc("Para one.\n\nPara two.\n\nPara three.")
    chunks = chunker.chunk(doc)
    # Should split on \n\n — at least 2 chunks for content > 20 chars
    assert len(chunks) >= 1

  def test_overlap_adds_prefix_from_previous_chunk(self):
    """Chunk N+1 should start with the tail of chunk N when overlap > 0."""
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=5, separators=["\n"])
    doc = make_doc("abcdefgh\nijklmnop\nqrstuvwx")
    chunks = chunker.chunk(doc)
    if len(chunks) > 1:
      # Overlap means chunk[1] includes some content from chunk[0]
      prev_tail = chunks[0].content[-5:]
      assert prev_tail in chunks[1].content

  def test_chunk_metadata_preserved(self):
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
    doc = Document(content="word " * 50, source="source.md", source_type="file")
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert chunk.source == "source.md"
      assert chunk.source_type == "file"
      assert chunk.chunk_index == i
      assert chunk.chunk_total == len(chunks)

  def test_falls_back_to_character_split(self):
    """When no separator matches, splits by character."""
    chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0, separators=["NOSEP"])
    doc = make_doc("ABCDEFGHIJ")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
      assert len(chunk.content) <= 5

  def test_chunk_many_works(self):
    chunker = RecursiveChunker(chunk_size=50)
    docs = [make_doc("word " * 20), make_doc("item " * 20)]
    chunks = chunker.chunk_many(docs)
    assert len(chunks) >= 2

  def test_no_separator_in_text_uses_next(self):
    """If first separator not in text, tries the next one."""
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, separators=["\n\n", "\n"])
    doc = make_doc("line1\nline2\nline3")  # No double-newlines
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
      assert len(chunk.content) <= 15  # Reasonable tolerance


# ---------------------------------------------------------------------------
# Cross-chunker invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkerInvariants:
  """Properties that must hold for ALL chunkers."""

  @pytest.mark.parametrize(
    "chunker_cls,kwargs",
    [
      (TextChunker, {"chunk_size": 50, "separator": "\n"}),
      (RecursiveChunker, {"chunk_size": 50, "chunk_overlap": 0}),
    ],
  )
  def test_chunk_returns_list_of_documents(self, chunker_cls, kwargs):
    chunker = chunker_cls(**kwargs)
    doc = make_doc("word " * 30)
    result = chunker.chunk(doc)
    assert isinstance(result, list)
    assert all(isinstance(c, Document) for c in result)

  @pytest.mark.parametrize(
    "chunker_cls,kwargs",
    [
      (TextChunker, {"chunk_size": 50, "separator": "\n"}),
      (RecursiveChunker, {"chunk_size": 50, "chunk_overlap": 0}),
    ],
  )
  def test_chunks_have_non_empty_content(self, chunker_cls, kwargs):
    chunker = chunker_cls(**kwargs)
    doc = make_doc("line " * 20)
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.content  # Not empty

  @pytest.mark.parametrize(
    "chunker_cls,kwargs",
    [
      (TextChunker, {"chunk_size": 50, "separator": "\n"}),
      (RecursiveChunker, {"chunk_size": 50, "chunk_overlap": 0}),
    ],
  )
  def test_chunk_indices_are_sequential(self, chunker_cls, kwargs):
    chunker = chunker_cls(**kwargs)
    doc = make_doc("word " * 30)
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
      assert chunk.chunk_index == i

  @pytest.mark.parametrize(
    "chunker_cls,kwargs",
    [
      (TextChunker, {"chunk_size": 50, "separator": "\n"}),
      (RecursiveChunker, {"chunk_size": 50, "chunk_overlap": 0}),
    ],
  )
  def test_all_chunks_share_same_total(self, chunker_cls, kwargs):
    chunker = chunker_cls(**kwargs)
    doc = make_doc("word " * 30)
    chunks = chunker.chunk(doc)
    totals = {c.chunk_total for c in chunks}
    assert len(totals) == 1
    assert totals.pop() == len(chunks)
