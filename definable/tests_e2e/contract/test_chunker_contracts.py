"""
Contract tests: Every Chunker implementation must satisfy these.

The Chunker ABC defines the contract:
  - chunk(document) → List[Document]
  - chunk_many(documents) → List[Document]
  - chunk() on empty doc returns []
  - All chunks have non-empty content
  - chunk_indices are 0-based and sequential
  - chunk_total matches number of chunks

To add a new Chunker: inherit this class and provide a `chunker` fixture.
CI should verify that every Chunker implementation has a corresponding
contract test class.

No API calls. Pure logic.
"""

import pytest

from definable.knowledge.chunker.base import Chunker
from definable.knowledge.chunker.recursive import RecursiveChunker
from definable.knowledge.chunker.text import TextChunker
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------


class ChunkerContractTests:
  """
  Abstract contract test suite.

  Every concrete Chunker must pass ALL tests in this class.
  Subclass this and provide a `chunker` fixture.
  """

  @pytest.fixture
  def chunker(self) -> Chunker:
    raise NotImplementedError("Subclass must provide a chunker fixture")

  @pytest.fixture
  def sample_doc(self) -> Document:
    return Document(
      content="word " * 100,  # 500 chars — will need splitting for small chunk_size
      name="sample",
      source="test.txt",
      source_type="text",
    )

  @pytest.fixture
  def empty_doc(self) -> Document:
    return Document(content="")

  # --- Contract: return types ---

  @pytest.mark.contract
  def test_chunk_returns_list(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    assert isinstance(result, list)

  @pytest.mark.contract
  def test_chunk_returns_document_instances(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    assert all(isinstance(c, Document) for c in result)

  @pytest.mark.contract
  def test_chunk_many_returns_list(self, chunker, sample_doc):
    result = chunker.chunk_many([sample_doc, sample_doc])
    assert isinstance(result, list)

  # --- Contract: empty document ---

  @pytest.mark.contract
  def test_chunk_empty_document_returns_empty_list(self, chunker, empty_doc):
    result = chunker.chunk(empty_doc)
    assert result == []

  @pytest.mark.contract
  def test_chunk_many_with_empty_docs_returns_empty_list(self, chunker, empty_doc):
    result = chunker.chunk_many([empty_doc, empty_doc])
    assert result == []

  # --- Contract: content invariants ---

  @pytest.mark.contract
  def test_all_chunks_have_non_empty_content(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    assert all(bool(c.content) for c in result)

  @pytest.mark.contract
  def test_all_text_is_preserved_across_chunks(self, chunker):
    """No text should be silently dropped during chunking."""
    doc = Document(content="alpha beta gamma delta epsilon zeta eta theta")
    chunks = chunker.chunk(doc)
    combined = " ".join(c.content for c in chunks)
    for word in ["alpha", "beta", "gamma", "delta", "epsilon"]:
      assert word in combined

  # --- Contract: metadata invariants ---

  @pytest.mark.contract
  def test_chunk_indices_are_sequential(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    for i, chunk in enumerate(result):
      assert chunk.chunk_index == i

  @pytest.mark.contract
  def test_chunk_totals_are_consistent(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    totals = {c.chunk_total for c in result}
    assert len(totals) == 1
    assert totals.pop() == len(result)

  @pytest.mark.contract
  def test_source_is_preserved(self, chunker):
    doc = Document(content="word " * 50, source="my_file.txt", source_type="file")
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.source == "my_file.txt"
      assert chunk.source_type == "file"

  @pytest.mark.contract
  def test_parent_id_is_set(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    for chunk in result:
      assert chunk.parent_id is not None

  @pytest.mark.contract
  def test_meta_data_includes_chunk_index_and_total(self, chunker, sample_doc):
    result = chunker.chunk(sample_doc)
    for i, chunk in enumerate(result):
      assert chunk.meta_data.get("chunk_index") == i
      assert chunk.meta_data.get("chunk_total") == len(result)

  @pytest.mark.contract
  def test_meta_data_from_source_doc_inherited(self, chunker):
    doc = Document(content="word " * 50, meta_data={"author": "Alice", "version": 2})
    chunks = chunker.chunk(doc)
    for chunk in chunks:
      assert chunk.meta_data.get("author") == "Alice"
      assert chunk.meta_data.get("version") == 2


# ---------------------------------------------------------------------------
# Concrete implementations — each must pass the full contract
# ---------------------------------------------------------------------------


class TestTextChunkerContract(ChunkerContractTests):
  """TextChunker satisfies the Chunker contract."""

  @pytest.fixture
  def chunker(self) -> Chunker:
    return TextChunker(chunk_size=50, separator="\n", chunk_overlap=0)


class TestRecursiveChunkerContract(ChunkerContractTests):
  """RecursiveChunker satisfies the Chunker contract."""

  @pytest.fixture
  def chunker(self) -> Chunker:
    return RecursiveChunker(chunk_size=50, chunk_overlap=0)
