"""
Unit tests for the Knowledge class.

Tests Knowledge construction, defaults, and agent integration helpers
using mock VectorDB and mock embedder. No API calls.

Covers:
  - Knowledge(vector_db=mock_db) creates instance
  - Knowledge.top_k default value
  - Knowledge.trigger default value
  - Knowledge field defaults and overrides
  - Knowledge._read_source dispatches correctly for Document / List[Document] / string
  - Knowledge.format_context outputs xml/markdown/json
  - Knowledge.extract_query from messages
  - Knowledge creates InMemoryVectorDB when no vector_db is provided
"""

from typing import List
from unittest.mock import MagicMock

import pytest

from definable.knowledge.base import Knowledge
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Mock VectorDB (minimal stub)
# ---------------------------------------------------------------------------


class MockVectorDB:
  """Minimal VectorDB stub for unit testing Knowledge."""

  def __init__(self, embedder=None):
    self.embedder = embedder
    self._docs: List[Document] = []
    self._created = False

  def create(self):
    self._created = True

  async def async_create(self):
    self._created = True

  def content_hash_exists(self, content_hash: str) -> bool:
    return False

  def upsert_available(self) -> bool:
    return False

  def insert(self, content_hash: str, documents: List[Document]):
    self._docs.extend(documents)

  async def ainsert(self, content_hash: str, documents: List[Document]):
    self._docs.extend(documents)

  def search(self, query: str, limit: int = 10, filters=None) -> List[Document]:
    return self._docs[:limit]

  async def asearch(self, query: str, limit: int = 10, filters=None) -> List[Document]:
    return self._docs[:limit]

  def delete_by_id(self, doc_id: str):
    self._docs = [d for d in self._docs if d.id != doc_id]

  def delete(self):
    self._docs.clear()

  def count(self) -> int:
    return len(self._docs)


# ---------------------------------------------------------------------------
# Tests: Knowledge construction and defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeConstruction:
  """Knowledge dataclass construction and field defaults."""

  def test_create_with_mock_vectordb(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    assert kb.vector_db is db

  def test_default_top_k_is_5(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.top_k == 5

  def test_default_trigger_is_always(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.trigger == "always"

  def test_default_context_format_is_xml(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.context_format == "xml"

  def test_default_context_position_is_system(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.context_position == "system"

  def test_default_query_from_is_last_user(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.query_from == "last_user"

  def test_default_max_query_length_is_500(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.max_query_length == 500

  def test_default_enabled_is_true(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.enabled is True

  def test_default_rerank_is_true(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.rerank is True

  def test_default_min_score_is_none(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.min_score is None

  def test_default_embedder_is_none(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.embedder is None

  def test_default_reranker_is_none(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.reranker is None

  def test_default_chunker_is_none(self):
    kb = Knowledge(vector_db=MockVectorDB())
    assert kb.chunker is None

  def test_custom_top_k(self):
    kb = Knowledge(vector_db=MockVectorDB(), top_k=10)
    assert kb.top_k == 10

  def test_custom_trigger(self):
    kb = Knowledge(vector_db=MockVectorDB(), trigger="auto")
    assert kb.trigger == "auto"

  def test_no_vectordb_creates_default(self):
    """Knowledge with no vector_db auto-creates InMemoryVectorDB."""
    kb = Knowledge()
    assert kb.vector_db is not None

  def test_embedder_passed_to_vectordb_if_vectordb_has_none(self):
    """If Knowledge has an embedder and vector_db.embedder is None, it's set."""
    mock_embedder = MagicMock()
    db = MockVectorDB(embedder=None)
    Knowledge(vector_db=db, embedder=mock_embedder)
    assert db.embedder is mock_embedder

  def test_embedder_not_overwritten_if_vectordb_has_one(self):
    """If vector_db already has an embedder, Knowledge's embedder doesn't overwrite it."""
    existing_embedder = MagicMock()
    new_embedder = MagicMock()
    db = MockVectorDB(embedder=existing_embedder)
    Knowledge(vector_db=db, embedder=new_embedder)
    assert db.embedder is existing_embedder


# ---------------------------------------------------------------------------
# Tests: _read_source dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeReadSource:
  """Knowledge._read_source dispatches based on input type."""

  def test_read_source_with_document_returns_list(self):
    kb = Knowledge(vector_db=MockVectorDB())
    doc = Document(content="hello")
    result = kb._read_source(doc, reader=None)
    assert result == [doc]

  def test_read_source_with_document_list_returns_same(self):
    kb = Knowledge(vector_db=MockVectorDB())
    docs = [Document(content="a"), Document(content="b")]
    result = kb._read_source(docs, reader=None)
    assert result == docs

  def test_read_source_with_string_creates_text_document(self):
    kb = Knowledge(vector_db=MockVectorDB())
    result = kb._read_source("raw text content", reader=None)
    assert len(result) == 1
    assert result[0].content == "raw text content"
    assert result[0].source_type == "text"


# ---------------------------------------------------------------------------
# Tests: format_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeFormatContext:
  """Knowledge.format_context outputs in the configured format."""

  def test_format_xml(self):
    kb = Knowledge(vector_db=MockVectorDB(), context_format="xml")
    docs = [Document(content="Hello world", name="doc1", source="test.txt")]
    result = kb.format_context(docs)
    assert "<knowledge_context>" in result
    assert "</knowledge_context>" in result
    assert "Hello world" in result

  def test_format_markdown(self):
    kb = Knowledge(vector_db=MockVectorDB(), context_format="markdown")
    docs = [Document(content="Hello world", name="doc1", source="test.txt")]
    result = kb.format_context(docs)
    assert "## Retrieved Context" in result
    assert "Hello world" in result
    assert "*Source: test.txt*" in result

  def test_format_json(self):
    kb = Knowledge(vector_db=MockVectorDB(), context_format="json")
    docs = [Document(content="Hello world", name="doc1")]
    result = kb.format_context(docs)
    import json

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["content"] == "Hello world"
    assert parsed[0]["name"] == "doc1"

  def test_format_xml_includes_reranking_score(self):
    kb = Knowledge(vector_db=MockVectorDB(), context_format="xml")
    docs = [Document(content="scored", reranking_score=0.95)]
    result = kb.format_context(docs)
    assert "relevance" in result
    assert "0.950" in result

  def test_format_empty_documents(self):
    kb = Knowledge(vector_db=MockVectorDB())
    result = kb.format_context([])
    assert "<knowledge_context>" in result
    assert "</knowledge_context>" in result


# ---------------------------------------------------------------------------
# Tests: extract_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeExtractQuery:
  """Knowledge.extract_query extracts queries from message lists."""

  def _make_message(self, role: str, content: str):
    """Create a simple message-like object."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg

  def test_last_user_message(self):
    kb = Knowledge(vector_db=MockVectorDB(), query_from="last_user")
    messages = [
      self._make_message("user", "first question"),
      self._make_message("assistant", "response"),
      self._make_message("user", "second question"),
    ]
    query = kb.extract_query(messages)
    assert query == "second question"

  def test_full_conversation(self):
    kb = Knowledge(vector_db=MockVectorDB(), query_from="full_conversation")
    messages = [
      self._make_message("user", "alpha"),
      self._make_message("assistant", "response"),
      self._make_message("user", "beta"),
    ]
    query = kb.extract_query(messages)
    assert query == "alpha beta"

  def test_no_user_message_returns_none(self):
    kb = Knowledge(vector_db=MockVectorDB(), query_from="last_user")
    messages = [self._make_message("assistant", "only assistant")]
    query = kb.extract_query(messages)
    assert query is None

  def test_max_query_length_truncates(self):
    kb = Knowledge(vector_db=MockVectorDB(), query_from="last_user", max_query_length=10)
    messages = [self._make_message("user", "a very long query that exceeds the limit")]
    query = kb.extract_query(messages)
    assert len(query) == 10  # type: ignore[arg-type]

  def test_empty_messages_returns_none(self):
    kb = Knowledge(vector_db=MockVectorDB(), query_from="last_user")
    query = kb.extract_query([])
    assert query is None


# ---------------------------------------------------------------------------
# Tests: add / search with mock VectorDB
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeAddAndSearch:
  """Knowledge.add() and search() with mock VectorDB."""

  def test_add_document_inserts(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    doc = Document(content="test content")
    kb.add(doc)
    assert db._created is True
    assert len(db._docs) == 1

  def test_add_document_list_inserts_all(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    docs = [Document(content="a"), Document(content="b"), Document(content="c")]
    kb.add(docs)
    assert len(db._docs) == 3

  def test_search_returns_documents(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    db._docs = [Document(content="result1"), Document(content="result2")]
    results = kb.search("query", top_k=5)
    assert len(results) == 2

  def test_search_respects_top_k(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    db._docs = [Document(content=f"doc{i}") for i in range(20)]
    results = kb.search("query", top_k=3)
    assert len(results) == 3

  def test_len_returns_vectordb_count(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    db._docs = [Document(content="a"), Document(content="b")]
    assert len(kb) == 2

  def test_clear_empties_vectordb(self):
    db = MockVectorDB()
    kb = Knowledge(vector_db=db)
    db._docs = [Document(content="a")]
    kb.clear()
    assert len(db._docs) == 0
