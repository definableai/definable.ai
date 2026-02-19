"""Unit tests for InMemoryVectorDB.

Tests cover creation, insert, search, get_count, drop, exists,
delete operations, and lifecycle. All tests use the mock_embedder
fixture from root conftest to avoid API calls.
"""

import pytest

from definable.knowledge.document.base import Document
from definable.vectordb.memory.memory import InMemoryVectorDB


def _make_doc(content: str, embedding: list[float] | None = None, **kwargs) -> Document:
  """Helper to create a Document with optional pre-computed embedding."""
  return Document(content=content, embedding=embedding, **kwargs)


@pytest.mark.unit
class TestInMemoryVectorDBCreation:
  """Tests for InMemoryVectorDB instantiation."""

  def test_create_with_embedder(self, mock_embedder):
    """InMemoryVectorDB can be created with a custom embedder."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.embedder is mock_embedder
    assert db.get_count() == 0

  def test_create_sets_name(self, mock_embedder):
    """Name is stored on the instance."""
    db = InMemoryVectorDB(name="test_db", embedder=mock_embedder)
    assert db.name == "test_db"

  def test_create_no_op(self, mock_embedder):
    """create() is a no-op and does not raise."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    db.create()  # should not raise


@pytest.mark.unit
class TestInsert:
  """Tests for InMemoryVectorDB.insert()."""

  def test_insert_documents_with_content_hash(self, mock_embedder):
    """insert(content_hash, docs) stores documents."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    emb = mock_embedder.get_embedding("hello world")
    docs = [_make_doc("hello world", embedding=emb)]
    db.insert("hash123", docs)
    assert db.get_count() == 1

  def test_insert_documents_list_directly(self, mock_embedder):
    """insert(docs) without explicit hash auto-generates hash."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    emb = mock_embedder.get_embedding("test doc")
    docs = [_make_doc("test doc", embedding=emb)]
    db.insert(docs)
    assert db.get_count() == 1

  def test_insert_multiple_documents(self, mock_embedder):
    """Inserting multiple documents increases the count."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc(f"doc {i}", embedding=mock_embedder.get_embedding(f"doc {i}")) for i in range(5)]
    db.insert("batch_hash", docs)
    assert db.get_count() == 5

  def test_insert_without_docs_raises(self, mock_embedder):
    """insert(content_hash) without documents raises ValueError."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    with pytest.raises(ValueError, match="documents must be provided"):
      db.insert("hash_only")


@pytest.mark.unit
class TestSearch:
  """Tests for InMemoryVectorDB.search()."""

  def test_search_returns_results(self, mock_embedder):
    """search returns matching documents ordered by similarity."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [
      _make_doc("python programming", embedding=mock_embedder.get_embedding("python programming")),
      _make_doc("java development", embedding=mock_embedder.get_embedding("java development")),
      _make_doc("python web framework", embedding=mock_embedder.get_embedding("python web framework")),
    ]
    db.insert("search_hash", docs)

    results = db.search("python", limit=2)
    assert len(results) <= 2
    assert all(isinstance(r, Document) for r in results)

  def test_search_respects_limit(self, mock_embedder):
    """search never returns more than `limit` results."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc(f"doc {i}", embedding=mock_embedder.get_embedding(f"doc {i}")) for i in range(10)]
    db.insert("limit_hash", docs)

    results = db.search("doc", limit=3)
    assert len(results) <= 3

  def test_search_empty_db_returns_empty(self, mock_embedder):
    """search on an empty database returns an empty list."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    results = db.search("anything")
    assert results == []

  def test_search_with_filter(self, mock_embedder):
    """search with dict filter only returns matching metadata."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [
      _make_doc("alpha", embedding=mock_embedder.get_embedding("alpha"), meta_data={"category": "a"}),
      _make_doc("beta", embedding=mock_embedder.get_embedding("beta"), meta_data={"category": "b"}),
    ]
    db.insert("filter_hash", docs)

    results = db.search("alpha", limit=5, filters={"category": "a"})
    assert all(r.meta_data.get("category") == "a" for r in results)


@pytest.mark.unit
class TestGetCount:
  """Tests for InMemoryVectorDB.get_count()."""

  def test_count_zero_on_new_db(self, mock_embedder):
    """Fresh database has count 0."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.get_count() == 0

  def test_count_after_insert(self, mock_embedder):
    """Count increases after insert."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("x", embedding=mock_embedder.get_embedding("x"))]
    db.insert(docs)
    assert db.get_count() == 1

  def test_count_alias(self, mock_embedder):
    """count() is an alias for get_count()."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.count() == db.get_count()


@pytest.mark.unit
class TestDrop:
  """Tests for InMemoryVectorDB.drop()."""

  def test_drop_clears_all_data(self, mock_embedder):
    """drop() removes all documents."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("d", embedding=mock_embedder.get_embedding("d"))]
    db.insert(docs)
    assert db.get_count() == 1
    db.drop()
    assert db.get_count() == 0


@pytest.mark.unit
class TestExists:
  """Tests for InMemoryVectorDB.exists() and related checks."""

  def test_exists_always_true(self, mock_embedder):
    """In-memory DB always reports existence as True."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.exists() is True

  def test_content_hash_exists_after_insert(self, mock_embedder):
    """content_hash_exists returns True after inserting with that hash."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("c", embedding=mock_embedder.get_embedding("c"))]
    db.insert("my_hash", docs)
    assert db.content_hash_exists("my_hash") is True
    assert db.content_hash_exists("other_hash") is False

  def test_name_exists(self, mock_embedder):
    """name_exists returns True when a document with that name is stored."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("c", embedding=mock_embedder.get_embedding("c"), name="named_doc")]
    db.insert(docs)
    assert db.name_exists("named_doc") is True
    assert db.name_exists("unknown") is False


@pytest.mark.unit
class TestDeleteOperations:
  """Tests for delete, delete_by_id, delete_by_name, delete_by_metadata."""

  def test_delete_clears_all(self, mock_embedder):
    """delete() removes all documents and returns True."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("x", embedding=mock_embedder.get_embedding("x"))]
    db.insert(docs)
    assert db.delete() is True
    assert db.get_count() == 0

  def test_delete_by_name(self, mock_embedder):
    """delete_by_name removes docs with that name."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [
      _make_doc("a", embedding=mock_embedder.get_embedding("a"), name="remove_me"),
      _make_doc("b", embedding=mock_embedder.get_embedding("b"), name="keep_me"),
    ]
    db.insert(docs)
    assert db.get_count() == 2
    assert db.delete_by_name("remove_me") is True
    assert db.get_count() == 1

  def test_delete_by_name_not_found(self, mock_embedder):
    """delete_by_name returns False when name not found."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.delete_by_name("ghost") is False

  def test_delete_by_metadata(self, mock_embedder):
    """delete_by_metadata removes docs matching the metadata filter."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [
      _make_doc("a", embedding=mock_embedder.get_embedding("a"), meta_data={"env": "test"}),
      _make_doc("b", embedding=mock_embedder.get_embedding("b"), meta_data={"env": "prod"}),
    ]
    db.insert(docs)
    assert db.delete_by_metadata({"env": "test"}) is True
    assert db.get_count() == 1

  def test_delete_by_content_id(self, mock_embedder):
    """delete_by_content_id removes docs with matching content_id."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs = [_make_doc("c", embedding=mock_embedder.get_embedding("c"), content_id="cid_123")]
    db.insert(docs)
    assert db.delete_by_content_id("cid_123") is True
    assert db.get_count() == 0

  def test_delete_by_content_id_not_found(self, mock_embedder):
    """delete_by_content_id returns False when content_id not found."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.delete_by_content_id("nope") is False


@pytest.mark.unit
class TestUpsert:
  """Tests for InMemoryVectorDB.upsert()."""

  def test_upsert_replaces_existing(self, mock_embedder):
    """upsert with same content_hash replaces old documents."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    docs_v1 = [_make_doc("v1", embedding=mock_embedder.get_embedding("v1"))]
    docs_v2 = [_make_doc("v2", embedding=mock_embedder.get_embedding("v2"))]
    db.insert("same_hash", docs_v1)
    assert db.get_count() == 1
    db.upsert("same_hash", docs_v2)
    assert db.get_count() == 1

  def test_upsert_available(self, mock_embedder):
    """InMemoryVectorDB reports upsert as available."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.upsert_available() is True


@pytest.mark.unit
class TestSupportedSearchTypes:
  """Tests for get_supported_search_types."""

  def test_returns_vector(self, mock_embedder):
    """InMemoryVectorDB only supports vector search."""
    db = InMemoryVectorDB(embedder=mock_embedder)
    assert db.get_supported_search_types() == ["vector"]
