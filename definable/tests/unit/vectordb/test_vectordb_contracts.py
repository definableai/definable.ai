"""
Contract tests: Every VectorDB implementation must satisfy these.

The VectorDB ABC defines the contract:
  - create() / async_create()
  - exists() / async_exists()
  - insert(content_hash_or_docs, documents) / async_insert()
  - search(query, limit) -> List[Document] / async_search()
  - delete_by_name(name) -> bool
  - delete_by_metadata(metadata) -> bool
  - drop() / async_drop()
  - get_count() -> int
  - content_hash_exists(hash) -> bool

InMemoryVectorDB tests run without external deps (uses HashEmbedder for search).
External stores (Qdrant, PgVector, etc.) are gated by provider marks.

To add a new VectorDB: inherit VectorDBContractTests and provide a `db` fixture.
"""

import pytest

from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------


class VectorDBContractTests:
  """
  Abstract contract test suite for all VectorDB implementations.

  Every concrete VectorDB must pass ALL tests in this class.
  """

  @pytest.fixture
  def db(self):
    """Provide a fresh VectorDB instance, cleaned up after each test."""
    raise NotImplementedError("Subclass must provide a db fixture")

  @pytest.fixture
  def sample_doc_with_embedding(self):
    """A pre-embedded document for insertion without API calls."""
    return Document(
      content="Sample document about machine learning.",
      name="ml_doc",
      embedding=[0.1] * 128,  # Match HashEmbedder dimensions
      meta_data={"source": "test"},
    )

  # --- Contract: lifecycle ---

  @pytest.mark.contract
  def test_create_does_not_raise(self, db):
    db.create()  # Should not raise

  @pytest.mark.contract
  def test_exists_returns_bool(self, db):
    result = db.exists()
    assert isinstance(result, bool)

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_async_exists_returns_bool(self, db):
    result = await db.async_exists()
    assert isinstance(result, bool)

  # --- Contract: count ---

  @pytest.mark.contract
  def test_get_count_starts_at_zero(self, db):
    assert db.get_count() == 0

  @pytest.mark.contract
  def test_get_count_increases_after_insert(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    assert db.get_count() == 1

  # --- Contract: insert ---

  @pytest.mark.contract
  def test_insert_with_content_hash_does_not_raise(self, db, sample_doc_with_embedding):
    db.insert("hash_abc", [sample_doc_with_embedding])

  @pytest.mark.contract
  def test_insert_without_content_hash_does_not_raise(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_async_insert_does_not_raise(self, db, sample_doc_with_embedding):
    await db.async_insert([sample_doc_with_embedding])

  # --- Contract: content_hash_exists ---

  @pytest.mark.contract
  def test_content_hash_exists_after_insert(self, db, sample_doc_with_embedding):
    content_hash = "test_hash_contract_xyz"
    db.insert(content_hash, [sample_doc_with_embedding])
    assert db.content_hash_exists(content_hash)

  @pytest.mark.contract
  def test_content_hash_not_exists_before_insert(self, db):
    assert not db.content_hash_exists("never_inserted_hash")

  # --- Contract: delete ---

  @pytest.mark.contract
  def test_delete_by_name_returns_bool(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    result = db.delete_by_name("ml_doc")
    assert isinstance(result, bool)

  @pytest.mark.contract
  def test_delete_by_name_removes_document(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    assert db.name_exists("ml_doc")
    db.delete_by_name("ml_doc")
    assert not db.name_exists("ml_doc")

  @pytest.mark.contract
  def test_delete_by_metadata_returns_bool(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    result = db.delete_by_metadata({"source": "test"})
    assert isinstance(result, bool)

  # --- Contract: drop ---

  @pytest.mark.contract
  def test_drop_clears_all_documents(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    assert db.get_count() > 0
    db.drop()
    assert db.get_count() == 0

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_async_drop_clears_all_documents(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    await db.async_drop()
    assert db.get_count() == 0

  # --- Contract: get_supported_search_types ---

  @pytest.mark.contract
  def test_get_supported_search_types_returns_list(self, db):
    result = db.get_supported_search_types()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(t, str) for t in result)


# ---------------------------------------------------------------------------
# InMemoryVectorDB — uses HashEmbedder (no API key needed)
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestInMemoryVectorDBContract(VectorDBContractTests):
  """InMemoryVectorDB satisfies the VectorDB contract."""

  @pytest.fixture
  def db(self, mock_embedder):
    from definable.vectordb import InMemoryVectorDB

    store = InMemoryVectorDB(embedder=mock_embedder)
    yield store
    store.drop()

  @pytest.mark.contract
  def test_search_returns_list_of_documents(self, db, sample_doc_with_embedding):
    """search() returns List[Document] — InMemoryVectorDB uses embedder for query."""
    db.insert([sample_doc_with_embedding])
    results = db.search("machine learning", limit=5)
    assert isinstance(results, list)
    assert all(isinstance(r, Document) for r in results)

  @pytest.mark.contract
  def test_search_empty_store_returns_empty(self, db):
    results = db.search("anything", limit=5)
    assert results == []

  @pytest.mark.contract
  def test_name_exists_returns_bool(self, db, sample_doc_with_embedding):
    db.insert([sample_doc_with_embedding])
    assert isinstance(db.name_exists("ml_doc"), bool)
    assert db.name_exists("ml_doc") is True
    assert db.name_exists("does_not_exist") is False
