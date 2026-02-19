"""
Integration tests for vector database backends.

Rules:
  - NO MOCKS — tests real insert/search/delete operations
  - InMemoryVectorDB uses a REAL embedder (OpenAI) for search
  - Each test fixture creates a fresh store and drops it after
  - Tests validate actual semantic search behavior, not just API shapes

Covers:
  - insert() stores documents and search() retrieves them
  - search() returns semantically relevant results
  - delete_by_name() removes documents
  - drop() clears all documents
  - content_hash_exists() tracks inserted content
  - upsert() is idempotent (no duplicate content)
  - filter by metadata
  - count() reflects actual stored document count
"""

import pytest

from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embedded_doc(content: str, embedding: list, name: str = None, meta_data: dict = None) -> Document:  # type: ignore[assignment]
  """Create a pre-embedded document for insertion without API calls."""
  return Document(
    content=content,
    name=name or content[:20],
    embedding=embedding,
    meta_data=meta_data or {},
  )


def unit_vector(dim: int, index: int) -> list:
  """Create a unit vector with 1.0 at position `index`, 0 elsewhere."""
  v = [0.0] * dim
  v[index] = 1.0
  return v


# ---------------------------------------------------------------------------
# InMemoryVectorDB — uses real OpenAI embedder for search queries
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestInMemoryVectorDB:
  """
  InMemoryVectorDB with real OpenAI embedder for search.

  Documents are pre-embedded with synthetic unit vectors (no API cost for inserts).
  Only search() calls the embedder (one API call per search query).
  """

  @pytest.fixture
  def db(self, openai_embedder):
    from definable.vectordb import InMemoryVectorDB

    store = InMemoryVectorDB(embedder=openai_embedder)
    yield store
    store.drop()

  def test_insert_and_search_returns_results(self, db, openai_embedder):
    """After inserting a document, search should return it."""
    embedding = openai_embedder.get_embedding("machine learning algorithms")
    doc = make_embedded_doc("Machine learning algorithms for classification", embedding)
    db.insert([doc])

    results = db.search("machine learning", limit=1)
    assert len(results) >= 1
    assert any("machine learning" in r.content.lower() or "classification" in r.content.lower() for r in results)

  def test_search_returns_most_relevant_result_first(self, db, openai_embedder):
    """The most semantically similar document should rank first."""
    # Insert two very different documents
    emb_python = openai_embedder.get_embedding("Python programming language features")
    emb_cooking = openai_embedder.get_embedding("How to bake chocolate cake recipe")

    db.insert([make_embedded_doc("Python programming language features", emb_python, name="python_doc")])
    db.insert([make_embedded_doc("How to bake chocolate cake recipe", emb_cooking, name="cooking_doc")])

    results = db.search("programming in Python", limit=2)
    assert len(results) >= 1
    # Python doc should be ranked first for a Python query
    assert "python" in results[0].content.lower() or "programming" in results[0].content.lower()

  def test_insert_multiple_documents(self, db, openai_embedder):
    """Multiple documents can be inserted in one batch."""
    texts = ["First document about cats", "Second document about dogs", "Third document about birds"]
    docs = [make_embedded_doc(t, openai_embedder.get_embedding(t)) for t in texts]
    db.insert(docs)
    assert db.get_count() == 3

  def test_count_reflects_insertions(self, db, openai_embedder):
    assert db.get_count() == 0
    emb = openai_embedder.get_embedding("test doc")
    db.insert([make_embedded_doc("test doc", emb)])
    assert db.get_count() == 1

  def test_drop_clears_all_documents(self, db, openai_embedder):
    emb = openai_embedder.get_embedding("temp doc")
    db.insert([make_embedded_doc("temp doc", emb)])
    assert db.get_count() > 0
    db.drop()
    assert db.get_count() == 0

  def test_delete_by_name_removes_document(self, db, openai_embedder):
    emb = openai_embedder.get_embedding("deletable document")
    doc = make_embedded_doc("deletable document", emb, name="to_delete")
    db.insert([doc])
    assert db.name_exists("to_delete")
    db.delete_by_name("to_delete")
    assert not db.name_exists("to_delete")

  def test_content_hash_exists_after_insert(self, db, openai_embedder):
    """Content hash should be tracked after insert(content_hash, documents) call."""
    emb = openai_embedder.get_embedding("hash tracked doc")
    doc = make_embedded_doc("hash tracked doc", emb)
    content_hash = "test_hash_abc123"
    db.insert(content_hash, [doc])
    assert db.content_hash_exists(content_hash)

  def test_upsert_replaces_existing_content(self, db, openai_embedder):
    """Upsert with same content_hash should update, not duplicate."""
    emb = openai_embedder.get_embedding("upsert test document")
    doc_v1 = make_embedded_doc("Version 1 content", emb, name="doc_v1")
    doc_v2 = make_embedded_doc("Version 2 content", emb, name="doc_v2")
    content_hash = "upsert_hash"

    db.insert(content_hash, [doc_v1])
    count_after_v1 = db.get_count()

    db.upsert(content_hash, [doc_v2])
    count_after_v2 = db.get_count()

    # Should not duplicate — count stays the same (or changes by at most doc diff)
    assert count_after_v2 == count_after_v1

  def test_filter_by_metadata(self, db, openai_embedder):
    """Search with metadata filters should only return matching documents."""
    emb = openai_embedder.get_embedding("filtered document")
    doc_pdf = make_embedded_doc("PDF source document", emb, meta_data={"source": "pdf"})
    doc_url = make_embedded_doc("URL source document", emb, meta_data={"source": "url"})

    db.insert([doc_pdf, doc_url])

    results = db.search("source document", limit=10, filters={"source": "pdf"})
    assert all(r.meta_data.get("source") == "pdf" for r in results)

  def test_empty_store_returns_empty_search(self, db):
    results = db.search("anything", limit=5)
    assert results == []

  def test_exists_always_true_for_in_memory(self, db):
    assert db.exists() is True

  @pytest.mark.asyncio
  async def test_async_insert_and_search(self, db, openai_embedder):
    emb = openai_embedder.get_embedding("async document")
    doc = make_embedded_doc("Async document content", emb)
    await db.async_insert([doc])
    results = await db.async_search("async document", limit=1)
    assert len(results) >= 1

  def test_delete_all_clears_store(self, db, openai_embedder):
    emb = openai_embedder.get_embedding("doc to delete all")
    db.insert([make_embedded_doc("doc to delete all", emb)])
    db.delete()
    assert db.get_count() == 0
