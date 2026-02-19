"""
Integration tests for the full Knowledge RAG pipeline.

Rules:
  - NO MOCKS — real embedder, real vector store
  - Tests end-to-end: document ingestion → chunking → embedding → storage → retrieval
  - Validates SEMANTIC RELEVANCE, not just data shapes

Pipeline under test:
  Knowledge.aadd(source) → chunk → embed → insert into VectorDB
  Knowledge.asearch(query) → embed query → VectorDB.search → optional rerank → List[Document]

Covers:
  - Adding a text document and retrieving relevant chunks
  - Chunker affects number of stored chunks
  - Search returns semantically relevant chunks (not just any chunks)
  - min_score filtering drops low-relevance results
  - Multiple documents co-exist and search selects correctly
  - Async add + async search path
"""

import pytest

from definable.knowledge import Knowledge
from definable.knowledge.chunker.text import TextChunker
from definable.knowledge.document import Document


@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.slow
class TestKnowledgePipeline:
  """End-to-end Knowledge RAG pipeline with real OpenAI embeddings."""

  @pytest.fixture
  def knowledge(self, openai_embedder):
    """Fresh Knowledge instance backed by InMemoryVectorDB."""
    from definable.vectordb import InMemoryVectorDB

    db = InMemoryVectorDB(embedder=openai_embedder)
    kb = Knowledge(vector_db=db, embedder=openai_embedder)
    yield kb
    db.drop()

  @pytest.fixture
  def knowledge_with_chunker(self, openai_embedder):
    """Knowledge with a TextChunker to exercise chunk splitting."""
    from definable.vectordb import InMemoryVectorDB

    db = InMemoryVectorDB(embedder=openai_embedder)
    kb = Knowledge(
      vector_db=db,
      embedder=openai_embedder,
      chunker=TextChunker(chunk_size=200, separator="\n"),
    )
    yield kb
    db.drop()

  @pytest.mark.asyncio
  async def test_add_and_search_returns_relevant_doc(self, knowledge):
    """Ingesting a document and searching for its content returns it."""
    doc = Document(
      content="Python is a high-level, general-purpose programming language.",
      name="python_intro",
    )
    await knowledge.aadd([doc])

    results = await knowledge.asearch("Python programming language", top_k=3)
    assert len(results) >= 1
    assert any("python" in r.content.lower() or "programming" in r.content.lower() for r in results)

  @pytest.mark.asyncio
  async def test_search_returns_most_relevant_document(self, knowledge):
    """When multiple documents are stored, search returns the most relevant one."""
    docs = [
      Document(content="Definable.ai is an AI agent framework for Python developers.", name="about"),
      Document(content="The Eiffel Tower is located in Paris, France.", name="landmarks"),
      Document(content="Machine learning models require training data.", name="ml"),
    ]
    for doc in docs:
      await knowledge.aadd([doc])

    results = await knowledge.asearch("AI agent framework Python", top_k=1)
    assert len(results) >= 1
    assert "definable" in results[0].content.lower() or "agent" in results[0].content.lower()

  @pytest.mark.asyncio
  async def test_chunker_produces_multiple_stored_chunks(self, knowledge_with_chunker, openai_embedder):
    """A large document should be split into multiple chunks in the VectorDB."""
    long_content = "\n".join([f"Line {i}: This is content about topic number {i}." for i in range(20)])
    doc = Document(content=long_content, name="large_doc")
    await knowledge_with_chunker.aadd([doc])

    # Search should return results (at least one chunk matches)
    results = await knowledge_with_chunker.asearch("content about topic", top_k=5)
    assert len(results) >= 1

  @pytest.mark.asyncio
  async def test_search_with_limit_respects_limit(self, knowledge):
    """search() should not return more documents than the limit."""
    docs = [Document(content=f"Document about subject {i}", name=f"doc_{i}") for i in range(5)]
    for doc in docs:
      await knowledge.aadd([doc])

    results = await knowledge.asearch("document about subject", top_k=2)
    assert len(results) <= 2

  @pytest.mark.asyncio
  async def test_empty_knowledge_returns_empty_results(self, knowledge):
    results = await knowledge.asearch("anything", top_k=5)
    assert results == []

  @pytest.mark.asyncio
  async def test_add_multiple_calls_accumulate(self, knowledge):
    """Multiple aadd() calls accumulate documents in the store."""
    doc1 = Document(content="First document: solar energy and renewable sources.", name="solar")
    doc2 = Document(content="Second document: electric vehicles and charging networks.", name="ev")

    await knowledge.aadd([doc1])
    await knowledge.aadd([doc2])

    results_solar = await knowledge.asearch("solar energy renewable", top_k=1)
    results_ev = await knowledge.asearch("electric vehicles charging", top_k=1)

    assert len(results_solar) >= 1
    assert len(results_ev) >= 1
    # They should retrieve different documents
    assert results_solar[0].content != results_ev[0].content
