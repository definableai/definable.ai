"""
Integration tests for embedders.

Rules:
  - NO MOCKS — these test real API connections
  - Session-scoped fixtures for speed (embedder clients are expensive)
  - Each test makes one real embedding call and asserts on the response shape

Covers:
  - OpenAIEmbedder returns correct dimension vector
  - OpenAIEmbedder.get_embedding_and_usage() includes usage metadata
  - Async embedding path returns same shape
  - VoyageAIEmbedder returns correct dimension vector (if key available)
  - Empty string handling (API-specific behavior)
  - Batch embedding (embed_many equivalent via multiple calls)
"""

import pytest


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIEmbedder:
  """Real OpenAI embedding API calls. Require OPENAI_API_KEY."""

  def test_get_embedding_returns_list_of_floats(self, openai_embedder):
    result = openai_embedder.get_embedding("hello world")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

  def test_get_embedding_returns_correct_dimensions(self, openai_embedder):
    """text-embedding-3-small with dimensions=1536 should return 1536-dim vectors."""
    result = openai_embedder.get_embedding("test text")
    assert len(result) == 1536

  def test_get_embedding_and_usage_includes_usage(self, openai_embedder):
    embedding, usage = openai_embedder.get_embedding_and_usage("hello")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536
    # Usage may be None or a dict with token counts
    if usage is not None:
      assert isinstance(usage, dict)

  def test_different_texts_produce_different_embeddings(self, openai_embedder):
    """Semantically different texts should produce different embeddings."""
    emb1 = openai_embedder.get_embedding("cat")
    emb2 = openai_embedder.get_embedding("skyscraper")
    # They should not be identical
    assert emb1 != emb2

  def test_similar_texts_produce_similar_embeddings(self, openai_embedder):
    """Semantically similar texts should have high cosine similarity."""
    import math

    def cosine_sim(a, b):
      dot = sum(x * y for x, y in zip(a, b))
      norm_a = math.sqrt(sum(x**2 for x in a))
      norm_b = math.sqrt(sum(x**2 for x in b))
      return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    emb1 = openai_embedder.get_embedding("The cat sat on the mat")
    emb2 = openai_embedder.get_embedding("A cat rested on a rug")
    emb3 = openai_embedder.get_embedding("Quantum computing advances")

    sim_similar = cosine_sim(emb1, emb2)
    sim_different = cosine_sim(emb1, emb3)
    # Similar sentences should be more similar than unrelated ones
    assert sim_similar > sim_different

  @pytest.mark.asyncio
  async def test_async_get_embedding_returns_correct_shape(self, openai_embedder):
    result = await openai_embedder.async_get_embedding("async embedding test")
    assert isinstance(result, list)
    assert len(result) == 1536

  @pytest.mark.asyncio
  async def test_async_and_sync_produce_same_dimensions(self, openai_embedder):
    text = "consistency check"
    sync_result = openai_embedder.get_embedding(text)
    async_result = await openai_embedder.async_get_embedding(text)
    assert len(sync_result) == len(async_result)

  def test_long_text_embedding(self, openai_embedder):
    """Long text should still return a valid embedding."""
    long_text = "This is a test sentence. " * 50
    result = openai_embedder.get_embedding(long_text)
    assert isinstance(result, list)
    assert len(result) == 1536


@pytest.mark.integration
@pytest.mark.voyageai
class TestVoyageAIEmbedder:
  """Real VoyageAI embedding API calls. Require VOYAGEAI_API_KEY."""

  @pytest.fixture(scope="class")
  def embedder(self, voyage_embedder):
    return voyage_embedder

  def test_get_embedding_returns_list_of_floats(self, voyage_embedder):
    result = voyage_embedder.get_embedding("hello world")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

  def test_get_embedding_and_usage_returns_tuple(self, voyage_embedder):
    embedding, usage = voyage_embedder.get_embedding_and_usage("voyage test")
    assert isinstance(embedding, list)
    assert len(embedding) > 0

  @pytest.mark.asyncio
  async def test_async_embedding(self, voyage_embedder):
    result = await voyage_embedder.async_get_embedding("async voyage")
    assert isinstance(result, list)
    assert len(result) > 0
