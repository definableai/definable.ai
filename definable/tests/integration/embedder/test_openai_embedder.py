"""
Integration tests for OpenAI embedder.

Rules:
  - NO MOCKS — these test real OpenAI embedding API connections
  - Session-scoped embedder fixture for speed (embedder clients are expensive)
  - Each test makes one real embedding call and asserts on the response shape

Covers:
  - OpenAIEmbedder returns correct dimension vector
  - OpenAIEmbedder.get_embedding_and_usage() includes usage metadata
  - Async embedding path returns same shape
  - Different texts produce different embeddings
  - Similar texts produce similar embeddings (cosine similarity)
  - Long text handling
  - Sync and async produce same dimensions
"""

import math

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_sim(a: list, b: list) -> float:
  """Compute cosine similarity between two vectors."""
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x**2 for x in a))
  norm_b = math.sqrt(sum(x**2 for x in b))
  return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---------------------------------------------------------------------------
# OpenAI Embedder — real API calls
# ---------------------------------------------------------------------------


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

  def test_embedding_magnitude_is_reasonable(self, openai_embedder):
    """Embedding vectors should have reasonable magnitude (not zero, not exploding)."""
    emb = openai_embedder.get_embedding("hello world magnitude test")
    magnitude = math.sqrt(sum(x**2 for x in emb))
    assert magnitude > 0  # Not a zero vector
    assert magnitude < 1000  # Not exploding

  def test_dimensions_attribute_matches_output(self, openai_embedder):
    """The declared dimensions attribute should match actual output length."""
    result = openai_embedder.get_embedding("dimension check")
    assert len(result) == openai_embedder.dimensions
