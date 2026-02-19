"""
Contract tests: Every Embedder implementation must satisfy these.

The Embedder ABC defines the contract:
  - get_embedding(text) → List[float]
  - get_embedding_and_usage(text) → (List[float], Optional[dict])
  - async_get_embedding(text) → List[float]
  - async_get_embedding_and_usage(text) → (List[float], Optional[dict])
  - Dimensions match the declared `dimensions` attribute
  - Embedding is a non-empty list of floats

To add a new Embedder: inherit this class and provide an `embedder` fixture.

Real API calls. Requires API keys.
"""

import math

import pytest


# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------


class EmbedderContractTests:
  """
  Abstract contract test suite for all Embedder implementations.

  Every concrete Embedder must pass ALL tests in this class.
  """

  @pytest.fixture
  def embedder(self):
    raise NotImplementedError("Subclass must provide an embedder fixture")

  @pytest.mark.contract
  def test_get_embedding_returns_list(self, embedder):
    result = embedder.get_embedding("hello")
    assert isinstance(result, list)

  @pytest.mark.contract
  def test_get_embedding_returns_floats(self, embedder):
    result = embedder.get_embedding("hello")
    assert all(isinstance(x, float) for x in result)

  @pytest.mark.contract
  def test_get_embedding_non_empty(self, embedder):
    result = embedder.get_embedding("hello")
    assert len(result) > 0

  @pytest.mark.contract
  def test_get_embedding_dimensions_match_declared(self, embedder):
    """Embedding dimension must match the declared `dimensions` attribute."""
    result = embedder.get_embedding("test")
    assert len(result) == embedder.dimensions

  @pytest.mark.contract
  def test_get_embedding_and_usage_returns_tuple(self, embedder):
    embedding, usage = embedder.get_embedding_and_usage("hello")
    assert isinstance(embedding, list)
    # usage may be None or a dict
    assert usage is None or isinstance(usage, dict)

  @pytest.mark.contract
  def test_get_embedding_and_usage_embedding_has_correct_dims(self, embedder):
    embedding, _ = embedder.get_embedding_and_usage("test")
    assert len(embedding) == embedder.dimensions

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_async_get_embedding_returns_list(self, embedder):
    result = await embedder.async_get_embedding("async test")
    assert isinstance(result, list)
    assert len(result) == embedder.dimensions

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_async_get_embedding_and_usage_returns_tuple(self, embedder):
    embedding, usage = await embedder.async_get_embedding_and_usage("async test")
    assert isinstance(embedding, list)
    assert len(embedding) == embedder.dimensions

  @pytest.mark.contract
  def test_different_texts_produce_different_embeddings(self, embedder):
    emb1 = embedder.get_embedding("python programming")
    emb2 = embedder.get_embedding("chocolate cake recipe")
    # Embeddings should be different for different texts
    assert emb1 != emb2

  @pytest.mark.contract
  def test_same_text_produces_consistent_embedding_shape(self, embedder):
    emb1 = embedder.get_embedding("consistent text")
    emb2 = embedder.get_embedding("consistent text")
    # Dimensions must be the same across calls
    assert len(emb1) == len(emb2)

  @pytest.mark.contract
  def test_embedding_is_unit_length_or_reasonable_magnitude(self, embedder):
    """Embedding vectors should have reasonable magnitude (not zero, not exploding)."""
    emb = embedder.get_embedding("hello world")
    magnitude = math.sqrt(sum(x**2 for x in emb))
    assert magnitude > 0  # Not a zero vector
    assert magnitude < 1000  # Not exploding


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIEmbedderContract(EmbedderContractTests):
  """OpenAIEmbedder satisfies the Embedder contract."""

  @pytest.fixture
  def embedder(self, openai_embedder):
    return openai_embedder


@pytest.mark.integration
@pytest.mark.voyageai
class TestVoyageAIEmbedderContract(EmbedderContractTests):
  """VoyageAIEmbedder satisfies the Embedder contract."""

  @pytest.fixture
  def embedder(self, voyage_embedder):
    return voyage_embedder
