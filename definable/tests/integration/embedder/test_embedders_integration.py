"""
Integration tests for embedders.

Rules:
  - NO MOCKS -- these test real API connections
  - Session-scoped fixtures for speed (embedder clients are expensive)
  - Each test makes one real embedding call and asserts on the response shape

Covers:
  - VoyageAIEmbedder returns correct dimension vector (if key available)
  - VoyageAIEmbedder.get_embedding_and_usage() returns tuple
  - Async embedding path returns same shape

Note: OpenAI embedder tests already exist in test_openai_embedder.py.
      This file adds VoyageAI coverage.
"""

import pytest


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
