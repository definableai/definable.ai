"""Tests for search provider backends (with mocking)."""

import pytest

from definable.research.search import CallableSearchProvider, create_search_provider
from definable.research.search.base import SearchProvider, SearchResult


class TestCreateSearchProvider:
  """Test the search provider factory."""

  def test_create_duckduckgo(self):
    provider = create_search_provider("duckduckgo")
    assert isinstance(provider, SearchProvider)

  def test_create_unknown_raises(self):
    with pytest.raises(ValueError, match="Unknown search provider"):
      create_search_provider("nonexistent")

  def test_create_google_without_config_raises(self):
    with pytest.raises(ValueError, match="api_key"):
      create_search_provider("google", config={})

  def test_create_serpapi_without_config_raises(self):
    with pytest.raises(ValueError, match="api_key"):
      create_search_provider("serpapi", config={})

  def test_create_google_with_config(self):
    provider = create_search_provider("google", config={"api_key": "test", "cse_id": "test"})
    assert isinstance(provider, SearchProvider)

  def test_create_serpapi_with_config(self):
    provider = create_search_provider("serpapi", config={"api_key": "test"})
    assert isinstance(provider, SearchProvider)


@pytest.mark.asyncio
class TestCallableSearchProvider:
  """Test the custom callable adapter."""

  async def test_callable_provider(self):
    async def my_search(query: str, max_results: int = 10):
      return [SearchResult(title="Test", url="https://test.com", snippet="Test result")]

    provider = CallableSearchProvider(my_search)
    results = await provider.search("test query")
    assert len(results) == 1
    assert results[0].title == "Test"

  async def test_callable_provider_passes_args(self):
    received: dict[str, object] = {}

    async def my_search(query: str, max_results: int = 10):
      received["query"] = query
      received["max_results"] = max_results
      return []

    provider = CallableSearchProvider(my_search)
    await provider.search("hello", 5)
    assert received["query"] == "hello"
    assert received["max_results"] == 5


@pytest.mark.asyncio
class TestMockSearchProvider:
  """Test the mock search provider from conftest."""

  async def test_mock_returns_results(self, mock_search):
    results = await mock_search.search("quantum computing")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)

  async def test_mock_tracks_calls(self, mock_search):
    await mock_search.search("query 1")
    await mock_search.search("query 2")
    assert mock_search.call_count == 2
    assert mock_search.queries == ["query 1", "query 2"]
