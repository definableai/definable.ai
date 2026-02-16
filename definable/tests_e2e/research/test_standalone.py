"""Tests for standalone DeepResearch usage (outside of Agent)."""

import pytest

from definable.research import DeepResearch, ResearchResult


@pytest.mark.asyncio
class TestStandaloneDeepResearch:
  """Test DeepResearch used independently (no Agent)."""

  async def test_standalone_arun(self, mock_model, mock_search, research_config):
    researcher = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await researcher.arun("What is quantum computing?")
    assert isinstance(result, ResearchResult)
    assert result.context

  async def test_standalone_with_custom_search(self, mock_model, research_config):
    from definable.research.search import CallableSearchProvider
    from definable.research.search.base import SearchResult

    async def custom_search(query: str, max_results: int = 10):
      return [SearchResult(title="Custom Result", url="https://custom.com", snippet="Custom")]

    provider = CallableSearchProvider(custom_search)
    researcher = DeepResearch(
      model=mock_model,
      search_provider=provider,
      config=research_config,
    )
    result = await researcher.arun("test")
    assert isinstance(result, ResearchResult)

  async def test_standalone_needs_research(self, mock_model, mock_search, research_config):
    researcher = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    needs = await researcher.needs_research("What are the latest AI developments?")
    assert isinstance(needs, bool)

  async def test_standalone_sub_questions_populated(self, mock_model, mock_search, research_config):
    researcher = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await researcher.arun("quantum computing overview")
    assert len(result.sub_questions) > 0

  async def test_standalone_metrics_populated(self, mock_model, mock_search, research_config):
    researcher = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await researcher.arun("quantum computing")
    assert result.metrics.total_time_ms > 0
    assert result.metrics.waves_executed >= 1
