"""Tests for the DeepResearch engine (full pipeline with mocks)."""

import pytest

from definable.research.config import DeepResearchConfig
from definable.research.engine import DeepResearch
from definable.research.models import ResearchResult


@pytest.mark.asyncio
class TestDeepResearchEngine:
  """Test the full research pipeline with mocked components."""

  async def test_basic_run(self, mock_model, mock_search, research_config):
    """Full pipeline run with mocked search and model."""
    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await engine.arun("What are the latest quantum computing developments?")

    assert isinstance(result, ResearchResult)
    assert result.context  # Should have synthesized context
    assert len(result.sub_questions) > 0
    assert result.metrics.waves_executed >= 1

  async def test_returns_sources(self, mock_model, mock_search, research_config):
    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await engine.arun("quantum computing")
    # May or may not have sources depending on mock page read success
    assert isinstance(result.sources, list)

  async def test_returns_metrics(self, mock_model, mock_search, research_config):
    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    result = await engine.arun("quantum computing")
    assert result.metrics.total_time_ms > 0
    assert result.metrics.waves_executed >= 1

  async def test_depth_quick(self, mock_model, mock_search):
    config = DeepResearchConfig(depth="quick", min_relevance=0.0)
    engine = DeepResearch(model=mock_model, search_provider=mock_search, config=config)
    result = await engine.arun("test query")
    assert result.metrics.waves_executed <= 1

  async def test_compression_model_separate(self, mock_model, mock_search, research_config):
    """Test using a separate compression model."""
    from tests_e2e.research.conftest import ResearchMockModel

    compression_mock = ResearchMockModel()

    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      compression_model=compression_mock,
      config=research_config,
    )
    result = await engine.arun("quantum")
    assert isinstance(result, ResearchResult)

  async def test_progress_callback(self, mock_model, mock_search, research_config):
    """Test that progress callback is invoked."""
    progress_calls = []

    def on_progress(wave, sources_read, facts_extracted, gaps_remaining, message):
      progress_calls.append({
        "wave": wave,
        "sources_read": sources_read,
        "facts_extracted": facts_extracted,
        "message": message,
      })

    engine = DeepResearch(
      model=mock_model,
      search_provider=mock_search,
      config=research_config,
    )
    await engine.arun("quantum computing", on_progress=on_progress)
    assert len(progress_calls) >= 1
    assert progress_calls[0]["wave"] == 1

  async def test_needs_research_true(self, mock_model, mock_search, research_config):
    engine = DeepResearch(model=mock_model, search_provider=mock_search, config=research_config)
    result = await engine.needs_research("What are the latest developments in quantum computing?")
    assert result is True

  async def test_empty_search_results(self, mock_model, research_config):
    """Test graceful handling when search returns no results."""
    from tests_e2e.research.conftest import MockSearchProvider

    empty_search = MockSearchProvider(results=[])
    engine = DeepResearch(model=mock_model, search_provider=empty_search, config=research_config)
    result = await engine.arun("query with no results")
    assert isinstance(result, ResearchResult)
    assert result.metrics.total_sources_found == 0


@pytest.mark.asyncio
class TestDeepResearchConfig:
  """Test config depth presets."""

  async def test_quick_preset(self):
    config = DeepResearchConfig(depth="quick").with_depth_preset()
    assert config.max_waves == 1
    assert config.max_sources == 8

  async def test_standard_preset(self):
    config = DeepResearchConfig(depth="standard").with_depth_preset()
    assert config.max_waves == 3
    assert config.max_sources == 15

  async def test_deep_preset(self):
    config = DeepResearchConfig(depth="deep").with_depth_preset()
    assert config.max_waves == 5
    assert config.max_sources == 30

  async def test_custom_overrides_preset(self):
    config = DeepResearchConfig(depth="deep", max_waves=2).with_depth_preset()
    assert config.max_waves == 2  # User override preserved
    assert config.max_sources == 30  # Preset applied
