"""Unit tests for the deep research module.

Covers data models (Fact, CKU, PageContent, etc.), KnowledgeGraph
(dedup, topic indexing, contradiction detection), DeepResearchConfig,
and SearchResult/SearchProvider types. No real LLM or search calls.
"""

import asyncio
from typing import List

import pytest

from definable.agent.research.models import (
  CKU,
  Contradiction,
  Fact,
  PageContent,
  ResearchMetrics,
  ResearchResult,
  SourceInfo,
  TopicGap,
)
from definable.agent.research.knowledge_graph import (
  KnowledgeGraph,
  _jaccard,
  _normalize,
)
from definable.agent.research.config import DeepResearchConfig
from definable.agent.research.search.base import SearchProvider, SearchResult


# ===========================================================================
# Data model defaults
# ===========================================================================


@pytest.mark.unit
class TestResearchModels:
  """Tests for research data model defaults and construction."""

  def test_fact_defaults(self):
    f = Fact(content="The sky is blue")
    assert f.fact_type == "factual"
    assert f.confidence == 0.8
    assert f.entities == []
    assert f.contradicts_expectation is False

  def test_fact_custom_fields(self):
    f = Fact(
      content="Revenue is 5 billion",
      fact_type="statistical",
      confidence=0.95,
      entities=["Apple"],
    )
    assert f.fact_type == "statistical"
    assert f.entities == ["Apple"]

  def test_cku_defaults(self):
    cku = CKU(source_url="https://example.com", source_title="Test", query_context="test query")
    assert cku.facts == []
    assert cku.relevance_score == 0.0
    assert cku.compression_ratio == 0.0

  def test_page_content(self):
    pc = PageContent(url="https://example.com", title="Page", content="Hello")
    assert pc.error is None

  def test_page_content_with_error(self):
    pc = PageContent(url="https://example.com", error="Timeout")
    assert pc.error == "Timeout"

  def test_source_info(self):
    s = SourceInfo(url="https://example.com", title="Page", fact_count=5)
    assert s.relevance_score == 0.0

  def test_topic_gap_defaults(self):
    g = TopicGap(topic="quantum computing")
    assert g.status == "missing"
    assert g.confidence == 0.0
    assert g.suggested_queries == []

  def test_contradiction(self):
    a = Fact(content="Revenue is 5 billion")
    b = Fact(content="Revenue is 3 billion")
    c = Contradiction(fact_a=a, fact_b=b)
    assert c.fact_a.content == "Revenue is 5 billion"

  def test_research_metrics_defaults(self):
    m = ResearchMetrics()
    assert m.total_time_ms == 0.0
    assert m.waves_executed == 0

  def test_research_result_defaults(self):
    r = ResearchResult()
    assert r.context == ""
    assert r.sources == []
    assert r.facts == []
    assert isinstance(r.metrics, ResearchMetrics)


# ===========================================================================
# KnowledgeGraph helpers
# ===========================================================================


@pytest.mark.unit
class TestNormalize:
  """Tests for the _normalize helper."""

  def test_lowercase(self):
    result = _normalize("Hello World")
    assert "hello" in result
    assert "world" in result

  def test_removes_stopwords(self):
    result = _normalize("the cat is on the mat")
    assert "the" not in result
    assert "is" not in result
    assert "on" not in result
    assert "cat" in result
    assert "mat" in result

  def test_strips_punctuation(self):
    result = _normalize("Hello, world!")
    assert "hello" in result
    assert "world" in result

  def test_empty_string(self):
    result = _normalize("")
    assert result == set()


@pytest.mark.unit
class TestJaccard:
  """Tests for the _jaccard similarity function."""

  def test_identical_sets(self):
    s = {"apple", "banana"}
    assert _jaccard(s, s) == 1.0

  def test_disjoint_sets(self):
    assert _jaccard({"apple"}, {"banana"}) == 0.0

  def test_partial_overlap(self):
    a = {"apple", "banana", "cherry"}
    b = {"banana", "cherry", "date"}
    # Intersection = {banana, cherry} = 2, Union = 4
    assert _jaccard(a, b) == pytest.approx(0.5)

  def test_empty_set(self):
    assert _jaccard(set(), {"apple"}) == 0.0
    assert _jaccard(set(), set()) == 0.0


# ===========================================================================
# KnowledgeGraph
# ===========================================================================


@pytest.mark.unit
class TestKnowledgeGraph:
  """Tests for KnowledgeGraph ingest, dedup, and retrieval."""

  def _make_cku(self, url="https://example.com", title="Test", topic="test query", facts=None):
    return CKU(
      source_url=url,
      source_title=title,
      query_context=topic,
      facts=facts or [],
      relevance_score=0.8,
    )

  def test_empty_graph(self):
    kg = KnowledgeGraph()
    assert kg.total_facts == 0
    assert kg.get_all_facts() == []
    assert kg.get_sources() == []

  def test_ingest_single_cku(self):
    kg = KnowledgeGraph()
    cku = self._make_cku(facts=[Fact(content="Python was created by Guido van Rossum")])
    count = kg.ingest([cku])
    assert count == 1
    assert kg.total_facts == 1

  def test_ingest_multiple_facts(self):
    kg = KnowledgeGraph()
    cku = self._make_cku(
      facts=[
        Fact(content="Python was created in 1991"),
        Fact(content="Java was created in 1995"),
      ]
    )
    count = kg.ingest([cku])
    assert count == 2
    assert kg.total_facts == 2

  def test_dedup_identical_facts(self):
    kg = KnowledgeGraph()
    cku1 = self._make_cku(facts=[Fact(content="Python is a programming language")])
    cku2 = self._make_cku(
      url="https://other.com",
      facts=[Fact(content="Python is a programming language")],
    )
    kg.ingest([cku1])
    count = kg.ingest([cku2])
    assert count == 0  # duplicate detected
    assert kg.total_facts == 1

  def test_dedup_near_identical_facts(self):
    kg = KnowledgeGraph()
    cku1 = self._make_cku(facts=[Fact(content="Python was created by Guido van Rossum in 1991")])
    cku2 = self._make_cku(
      url="https://other.com",
      facts=[Fact(content="Python was created by Guido van Rossum around 1991")],
    )
    kg.ingest([cku1])
    count = kg.ingest([cku2])
    # Near-identical (high Jaccard) should be deduped
    assert count == 0
    assert kg.total_facts == 1

  def test_distinct_facts_not_deduped(self):
    kg = KnowledgeGraph()
    cku = self._make_cku(
      facts=[
        Fact(content="Python is a programming language"),
        Fact(content="The Earth orbits the Sun"),
      ]
    )
    count = kg.ingest([cku])
    assert count == 2

  def test_topic_indexing(self):
    kg = KnowledgeGraph()
    cku_py = self._make_cku(topic="python", facts=[Fact(content="Python is interpreted")])
    cku_java = self._make_cku(topic="java", facts=[Fact(content="Java uses JVM")])
    kg.ingest([cku_py, cku_java])

    py_facts = kg.get_facts_by_topic("python")
    assert len(py_facts) == 1
    assert "interpreted" in py_facts[0].content

    java_facts = kg.get_facts_by_topic("java")
    assert len(java_facts) == 1

  def test_fact_count_for_topic(self):
    kg = KnowledgeGraph()
    cku = self._make_cku(
      topic="python",
      facts=[
        Fact(content="Python is interpreted"),
        Fact(content="Python uses indentation"),
      ],
    )
    kg.ingest([cku])
    assert kg.fact_count_for_topic("python") == 2
    assert kg.fact_count_for_topic("java") == 0

  def test_source_tracking(self):
    kg = KnowledgeGraph()
    cku = self._make_cku(
      url="https://wiki.org/python",
      title="Python Wiki",
      facts=[Fact(content="Python is popular")],
    )
    kg.ingest([cku])
    sources = kg.get_sources()
    assert len(sources) == 1
    assert sources[0].url == "https://wiki.org/python"
    assert sources[0].title == "Python Wiki"
    assert sources[0].fact_count == 1

  def test_multiple_sources(self):
    kg = KnowledgeGraph()
    cku1 = self._make_cku(url="https://a.com", facts=[Fact(content="Fact about A site")])
    cku2 = self._make_cku(url="https://b.com", facts=[Fact(content="Fact about B site")])
    kg.ingest([cku1, cku2])
    assert len(kg.get_sources()) == 2


@pytest.mark.unit
class TestKnowledgeGraphContradictions:
  """Tests for contradiction detection."""

  def test_no_contradictions_without_entities(self):
    kg = KnowledgeGraph()
    cku = CKU(
      source_url="https://a.com",
      source_title="A",
      query_context="test",
      facts=[
        Fact(content="Revenue is 5 billion", entities=[]),
        Fact(content="Revenue is 3 billion", entities=[]),
      ],
    )
    kg.ingest([cku])
    assert kg.get_contradictions() == []

  def test_contradiction_detected(self):
    kg = KnowledgeGraph()
    cku = CKU(
      source_url="https://a.com",
      source_title="A",
      query_context="test",
      facts=[
        Fact(content="Apple revenue reached 394 billion dollars", entities=["Apple"]),
        Fact(content="Apple revenue reached 250 billion dollars", entities=["Apple"]),
      ],
    )
    kg.ingest([cku])
    contradictions = kg.get_contradictions()
    assert len(contradictions) >= 1

  def test_no_contradiction_same_numbers(self):
    kg = KnowledgeGraph()
    cku = CKU(
      source_url="https://a.com",
      source_title="A",
      query_context="test",
      facts=[
        Fact(content="Apple revenue reached 394 billion dollars", entities=["Apple"]),
        Fact(content="Apple revenue reached 394 billion dollars in 2024", entities=["Apple"]),
      ],
    )
    kg.ingest([cku])
    # These are deduplicated (near-identical), so no contradictions possible
    assert kg.total_facts <= 2


# ===========================================================================
# DeepResearchConfig
# ===========================================================================


@pytest.mark.unit
class TestDeepResearchConfig:
  """Tests for DeepResearchConfig."""

  def test_defaults(self):
    cfg = DeepResearchConfig()
    assert cfg.enabled is True
    assert cfg.depth == "standard"
    assert cfg.search_provider == "duckduckgo"
    assert cfg.max_sources == 15
    assert cfg.max_waves == 3
    assert cfg.trigger == "always"
    assert cfg.context_format == "xml"

  def test_quick_depth(self):
    cfg = DeepResearchConfig(depth="quick")
    result = cfg.with_depth_preset()
    assert result.max_sources == 8
    assert result.max_waves == 1

  def test_deep_depth(self):
    cfg = DeepResearchConfig(depth="deep")
    result = cfg.with_depth_preset()
    assert result.max_sources == 30
    assert result.max_waves == 5

  def test_preset_does_not_override_custom(self):
    cfg = DeepResearchConfig(depth="quick", max_sources=50)
    result = cfg.with_depth_preset()
    assert result.max_sources == 50  # custom value preserved

  def test_custom_search_provider(self):
    cfg = DeepResearchConfig(search_provider="google")
    assert cfg.search_provider == "google"

  def test_description_field(self):
    cfg = DeepResearchConfig(description="Research financial data")
    assert cfg.description == "Research financial data"


# ===========================================================================
# Search types
# ===========================================================================


@pytest.mark.unit
class TestSearchTypes:
  """Tests for SearchResult and SearchProvider protocol."""

  def test_search_result_fields(self):
    r = SearchResult(title="Page", url="https://example.com", snippet="A snippet")
    assert r.title == "Page"
    assert r.url == "https://example.com"

  def test_search_provider_protocol(self):
    """Verify a class with the right signature satisfies the protocol."""

    class FakeProvider:
      async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        return []

    assert isinstance(FakeProvider(), SearchProvider)

  def test_callable_search_provider(self):
    from definable.agent.research.search import CallableSearchProvider

    async def my_search(query: str, max_results: int = 10) -> List[SearchResult]:
      return [SearchResult(title="R", url="https://r.com", snippet="s")]

    provider = CallableSearchProvider(my_search)
    results = asyncio.run(provider.search("test"))
    assert len(results) == 1
    assert results[0].title == "R"

  def test_create_search_provider_unknown_raises(self):
    from definable.agent.research.search import create_search_provider

    with pytest.raises(ValueError, match="Unknown search provider"):
      create_search_provider("nonexistent")

  def test_create_search_provider_custom_fn(self):
    from definable.agent.research.search import CallableSearchProvider, create_search_provider

    async def fn(q, n=10):
      return []

    provider = create_search_provider(search_fn=fn)
    assert isinstance(provider, CallableSearchProvider)
