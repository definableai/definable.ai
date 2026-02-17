"""Tests for the knowledge graph — dedup, contradiction detection, ingestion."""

from definable.research.knowledge_graph import KnowledgeGraph
from definable.research.models import CKU, Fact


class TestKnowledgeGraphIngestion:
  """Test fact ingestion and deduplication."""

  def test_ingest_single_cku(self, sample_ckus):
    kg = KnowledgeGraph()
    new = kg.ingest([sample_ckus[0]])
    assert new == 2
    assert kg.total_facts == 2

  def test_ingest_multiple_ckus(self, sample_ckus):
    kg = KnowledgeGraph()
    new = kg.ingest(sample_ckus)
    assert new == 3
    assert kg.total_facts == 3

  def test_dedup_exact_duplicate(self):
    kg = KnowledgeGraph()
    fact = Fact(content="IBM unveiled a 1121-qubit Condor processor.", entities=["IBM"])
    cku1 = CKU(source_url="https://a.com", source_title="A", query_context="q", facts=[fact], relevance_score=0.8)
    cku2 = CKU(source_url="https://b.com", source_title="B", query_context="q", facts=[fact], relevance_score=0.8)
    new1 = kg.ingest([cku1])
    new2 = kg.ingest([cku2])
    assert new1 == 1
    assert new2 == 0  # Duplicate
    assert kg.total_facts == 1

  def test_dedup_near_duplicate(self):
    kg = KnowledgeGraph()
    fact1 = Fact(content="IBM unveiled a 1121-qubit Condor processor in 2023.", entities=["IBM"])
    fact2 = Fact(content="IBM unveiled the 1121-qubit Condor processor.", entities=["IBM"])
    cku1 = CKU(source_url="https://a.com", source_title="A", query_context="q", facts=[fact1], relevance_score=0.8)
    cku2 = CKU(source_url="https://b.com", source_title="B", query_context="q", facts=[fact2], relevance_score=0.8)
    kg.ingest([cku1])
    new = kg.ingest([cku2])
    assert new == 0  # Near-duplicate should be caught
    assert kg.total_facts == 1

  def test_distinct_facts_not_deduped(self):
    kg = KnowledgeGraph()
    fact1 = Fact(content="IBM has a 1121-qubit processor.", entities=["IBM"])
    fact2 = Fact(content="Drug discovery is a key use case for quantum.", entities=["drug discovery"])
    cku = CKU(source_url="https://a.com", source_title="A", query_context="q", facts=[fact1, fact2], relevance_score=0.8)
    new = kg.ingest([cku])
    assert new == 2

  def test_source_tracking(self, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)
    sources = kg.get_sources()
    assert len(sources) == 2
    urls = {s.url for s in sources}
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls


class TestKnowledgeGraphTopics:
  """Test topic-based fact retrieval."""

  def test_facts_by_topic(self, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)
    hw_facts = kg.get_facts_by_topic("What are quantum computing hardware advances?")
    assert len(hw_facts) == 2

  def test_facts_missing_topic(self, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)
    facts = kg.get_facts_by_topic("nonexistent topic")
    assert len(facts) == 0

  def test_fact_count_for_topic(self, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)
    assert kg.fact_count_for_topic("What are quantum computing hardware advances?") == 2
    assert kg.fact_count_for_topic("What are quantum computing applications?") == 1


class TestContradictionDetection:
  """Test contradiction detection."""

  def test_detects_numeric_contradiction(self):
    kg = KnowledgeGraph()
    fact1 = Fact(content="IBM has 1,121 qubits in its processor.", entities=["IBM"])
    fact2 = Fact(content="IBM has 1,000 qubits in its processor.", entities=["IBM"])
    cku1 = CKU(source_url="https://a.com", source_title="A", query_context="q", facts=[fact1], relevance_score=0.8)
    cku2 = CKU(source_url="https://b.com", source_title="B", query_context="q", facts=[fact2], relevance_score=0.8)
    kg.ingest([cku1, cku2])
    contradictions = kg.get_contradictions()
    assert len(contradictions) >= 1

  def test_no_contradiction_on_different_topics(self):
    kg = KnowledgeGraph()
    fact1 = Fact(content="IBM has 1,121 qubits.", entities=["IBM"])
    fact2 = Fact(content="Google has 70 qubits.", entities=["Google"])
    cku = CKU(source_url="https://a.com", source_title="A", query_context="q", facts=[fact1, fact2], relevance_score=0.8)
    kg.ingest([cku])
    contradictions = kg.get_contradictions()
    # Different entities, not a contradiction
    assert len(contradictions) == 0

  def test_empty_graph_no_contradictions(self):
    kg = KnowledgeGraph()
    assert kg.get_contradictions() == []
