"""Shared fixtures for deep research e2e tests."""

import json
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from definable.agents.testing import MockModel
from definable.models.metrics import Metrics
from definable.research.config import DeepResearchConfig
from definable.research.models import CKU, Fact, PageContent
from definable.research.search.base import SearchResult


class MockSearchProvider:
  """Mock search provider that returns canned results."""

  def __init__(self, results: Optional[List[SearchResult]] = None):
    self._results = results if results is not None else [
      SearchResult(
        title="Quantum Computing Advances 2025",
        url="https://example.com/quantum-2025",
        snippet="IBM unveiled a 1121-qubit processor in 2025.",
      ),
      SearchResult(
        title="Google Quantum Supremacy Update",
        url="https://example.com/google-quantum",
        snippet="Google's Willow chip demonstrated error correction.",
      ),
      SearchResult(
        title="Quantum Computing Applications",
        url="https://example.com/quantum-apps",
        snippet="Drug discovery and cryptography are leading use cases.",
      ),
    ]
    self.call_count = 0
    self.queries: List[str] = []

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
    self.call_count += 1
    self.queries.append(query)
    return self._results[:max_results]


class ResearchMockModel(MockModel):
  """MockModel with research-specific JSON responses.

  Routes calls based on prompt content:
  - Decompose prompt → JSON sub-questions array
  - CKU extraction prompt → JSON facts
  - Gap analysis prompt → JSON assessments
  - Needs research prompt → JSON bool
  - Synthesis prompt → formatted context block
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._decompose_response = json.dumps([
      "What are the latest quantum computing hardware advances?",
      "What are the main applications of quantum computing?",
      "What are the challenges facing quantum computing?",
    ])
    self._cku_response = json.dumps({
      "facts": [
        {
          "content": "IBM unveiled a 1121-qubit Condor processor.",
          "fact_type": "factual",
          "confidence": 0.95,
          "entities": ["IBM", "Condor"],
          "source_sentence": "IBM unveiled a 1121-qubit Condor processor in 2023.",
        },
        {
          "content": "Quantum error correction is a key milestone.",
          "fact_type": "factual",
          "confidence": 0.9,
          "entities": ["quantum error correction"],
          "source_sentence": "Achieving quantum error correction is considered a key milestone.",
        },
      ],
      "relevance_score": 0.85,
      "page_summary": "Overview of recent quantum computing hardware advances.",
      "suggested_followup": "",
    })
    self._gap_response = json.dumps({
      "assessments": [
        {"topic": "hardware advances", "status": "sufficient", "confidence": 0.9, "suggested_queries": []},
        {"topic": "applications", "status": "partial", "confidence": 0.6, "suggested_queries": ["quantum computing drug discovery"]},
        {"topic": "challenges", "status": "missing", "confidence": 0.3, "suggested_queries": ["quantum decoherence challenges"]},
      ],
      "new_queries": ["quantum computing drug discovery", "quantum decoherence challenges"],
    })
    self._needs_research_response = json.dumps({"needs_research": True, "reason": "Current topic"})
    self._synthesis_response = (
      "<research_context>\n"
      "  <topic>Quantum Computing</topic>\n"
      "  <fact>IBM has a 1121-qubit processor</fact>\n"
      "</research_context>"
    )

  async def ainvoke(self, messages=None, tools=None, system_message=None, output_schema=None, **kwargs):
    self._call_history.append({
      "messages": messages,
      "tools": tools,
      "system_message": system_message,
      "output_schema": output_schema,
      **kwargs,
    })

    # Route based on prompt content
    prompt = ""
    if messages:
      for msg in messages:
        if hasattr(msg, "content") and msg.content:
          prompt += str(msg.content)

    response = MagicMock()
    response.tool_executions = []
    response.tool_calls = []
    response.response_usage = Metrics()
    response.reasoning_content = None
    response.citations = None
    response.images = None
    response.videos = None
    response.audios = None

    if "decompose it into" in prompt.lower() or "research planner" in prompt.lower():
      response.content = self._decompose_response
    elif "extract all relevant facts" in prompt.lower() or "information extraction" in prompt.lower():
      response.content = self._cku_response
    elif "coverage analyst" in prompt.lower() or "gap" in prompt.lower():
      response.content = self._gap_response
    elif "needs_research" in prompt.lower() or "query classifier" in prompt.lower():
      response.content = self._needs_research_response
    elif "research synthesizer" in prompt.lower() or "synthesize" in prompt.lower():
      response.content = self._synthesis_response
    else:
      response.content = self.responses[min(self._call_count, len(self.responses) - 1)]

    self._call_count += 1
    return response


@pytest.fixture
def mock_search():
  """Mock search provider fixture."""
  return MockSearchProvider()


@pytest.fixture
def mock_model():
  """Research-aware mock model fixture."""
  return ResearchMockModel(responses=["Mock response"])


@pytest.fixture
def research_config():
  """Quick research config for testing."""
  return DeepResearchConfig(
    depth="quick",
    max_sources=3,
    max_waves=1,
    parallel_searches=2,
    parallel_reads=3,
    min_relevance=0.0,
    early_termination_threshold=0.0,
  )


@pytest.fixture
def sample_pages():
  """Sample PageContent objects for testing."""
  return [
    PageContent(
      url="https://example.com/page1",
      title="Quantum Computing Advances",
      content="IBM unveiled a 1121-qubit Condor processor in 2023. "
      "This represents a major milestone in quantum computing hardware. "
      "Google also demonstrated quantum error correction with its Willow chip.",
    ),
    PageContent(
      url="https://example.com/page2",
      title="Applications of Quantum Computing",
      content="Quantum computing has promising applications in drug discovery, "
      "cryptography, and materials science. Companies like IBM and Google "
      "are leading the charge in making quantum useful.",
    ),
  ]


@pytest.fixture
def sample_ckus():
  """Sample CKU objects for testing."""
  return [
    CKU(
      source_url="https://example.com/page1",
      source_title="Quantum Computing Advances",
      query_context="What are quantum computing hardware advances?",
      facts=[
        Fact(
          content="IBM unveiled a 1121-qubit Condor processor.",
          fact_type="factual",
          confidence=0.95,
          entities=["IBM", "Condor"],
        ),
        Fact(
          content="Google demonstrated quantum error correction with Willow.",
          fact_type="factual",
          confidence=0.9,
          entities=["Google", "Willow"],
        ),
      ],
      relevance_score=0.85,
      raw_token_count=50,
      compressed_token_count=20,
      compression_ratio=0.4,
    ),
    CKU(
      source_url="https://example.com/page2",
      source_title="Applications of Quantum Computing",
      query_context="What are quantum computing applications?",
      facts=[
        Fact(
          content="Drug discovery is a leading quantum computing use case.",
          fact_type="factual",
          confidence=0.85,
          entities=["drug discovery"],
        ),
      ],
      relevance_score=0.7,
      raw_token_count=40,
      compressed_token_count=12,
      compression_ratio=0.3,
    ),
  ]
