"""Tests for the gap analyzer."""

import pytest

from definable.research.gap_analyzer import analyze
from definable.research.knowledge_graph import KnowledgeGraph


@pytest.mark.asyncio
class TestGapAnalyzer:
  """Test coverage analysis and gap query generation."""

  async def test_analyze_with_gaps(self, mock_model, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)

    sub_questions = [
      "What are quantum computing hardware advances?",
      "What are quantum computing applications?",
      "What are quantum computing challenges?",
    ]
    gaps, new_queries = await analyze("quantum computing overview", sub_questions, kg, mock_model)

    assert len(gaps) > 0
    assert isinstance(new_queries, list)

  async def test_analyze_returns_assessments(self, mock_model, sample_ckus):
    kg = KnowledgeGraph()
    kg.ingest(sample_ckus)

    gaps, _ = await analyze(
      "quantum",
      ["What are quantum computing hardware advances?"],
      kg,
      mock_model,
    )

    for gap in gaps:
      assert gap.status in ("sufficient", "partial", "missing")
      assert 0.0 <= gap.confidence <= 1.0

  async def test_analyze_empty_graph(self, mock_model):
    kg = KnowledgeGraph()
    gaps, queries = await analyze("test", ["sub-q 1"], kg, mock_model)
    assert isinstance(gaps, list)
    assert isinstance(queries, list)
