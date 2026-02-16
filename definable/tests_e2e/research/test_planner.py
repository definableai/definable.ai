"""Tests for the query planner (decomposition)."""

import json

import pytest

from definable.agents.testing import MockModel
from definable.research.planner import decompose


@pytest.mark.asyncio
class TestDecompose:
  """Test query decomposition."""

  async def test_decompose_returns_sub_questions(self, mock_model):
    result = await decompose("What are the latest quantum computing developments?", mock_model)
    assert isinstance(result, list)
    assert len(result) >= 2
    assert all(isinstance(q, str) for q in result)

  async def test_decompose_fallback_on_parse_error(self):
    """If the model returns non-JSON, fallback to original query."""
    model = MockModel(responses=["This is not valid JSON at all"])
    result = await decompose("What is quantum computing?", model)
    assert result == ["What is quantum computing?"]

  async def test_decompose_fallback_on_non_list(self):
    """If the model returns JSON but not a list, fallback."""
    model = MockModel(responses=[json.dumps({"not": "a list"})])
    result = await decompose("test query", model)
    assert result == ["test query"]

  async def test_decompose_with_markdown_fences(self):
    """Model might wrap JSON in ```json ... ``` fences."""
    sub_qs = ["What is A?", "What is B?", "What is C?"]
    model = MockModel(responses=["```json\n" + json.dumps(sub_qs) + "\n```"])
    result = await decompose("complex query", model)
    assert result == sub_qs
