"""Unit tests for the KnowledgeToolkit.

Covers initialization, tools property, search_knowledge, and get_document_count.
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest

from definable.agent.toolkits.knowledge import KnowledgeToolkit
from definable.agent.toolkit import Toolkit


# ---------------------------------------------------------------------------
# Fake Knowledge for testing (no real vector DB or embedder needed)
# ---------------------------------------------------------------------------


@dataclass
class FakeDoc:
  content: str = ""
  name: Optional[str] = None
  reranking_score: Optional[float] = None


class FakeKnowledge:
  """Minimal knowledge base mock for testing the toolkit."""

  def __init__(self, docs: Optional[List[FakeDoc]] = None):
    self._docs = docs or []

  def search(self, query: str, top_k: int = 5, rerank: bool = True) -> List[FakeDoc]:
    return self._docs[:top_k]

  def __len__(self) -> int:
    return len(self._docs)


class FailingKnowledge:
  """Knowledge base that raises on search."""

  def search(self, query: str, **kw) -> List:
    raise RuntimeError("Search failed")

  def __len__(self) -> int:
    return 0


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.unit
class TestKnowledgeToolkitInit:
  """Tests for KnowledgeToolkit initialization."""

  def test_is_toolkit_subclass(self):
    kb = FakeKnowledge()
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]
    assert isinstance(tk, Toolkit)

  def test_stores_knowledge_ref(self):
    kb = FakeKnowledge()
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]
    assert tk._knowledge is kb  # type: ignore[comparison-overlap]

  def test_default_top_k(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge())  # type: ignore[arg-type]
    assert tk._top_k == 5

  def test_custom_top_k(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge(), top_k=10)  # type: ignore[arg-type]
    assert tk._top_k == 10

  def test_rerank_default(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge())  # type: ignore[arg-type]
    assert tk._rerank is True

  def test_rerank_disabled(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge(), rerank=False)  # type: ignore[arg-type]
    assert tk._rerank is False

  def test_dependencies_include_knowledge(self):
    kb = FakeKnowledge()
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]
    assert "_kb_toolkit_knowledge" in tk.dependencies
    assert tk.dependencies["_kb_toolkit_knowledge"] is kb


@pytest.mark.unit
class TestKnowledgeToolkitTools:
  """Tests for KnowledgeToolkit.tools property."""

  def test_provides_two_tools(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge())  # type: ignore[arg-type]
    tools = tk.tools
    assert len(tools) == 2

  def test_tool_names(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge())  # type: ignore[arg-type]
    names = {t.name for t in tk.tools}
    assert "search_knowledge" in names
    assert "get_document_count" in names

  def test_repr(self):
    tk = KnowledgeToolkit(knowledge=FakeKnowledge())  # type: ignore[arg-type]
    r = repr(tk)
    assert "KnowledgeToolkit" in r
    assert "tools=2" in r


@pytest.mark.unit
class TestSearchKnowledgeTool:
  """Tests for the search_knowledge tool."""

  def test_returns_results(self):
    docs = [FakeDoc(content="Policy: 20 days PTO", name="HR Policy")]
    kb = FakeKnowledge(docs=docs)
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="PTO policy")  # type: ignore[misc]
    assert "Policy: 20 days PTO" in result
    assert "HR Policy" in result

  def test_no_results(self):
    kb = FakeKnowledge(docs=[])
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="nothing")  # type: ignore[misc]
    assert "No relevant documents found" in result

  def test_custom_top_k(self):
    docs = [FakeDoc(content=f"Doc {i}") for i in range(10)]
    kb = FakeKnowledge(docs=docs)
    tk = KnowledgeToolkit(knowledge=kb, top_k=3)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="test")  # type: ignore[misc]
    # Should have at most 3 doc sections
    assert result.count("[") <= 3

  def test_top_k_override(self):
    docs = [FakeDoc(content=f"Doc {i}") for i in range(10)]
    kb = FakeKnowledge(docs=docs)
    tk = KnowledgeToolkit(knowledge=kb, top_k=3)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="test", top_k=1)  # type: ignore[misc]
    assert result.count("[") <= 1

  def test_relevance_score_shown(self):
    docs = [FakeDoc(content="Test", reranking_score=0.95)]
    kb = FakeKnowledge(docs=docs)
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="test")  # type: ignore[misc]
    assert "0.950" in result

  def test_error_handled_gracefully(self):
    kb = FailingKnowledge()
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    search_tool = next(t for t in tk.tools if t.name == "search_knowledge")
    result = search_tool.entrypoint(query="test")  # type: ignore[misc]
    assert "Error searching knowledge base" in result


@pytest.mark.unit
class TestGetDocumentCountTool:
  """Tests for the get_document_count tool."""

  def test_returns_count(self):
    docs = [FakeDoc(content=f"Doc {i}") for i in range(5)]
    kb = FakeKnowledge(docs=docs)
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    count_tool = next(t for t in tk.tools if t.name == "get_document_count")
    result = count_tool.entrypoint()  # type: ignore[misc]
    assert "5 document(s)" in result

  def test_empty_knowledge_base(self):
    kb = FakeKnowledge(docs=[])
    tk = KnowledgeToolkit(knowledge=kb)  # type: ignore[arg-type]

    count_tool = next(t for t in tk.tools if t.name == "get_document_count")
    result = count_tool.entrypoint()  # type: ignore[misc]
    assert "0 document(s)" in result
