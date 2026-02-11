"""E2E tests for Knowledge middleware and toolkit integration."""

from unittest.mock import AsyncMock, patch

import pytest

from definable.agents.config import KnowledgeConfig
from definable.agents.testing import AgentTestCase
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
from definable.run.agent import (
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  run_output_event_from_dict,
)


@pytest.mark.e2e
class TestKnowledgeE2E(AgentTestCase):
  """End-to-end tests for Knowledge integration."""

  def test_in_memory_vector_db_add_search(self):
    """InMemoryVectorDB can add and search documents."""
    vector_db = InMemoryVectorDB(dimensions=3)

    docs = [
      Document(id="d1", content="Hello world", embedding=[1.0, 0.0, 0.0]),
      Document(id="d2", content="Goodbye world", embedding=[0.0, 1.0, 0.0]),
    ]
    vector_db.add(docs)

    assert vector_db.count() == 2

    # Search with query embedding similar to d1
    results = vector_db.search(query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "d1"

  def test_knowledge_with_in_memory_db(self, in_memory_knowledge):
    """Knowledge instance works with InMemoryVectorDB."""
    kb = in_memory_knowledge

    # Verify documents were added
    count = kb.vector_db.count()
    assert count == 3

  def test_knowledge_config_creation(self, in_memory_knowledge):
    """KnowledgeConfig can be created with Knowledge instance."""
    config = KnowledgeConfig(
      knowledge=in_memory_knowledge,
      top_k=2,
      context_format="xml",
      enabled=True,
    )

    assert config.knowledge == in_memory_knowledge
    assert config.top_k == 2
    assert config.context_format == "xml"
    assert config.enabled is True

  def test_agent_with_knowledge_config(self, in_memory_knowledge, mock_model):
    """Agent can be created with KnowledgeConfig."""
    agent = self.create_agent(
      model=mock_model,
      config_kwargs={
        "knowledge": KnowledgeConfig(
          knowledge=in_memory_knowledge,
          top_k=2,
          enabled=True,
        )
      },
    )

    assert agent.config.knowledge is not None
    assert agent.config.knowledge.enabled is True

  def test_knowledge_toolkit_available(self, in_memory_knowledge, mock_model):
    """KnowledgeToolkit can be used with agent."""
    from definable.agents.toolkits.knowledge import KnowledgeToolkit

    tk = KnowledgeToolkit(knowledge=in_memory_knowledge)
    agent = self.create_agent(model=mock_model, toolkits=[tk])

    all_tools = agent._flatten_tools()
    tool_names = list(all_tools.keys())

    # KnowledgeToolkit should expose search_knowledge tool
    assert "search_knowledge" in tool_names

  def test_knowledge_toolkit_document_count(self, in_memory_knowledge):
    """KnowledgeToolkit get_document_count works."""
    from definable.agents.toolkits.knowledge import KnowledgeToolkit

    tk = KnowledgeToolkit(knowledge=in_memory_knowledge)
    tools = tk.tools
    tool_names = [t.name for t in tools]

    # Should have document count tool
    assert "get_document_count" in tool_names

  def test_knowledge_config_defaults(self):
    """KnowledgeConfig has sensible defaults."""
    kb = Knowledge(vector_db=InMemoryVectorDB(dimensions=3))
    config = KnowledgeConfig(knowledge=kb)

    assert config.top_k == 5  # default
    assert config.context_format == "xml"  # default
    assert config.enabled is True  # default

  def test_knowledge_config_context_formats(self, in_memory_knowledge):
    """KnowledgeConfig supports different context formats."""
    for fmt in ["xml", "markdown", "json"]:
      config = KnowledgeConfig(
        knowledge=in_memory_knowledge,
        context_format=fmt,
      )
      assert config.context_format == fmt

  def test_knowledge_config_disabled(self, in_memory_knowledge, mock_model):
    """Knowledge can be disabled via config."""
    agent = self.create_agent(
      model=mock_model,
      config_kwargs={
        "knowledge": KnowledgeConfig(
          knowledge=in_memory_knowledge,
          enabled=False,
        )
      },
    )

    assert agent.config.knowledge.enabled is False

  def test_vector_db_clear(self):
    """InMemoryVectorDB can be cleared."""
    vector_db = InMemoryVectorDB(dimensions=3)
    docs = [Document(id="d1", content="Test", embedding=[1.0, 0.0, 0.0])]
    vector_db.add(docs)

    assert vector_db.count() == 1

    vector_db.clear()
    assert vector_db.count() == 0

  def test_vector_db_delete(self):
    """InMemoryVectorDB can delete specific documents."""
    vector_db = InMemoryVectorDB(dimensions=3)
    docs = [
      Document(id="d1", content="Keep", embedding=[1.0, 0.0, 0.0]),
      Document(id="d2", content="Delete", embedding=[0.0, 1.0, 0.0]),
    ]
    vector_db.add(docs)

    assert vector_db.count() == 2

    vector_db.delete(ids=["d2"])
    assert vector_db.count() == 1

    # Verify correct document remains
    results = vector_db.search(query_embedding=[1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].id == "d1"

  @pytest.mark.asyncio
  async def test_vector_db_async_operations(self):
    """InMemoryVectorDB async methods work."""
    vector_db = InMemoryVectorDB(dimensions=3)
    docs = [Document(id="d1", content="Async test", embedding=[1.0, 0.0, 0.0])]

    await vector_db.aadd(docs)
    assert vector_db.count() == 1

    results = await vector_db.asearch(query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1

    await vector_db.adelete(ids=["d1"])
    assert vector_db.count() == 0


@pytest.mark.e2e
class TestKnowledgeEvents(AgentTestCase):
  """Tests for knowledge retrieval event emission."""

  def test_knowledge_event_serialization_roundtrip(self):
    """KnowledgeRetrievalStarted/Completed events serialize and deserialize correctly."""
    started = KnowledgeRetrievalStartedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      query="test query",
    )
    d = started.to_dict()
    assert d["event"] == "KnowledgeRetrievalStarted"
    assert d["query"] == "test query"
    reconstructed = run_output_event_from_dict(d)
    assert isinstance(reconstructed, KnowledgeRetrievalStartedEvent)
    assert reconstructed.query == "test query"

    completed = KnowledgeRetrievalCompletedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      query="test query",
      documents_found=5,
      documents_used=3,
      duration_ms=42.5,
    )
    d = completed.to_dict()
    assert d["event"] == "KnowledgeRetrievalCompleted"
    assert d["documents_found"] == 5
    assert d["documents_used"] == 3
    assert d["duration_ms"] == 42.5
    reconstructed = run_output_event_from_dict(d)
    assert isinstance(reconstructed, KnowledgeRetrievalCompletedEvent)
    assert reconstructed.documents_found == 5
    assert reconstructed.documents_used == 3

  @pytest.mark.asyncio
  async def test_knowledge_events_emitted_on_arun(self, mock_model, in_memory_knowledge):
    """Agent.arun() emits KnowledgeRetrievalStarted/Completed events."""
    # Mock asearch to return docs without requiring an embedder
    mock_docs = [
      Document(id="d1", content="Python info", reranking_score=0.9),
      Document(id="d2", content="ML info", reranking_score=0.8),
    ]
    with patch.object(in_memory_knowledge, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = self.create_agent(
        model=mock_model,
        config_kwargs={
          "knowledge": KnowledgeConfig(
            knowledge=in_memory_knowledge,
            top_k=2,
            enabled=True,
          )
        },
      )

      # Collect events via trace writer mock
      emitted_events = []
      original_emit = agent._emit

      def capture_emit(event):
        emitted_events.append(event)
        original_emit(event)

      agent._emit = capture_emit

      await agent.arun("Tell me about Python")

    event_types = [type(e).__name__ for e in emitted_events]
    assert "KnowledgeRetrievalStartedEvent" in event_types
    assert "KnowledgeRetrievalCompletedEvent" in event_types

    # Verify started event
    started = next(e for e in emitted_events if isinstance(e, KnowledgeRetrievalStartedEvent))
    assert started.query == "Tell me about Python"

    # Verify completed event
    completed = next(e for e in emitted_events if isinstance(e, KnowledgeRetrievalCompletedEvent))
    assert completed.query == "Tell me about Python"
    assert completed.documents_found == 2
    assert completed.documents_used == 2
    assert completed.duration_ms is not None
    assert completed.duration_ms >= 0

  @pytest.mark.asyncio
  async def test_knowledge_events_emitted_on_stream(self, mock_model, in_memory_knowledge):
    """Agent.arun_stream() yields KnowledgeRetrievalStarted/Completed events."""
    mock_docs = [
      Document(id="d1", content="Python info", reranking_score=0.9),
    ]
    with patch.object(in_memory_knowledge, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = self.create_agent(
        model=mock_model,
        config_kwargs={
          "knowledge": KnowledgeConfig(
            knowledge=in_memory_knowledge,
            top_k=2,
            enabled=True,
          )
        },
      )

      events = []
      async for evt in agent.arun_stream("Tell me about Python"):
        events.append(evt)

    event_types = [type(e).__name__ for e in events]
    assert "KnowledgeRetrievalStartedEvent" in event_types
    assert "KnowledgeRetrievalCompletedEvent" in event_types

    # Verify ordering: knowledge events before run started
    kr_started_idx = event_types.index("KnowledgeRetrievalStartedEvent")
    kr_completed_idx = event_types.index("KnowledgeRetrievalCompletedEvent")
    run_started_idx = event_types.index("RunStartedEvent")
    assert kr_started_idx < kr_completed_idx < run_started_idx

  @pytest.mark.asyncio
  async def test_knowledge_events_not_emitted_when_disabled(self, mock_model, in_memory_knowledge):
    """No knowledge events emitted when knowledge is disabled."""
    agent = self.create_agent(
      model=mock_model,
      config_kwargs={
        "knowledge": KnowledgeConfig(
          knowledge=in_memory_knowledge,
          enabled=False,
        )
      },
    )

    emitted_events = []
    original_emit = agent._emit

    def capture_emit(event):
      emitted_events.append(event)
      original_emit(event)

    agent._emit = capture_emit

    await agent.arun("Tell me about Python")

    event_types = [type(e).__name__ for e in emitted_events]
    assert "KnowledgeRetrievalStartedEvent" not in event_types
    assert "KnowledgeRetrievalCompletedEvent" not in event_types
