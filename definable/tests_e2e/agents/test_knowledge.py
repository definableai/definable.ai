"""E2E tests for Knowledge middleware and toolkit integration."""

import pytest

from definable.agents.config import KnowledgeConfig
from definable.agents.testing import AgentTestCase
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.memory import InMemoryVectorDB


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
