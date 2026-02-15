"""E2E tests — RAG (Knowledge) Workflows.

Scenario: "I want my agent to answer questions from my documents."

Tests requiring real embeddings use OPENAI_API_KEY.
InMemoryVectorDB tests with pre-computed embeddings run without API keys.
"""

import pytest

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, KnowledgeConfig, TracingConfig
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.memory import InMemoryVectorDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_db():
  """InMemoryVectorDB with 3 dimensions for testing."""
  return InMemoryVectorDB(dimensions=3)


@pytest.fixture
def populated_knowledge(vector_db):
  """Knowledge base with pre-embedded test documents."""
  docs = [
    Document(
      id="acme-founding",
      content="Acme Corp was founded in 2020 by Jane Smith. It specializes in AI-powered analytics.",
      embedding=[0.9, 0.1, 0.0],
      meta_data={"topic": "company"},
    ),
    Document(
      id="acme-product",
      content="Acme's flagship product is SmartDash, an analytics dashboard launched in 2023.",
      embedding=[0.8, 0.2, 0.0],
      meta_data={"topic": "product"},
    ),
    Document(
      id="acme-location",
      content="Acme Corp headquarters is located at 456 Innovation Drive, Austin, Texas.",
      embedding=[0.7, 0.3, 0.0],
      meta_data={"topic": "location"},
    ),
  ]
  vector_db.add(docs)
  return Knowledge(vector_db=vector_db)


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorDB (no API key needed)
# ---------------------------------------------------------------------------


class TestKnowledgeBasic:
  """Knowledge base operations without API calls."""

  def test_add_text_documents_and_search(self, vector_db):
    """Add documents with embeddings, search returns the most relevant."""
    docs = [
      Document(id="d1", content="Python is a programming language", embedding=[1.0, 0.0, 0.0]),
      Document(id="d2", content="JavaScript runs in the browser", embedding=[0.0, 1.0, 0.0]),
      Document(id="d3", content="Rust is a systems language", embedding=[0.5, 0.5, 0.0]),
    ]
    vector_db.add(docs)

    results = vector_db.search(query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "d1"

  def test_add_and_remove_documents(self, vector_db):
    """Add 5 docs, remove 2 -> search only returns remaining 3."""
    docs = [Document(id=f"d{i}", content=f"Document {i}", embedding=[float(i) / 5, 0.0, 0.0]) for i in range(5)]
    vector_db.add(docs)
    assert vector_db.count() == 5

    vector_db.delete(ids=["d0", "d1"])
    assert vector_db.count() == 3

    results = vector_db.search(query_embedding=[1.0, 0.0, 0.0], top_k=10)
    result_ids = {r.id for r in results}
    assert "d0" not in result_ids
    assert "d1" not in result_ids

  def test_knowledge_clear(self, vector_db):
    """Clear knowledge base -> search returns empty."""
    docs = [
      Document(id="d1", content="Test doc", embedding=[1.0, 0.0, 0.0]),
    ]
    vector_db.add(docs)
    assert vector_db.count() == 1

    vector_db.clear()
    assert vector_db.count() == 0

    results = vector_db.search(query_embedding=[1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 0

  @pytest.mark.asyncio
  async def test_async_operations(self, vector_db):
    """Async add, search, delete all work correctly."""
    docs = [Document(id="d1", content="Async test", embedding=[1.0, 0.0, 0.0])]
    await vector_db.aadd(docs)
    assert vector_db.count() == 1

    results = await vector_db.asearch(query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1

    await vector_db.adelete(ids=["d1"])
    assert vector_db.count() == 0


# ---------------------------------------------------------------------------
# Tests: Agent + Knowledge (requires OpenAI)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestAgentWithKnowledge:
  """Agent uses knowledge base to answer questions."""

  @pytest.mark.asyncio
  async def test_agent_with_knowledge_answers_from_docs(self, openai_model, populated_knowledge):
    """Agent with knowledge retrieves and uses document content."""
    from unittest.mock import AsyncMock, patch

    mock_docs = [
      Document(
        id="acme-founding",
        content="Acme Corp was founded in 2020 by Jane Smith. It specializes in AI-powered analytics.",
        reranking_score=0.9,
      ),
    ]

    with patch.object(populated_knowledge, "asearch", new_callable=AsyncMock, return_value=mock_docs):
      agent = Agent(
        model=openai_model,
        instructions="Answer questions using the provided knowledge context.",
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=populated_knowledge,
            top_k=2,
            enabled=True,
          ),
        ),
      )
      output = await agent.arun("Who founded Acme Corp?")

    assert output.content is not None
    assert "jane" in output.content.lower() or "smith" in output.content.lower()

  @pytest.mark.asyncio
  async def test_agent_without_knowledge_cannot_answer(self, openai_model):
    """Without knowledge, agent cannot answer questions about fictional company."""
    agent = Agent(
      model=openai_model,
      instructions="Only answer from your knowledge. If you don't know, say 'I don't know'.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )
    output = await agent.arun("Who founded Xylotron Dynamics Inc and in what year?")

    assert output.content is not None
    # Should not confidently answer about a fictional company
    content_lower = output.content.lower()
    assert any(phrase in content_lower for phrase in ["don't know", "not sure", "no information", "cannot", "unable", "don't have"])

  @pytest.mark.asyncio
  async def test_knowledge_toolkit_search(self, openai_model, populated_knowledge):
    """Agent with KnowledgeToolkit can search the knowledge base."""
    from definable.agents.toolkits.knowledge import KnowledgeToolkit

    tk = KnowledgeToolkit(knowledge=populated_knowledge)
    agent = Agent(
      model=openai_model,
      toolkits=[tk],
      instructions="Search the knowledge base to answer questions about the company.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # Verify toolkit is registered
    all_tools = agent._flatten_tools()
    assert "search_knowledge" in all_tools

  @pytest.mark.asyncio
  async def test_knowledge_config_disabled(self, openai_model, populated_knowledge):
    """Disabled knowledge config means no knowledge retrieval."""
    from unittest.mock import AsyncMock, patch

    with patch.object(populated_knowledge, "asearch", new_callable=AsyncMock) as mock_search:
      agent = Agent(
        model=openai_model,
        instructions="Answer questions.",
        config=AgentConfig(
          tracing=TracingConfig(enabled=False),
          knowledge=KnowledgeConfig(
            knowledge=populated_knowledge,
            enabled=False,
          ),
        ),
      )
      await agent.arun("Who founded Acme?")

    # asearch should not have been called
    mock_search.assert_not_called()
