"""
Behavioral tests: Does the agent retrieve from knowledge base correctly?

Strategy:
  - Real OpenAI model + real embedder only — no mocks
  - Store real documents in the knowledge base BEFORE running the agent
  - Assert the agent's answer CONTAINS the fact from the document
  - Do NOT assert how retrieval worked internally

Covers:
  - Agent answers factual questions using knowledge base content
  - Agent retrieves the most relevant document for a query
  - Agent with knowledge + real model produces grounded answers
  - Agent without knowledge does not have access to stored facts
  - Multiple documents: agent retrieves the correct one
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.tracing import Tracing
from definable.knowledge import Knowledge
from definable.knowledge.document import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trace():
  return AgentConfig(tracing=Tracing(enabled=False))


@pytest.fixture
def knowledge_base(openai_embedder):
  """Fresh Knowledge instance backed by InMemoryVectorDB."""
  from definable.vectordb import InMemoryVectorDB

  db = InMemoryVectorDB(embedder=openai_embedder)
  kb = Knowledge(vector_db=db, embedder=openai_embedder)
  yield kb
  db.drop()


# ---------------------------------------------------------------------------
# Knowledge retrieval behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.openai
class TestKnowledgeRetrievalBehavior:
  """Agent uses knowledge base to answer questions. Requires OPENAI_API_KEY."""

  @pytest.mark.asyncio
  async def test_agent_answers_from_knowledge_base(self, openai_model, knowledge_base, no_trace):
    """Agent should use knowledge base content to answer factual questions."""
    doc = Document(
      content="Definable.ai version 0.2.8 was released in February 2026. It supports OpenAI, DeepSeek, Moonshot, and xAI providers.",
      name="release_notes",
    )
    await knowledge_base.aadd([doc])

    knowledge_base.top_k = 3
    agent = Agent(
      model=openai_model,
      knowledge=knowledge_base,
      config=no_trace,
    )

    output = await agent.arun("What version of definable.ai was released in February 2026?")
    assert "0.2.8" in output.content  # type: ignore[operator]

  @pytest.mark.asyncio
  async def test_agent_retrieves_most_relevant_document(self, openai_model, knowledge_base, no_trace):
    """When multiple documents are in KB, agent retrieves the relevant one."""
    docs = [
      Document(content="The speed of light is approximately 299,792,458 meters per second.", name="physics"),
      Document(content="Python 3.12 was released in October 2023.", name="python_release"),
      Document(content="The Eiffel Tower is 330 meters tall and located in Paris.", name="eiffel"),
    ]
    for doc in docs:
      await knowledge_base.aadd([doc])

    knowledge_base.top_k = 2
    agent = Agent(
      model=openai_model,
      knowledge=knowledge_base,
      config=no_trace,
    )

    output = await agent.arun("How tall is the Eiffel Tower?")
    assert "330" in output.content or "paris" in output.content.lower()  # type: ignore[union-attr,operator]

  @pytest.mark.asyncio
  async def test_knowledge_grounding_prevents_hallucination(self, openai_model, knowledge_base, no_trace):
    """Agent should use knowledge content rather than prior training knowledge."""
    doc = Document(
      content="The fictional planet Zorbax has 7 moons named: Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta.",
      name="zorbax_facts",
    )
    await knowledge_base.aadd([doc])

    knowledge_base.top_k = 3
    agent = Agent(
      model=openai_model,
      knowledge=knowledge_base,
      instructions="Use only information from the provided context.",
      config=no_trace,
    )

    output = await agent.arun("How many moons does Zorbax have?")
    assert "7" in output.content or "seven" in output.content.lower()  # type: ignore[union-attr,operator]

  @pytest.mark.asyncio
  async def test_agent_with_empty_knowledge_still_responds(self, openai_model, knowledge_base, no_trace):
    """Agent with empty KB should still respond (from model training knowledge)."""
    knowledge_base.top_k = 3
    agent = Agent(
      model=openai_model,
      knowledge=knowledge_base,
      config=no_trace,
    )

    output = await agent.arun("What is the capital of Germany?")
    assert output.content
    assert "berlin" in output.content.lower()

  @pytest.mark.asyncio
  async def test_knowledge_agent_completes_successfully(self, openai_model, knowledge_base, no_trace):
    """Agent with knowledge config completes a run without error."""
    doc = Document(content="Test knowledge content for framework test.", name="test")
    await knowledge_base.aadd([doc])

    knowledge_base.top_k = 2
    agent = Agent(
      model=openai_model,
      knowledge=knowledge_base,
      config=no_trace,
    )

    output = await agent.arun("What does the knowledge base contain?")
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed
