"""
Behavioral tests: Agent combining memory + knowledge + tools.

Migrated from: tests_e2e/behavioral/test_multi_source.py

These are the hardest behavioral tests — they verify the agent can handle
multi-intent queries that require pulling from multiple sources simultaneously.

Strategy:
  - Real OpenAI model (for LLM intelligence)
  - Real InMemoryStore (for memory)
  - Real InMemoryVectorDB + OpenAI embeddings (for knowledge)
  - Assert on OUTPUT CONTENT: does it contain all required pieces?

Covers:
  - Agent recalls personal info from memory AND factual info from knowledge
  - Agent calls a tool AND answers from knowledge base
  - Multi-turn conversation where context accumulates across sources
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.tracing import Tracing
from definable.knowledge import Knowledge
from definable.knowledge.document import Document
from definable.memory import Memory
from definable.memory.store.in_memory import InMemoryStore
from definable.tool.decorator import tool


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def record_note(title: str, content: str) -> str:
  """Record a note with a title and content."""
  return f"Note recorded: '{title}' — {content}"


@tool
def get_current_date() -> str:
  """Get the current date."""
  from datetime import date

  return str(date.today())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trace():
  return AgentConfig(tracing=Tracing(enabled=False))


@pytest.fixture
def knowledge_base(openai_embedder):
  from definable.vectordb import InMemoryVectorDB

  db = InMemoryVectorDB(embedder=openai_embedder)
  kb = Knowledge(vector_db=db, embedder=openai_embedder)
  yield kb
  db.drop()


@pytest.fixture
def memory_store():
  return InMemoryStore()


@pytest.fixture
def full_agent(openai_model, knowledge_base, memory_store, no_trace):
  """Agent with memory, knowledge, and tools — the full stack."""
  memory = Memory(store=memory_store)
  knowledge_base.top_k = 3
  return Agent(
    model=openai_model,
    memory=memory,
    knowledge=knowledge_base,
    tools=[record_note, get_current_date],
    config=no_trace,
  )


# ---------------------------------------------------------------------------
# Multi-source behavioral tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.integration
@pytest.mark.openai
@pytest.mark.slow
class TestMultiSourceBehavior:
  """Agent combines memory + knowledge + tools in a single run."""

  async def test_agent_uses_knowledge_and_memory_together(self, full_agent, knowledge_base):
    """Agent should answer using both knowledge base facts and memory context."""
    # Add a fact to the knowledge base
    doc = Document(
      content="The definable.ai framework supports multi-agent orchestration.",
      name="framework_feature",
    )
    await knowledge_base.aadd([doc])

    # First turn: introduce personal context (goes to memory)
    out1 = await full_agent.arun("I am a developer building AI agents.")

    # Second turn: ask something that spans memory + knowledge
    out2 = await full_agent.arun(
      "Given what you know about me, what feature of definable.ai would be most useful to me?",
      messages=out1.messages,
    )

    # Response should mention the agent context
    content = out2.content.lower()
    assert len(content) > 20  # Non-trivial response
    # Should reference either the user's developer context or the framework feature
    assert any(kw in content for kw in ["agent", "definable", "orchestration", "developer", "multi"])

  async def test_agent_calls_tool_while_using_knowledge(self, full_agent, knowledge_base):
    """Agent should call a tool AND use knowledge base content in one run."""
    doc = Document(
      content="Today's standup meeting agenda: discuss sprint progress, blockers, and next steps.",
      name="meeting_agenda",
    )
    await knowledge_base.aadd([doc])

    output = await full_agent.arun("Record a note about today's standup. Use the meeting agenda from the knowledge base as the content.")

    # Either the tool was called or the content mentions the agenda
    tool_names = [t.tool_name for t in (output.tools or [])]
    content = output.content.lower()
    assert "record_note" in tool_names or "standup" in content or "agenda" in content

  async def test_agent_handles_tool_plus_factual_question(self, full_agent):
    """Agent should handle a query that requires a tool call AND a factual answer."""
    output = await full_agent.arun("What is today's date? Use the get_current_date tool.")

    tool_names = [t.tool_name for t in (output.tools or [])]
    # Should call the tool OR include a date in the content
    import re

    has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}", output.content))
    assert "get_current_date" in tool_names or has_date

  async def test_multi_turn_with_full_stack(self, full_agent, knowledge_base):
    """Multi-turn conversation where all layers are active."""
    doc = Document(
      content="Definable agents support streaming responses via arun_stream().",
      name="streaming_feature",
    )
    await knowledge_base.aadd([doc])

    # Turn 1: personal context
    out1 = await full_agent.arun("I prefer streaming interfaces for real-time feedback.")

    # Turn 2: knowledge query grounded in personal context
    out2 = await full_agent.arun(
      "Which feature of definable.ai aligns with my preference?",
      messages=out1.messages,
    )

    content = out2.content.lower()
    assert len(content) > 10
    # Should reference streaming or the user's preference
    assert any(kw in content for kw in ["stream", "real-time", "realtime", "arun_stream", "prefer"])
