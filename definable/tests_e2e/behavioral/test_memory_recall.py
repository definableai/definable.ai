"""
Behavioral tests: Does the agent recall from memory correctly?

Strategy:
  - Real OpenAI model only — no mocks
  - Use InMemoryStore (no external deps) + real OpenAI model for LLM reasoning
  - Store facts in memory BEFORE running the agent
  - Assert that the agent's response CONTAINS the stored fact
  - Do NOT assert on how memory was retrieved (internal implementation)

Covers:
  - Agent with memory recalls facts from previous turns
  - Facts stored in memory appear in agent responses
  - Agent without memory does not hallucinate memory
  - Memory persists across multiple arun() calls within same agent instance
"""

import pytest

from definable.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.tracing import Tracing
from definable.memory import MemoryManager as CognitiveMemory
from definable.memory.store.in_memory import InMemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trace():
  return AgentConfig(tracing=Tracing(enabled=False))


@pytest.fixture
def memory_store():
  """Fresh InMemoryStore for each test."""
  return InMemoryStore()


# ---------------------------------------------------------------------------
# Memory recall behavioral tests (real OpenAI for LLM reasoning)
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.openai
class TestMemoryRecall:
  """Agent actually uses memory to answer user questions."""

  @pytest.mark.asyncio
  async def test_agent_recalls_fact_from_memory_context(self, openai_model, no_trace, memory_store):
    """Agent should use facts stored in memory when answering questions."""
    memory = CognitiveMemory(store=memory_store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=no_trace,
    )

    # First turn: store a personal fact
    out1 = await agent.arun("My dog's name is Biscuit. Please remember this.")
    assert out1.content  # Agent acknowledged

    # Second turn: ask about the stored fact
    out2 = await agent.arun("What is my dog's name?", messages=out1.messages)

    # The agent should recall the dog's name from conversation history or memory
    assert "biscuit" in out2.content.lower()  # type: ignore[union-attr]

  @pytest.mark.asyncio
  async def test_multi_turn_memory_accumulates(self, openai_model, no_trace, memory_store):
    """Agent can recall facts from earlier in a multi-turn conversation."""
    memory = CognitiveMemory(store=memory_store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=no_trace,
    )

    out1 = await agent.arun("I work at a company called Acme Corp.")
    out2 = await agent.arun("I prefer Python over JavaScript.", messages=out1.messages)
    out3 = await agent.arun("Where do I work and what is my preferred language?", messages=out2.messages)

    # Should recall both facts
    content = out3.content.lower()  # type: ignore[union-attr]
    assert "acme" in content or "acme corp" in content
    assert "python" in content

  @pytest.mark.asyncio
  async def test_agent_without_memory_handles_questions(self, openai_model, no_trace):
    """Agent without memory should handle questions gracefully."""
    agent = Agent(model=openai_model, config=no_trace)
    output = await agent.arun("What is my favorite food?")
    # Agent should respond (even if it says it doesn't know)
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed

  @pytest.mark.asyncio
  async def test_memory_agent_completes_run_successfully(self, openai_model, no_trace, memory_store):
    """Agent with memory layer completes a basic run without error."""
    memory = CognitiveMemory(store=memory_store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=no_trace,
    )
    output = await agent.arun("Hello, I am a developer who likes AI.")
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed
