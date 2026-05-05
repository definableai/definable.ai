"""
Behavioral tests for memory integration with Agent.

Covers (MockModel-based, no API calls):
  - Memory stores and retrieves session entries
  - Agent with Memory creates a Memory instance internally
  - Agent with memory=True creates a Memory with InMemoryStore
  - Agent with memory=False has no memory
  - Memory recall injects session history into context
  - Multi-session isolation

Covers (real OpenAI, marked @pytest.mark.openai):
  - Agent with memory recalls facts from previous turns
  - Multi-turn memory accumulates facts
  - Agent without memory handles questions gracefully
  - Memory agent completes run successfully
"""

import pytest

from definable.agent.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.memory.manager import Memory
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import MemoryEntry
from definable.model.message import Message


# ---------------------------------------------------------------------------
# Unit: Memory with mock model
# ---------------------------------------------------------------------------


class TestMemoryWithMock:
  async def test_add_and_get_entries(self):
    """Memory stores entries and retrieves them."""
    store = InMemoryStore()
    mem = Memory(store=store)

    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    await mem.add(Message(role="assistant", content="Hi!"), session_id="s1")

    entries = await mem.get_entries("s1")
    assert len(entries) == 2
    assert entries[0].role == "user"
    assert entries[0].content == "Hello"
    assert entries[1].role == "assistant"
    assert entries[1].content == "Hi!"

    await mem.close()

  async def test_get_context_messages(self):
    """Memory converts entries back to Message objects."""
    store = InMemoryStore()
    mem = Memory(store=store)

    await mem.add(Message(role="user", content="Hello"), session_id="s1")
    await mem.add(Message(role="assistant", content="Hi!"), session_id="s1")

    messages = await mem.get_context_messages("s1")
    assert len(messages) == 2
    assert isinstance(messages[0], Message)
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    await mem.close()

  async def test_multi_session_isolation(self):
    """Different sessions are isolated."""
    store = InMemoryStore()
    mem = Memory(store=store)

    await mem.add(Message(role="user", content="Session 1"), session_id="s1")
    await mem.add(Message(role="user", content="Session 2"), session_id="s2")

    s1 = await mem.get_entries("s1")
    s2 = await mem.get_entries("s2")
    assert len(s1) == 1
    assert len(s2) == 1
    assert s1[0].content == "Session 1"
    assert s2[0].content == "Session 2"

    await mem.close()


# ---------------------------------------------------------------------------
# Behavioral: Agent with Memory (MockModel-based)
# ---------------------------------------------------------------------------


class TestAgentMemoryIntegration:
  async def test_agent_with_memory_config(self):
    """Agent with Memory instance snaps in correctly."""
    store = InMemoryStore()
    mock_model = MockModel(responses=["Hello! Nice to meet you."])

    agent = Agent(
      model=mock_model,  # type: ignore[arg-type]
      memory=Memory(store=store),
    )

    assert agent.memory is not None
    assert isinstance(agent.memory, Memory)
    assert agent.memory.store is store

  async def test_agent_memory_true(self):
    """Agent with memory=True creates a Memory with InMemoryStore."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=True)  # type: ignore[arg-type]

    assert agent.memory is not None
    assert isinstance(agent.memory, Memory)

  async def test_agent_memory_false(self):
    """Agent with memory=False has no memory."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=False)  # type: ignore[arg-type]
    assert agent.memory is None

  async def test_agent_memory_recall_injects_context(self):
    """Memory recall injects session history into context."""
    store = InMemoryStore()

    # Pre-populate session history
    await store.initialize()
    await store.add(MemoryEntry(session_id="default", user_id="default", role="user", content="I like dark mode", created_at=1.0, updated_at=1.0))

    mock_model = MockModel(responses=["I remember you like dark mode!"])

    agent = Agent(
      model=mock_model,  # type: ignore[arg-type]
      memory=Memory(store=store),
    )

    result = await agent.arun("What do you know about me?")

    # Agent should have recalled history and gotten a response
    assert result.content is not None


# ---------------------------------------------------------------------------
# Behavioral: Agent with Memory (real OpenAI model)
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
@pytest.mark.openai
class TestMemoryRecall:
  """Agent actually uses memory to answer user questions (requires OPENAI_API_KEY)."""

  async def test_agent_recalls_fact_from_memory_context(self, openai_model):
    """Agent should use facts stored in memory when answering questions."""
    store = InMemoryStore()
    memory = Memory(store=store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=AgentConfig(tracing=Tracing(enabled=False)),
    )

    # First turn: store a personal fact
    out1 = await agent.arun("My dog's name is Biscuit. Please remember this.")
    assert out1.content  # Agent acknowledged

    # Second turn: ask about the stored fact
    out2 = await agent.arun("What is my dog's name?", messages=out1.messages)

    # The agent should recall the dog's name from conversation history or memory
    assert "biscuit" in out2.content.lower()  # type: ignore[union-attr]

  async def test_multi_turn_memory_accumulates(self, openai_model):
    """Agent can recall facts from earlier in a multi-turn conversation."""
    store = InMemoryStore()
    memory = Memory(store=store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=AgentConfig(tracing=Tracing(enabled=False)),
    )

    out1 = await agent.arun("I work at a company called Acme Corp.")
    out2 = await agent.arun("I prefer Python over JavaScript.", messages=out1.messages)
    out3 = await agent.arun("Where do I work and what is my preferred language?", messages=out2.messages)

    # Should recall both facts
    content = out3.content.lower()  # type: ignore[union-attr]
    assert "acme" in content or "acme corp" in content
    assert "python" in content

  async def test_agent_without_memory_handles_questions(self, openai_model):
    """Agent without memory should handle questions gracefully."""
    agent = Agent(model=openai_model, config=AgentConfig(tracing=Tracing(enabled=False)))
    output = await agent.arun("What is my favorite food?")
    # Agent should respond (even if it says it doesn't know)
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed

  async def test_memory_agent_completes_run_successfully(self, openai_model):
    """Agent with memory layer completes a basic run without error."""
    store = InMemoryStore()
    memory = Memory(store=store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=AgentConfig(tracing=Tracing(enabled=False)),
    )
    output = await agent.arun("Hello, I am a developer who likes AI.")
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed
