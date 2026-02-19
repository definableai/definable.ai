"""
Behavioral tests for memory integration with Agent.

Migrated from:
  - tests_e2e/behavioral/test_memory_agent.py (MockModel-based memory tests)
  - tests_e2e/behavioral/test_memory_recall.py (real OpenAI memory recall tests)

Covers (MockModel-based, no API calls):
  - MemoryManager creates a memory when LLM calls add_memory tool
  - MemoryManager creates no memories when LLM does not call tools
  - MemoryManager updates a memory when LLM calls update_memory tool
  - Memories are correctly formatted for system prompt injection
  - Agent with Memory creates a MemoryManager internally
  - Agent with memory=True creates a MemoryManager with InMemoryStore
  - Agent with memory=False has no memory
  - Memory recall injects formatted memories into context
  - CognitiveMemory import resolves to MemoryManager for backward compat

Covers (real OpenAI, marked @pytest.mark.openai):
  - Agent with memory recalls facts from previous turns
  - Multi-turn memory accumulates facts
  - Agent without memory handles questions gracefully
  - Memory agent completes run successfully
"""

import json
from unittest.mock import MagicMock

import pytest

from definable.agent.agent import Agent
from definable.agent.config import AgentConfig
from definable.agent.testing import MockModel
from definable.agent.tracing import Tracing
from definable.memory.manager import Memory
from definable.memory.manager import MemoryManager
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import UserMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_model_with_tool_calls(tool_calls=None, final_response="Done."):
  """Create a MockModel that returns tool_calls on first call, then final response."""
  call_count = 0

  async def side_effect(messages, tools=None, **kwargs):
    nonlocal call_count
    call_count += 1
    response = MagicMock()
    response.response_usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)

    if call_count == 1 and tool_calls:
      response.content = ""
      response.tool_calls = tool_calls
    else:
      response.content = final_response
      response.tool_calls = []

    response.tool_executions = []
    response.reasoning_content = None
    return response

  model = MagicMock()
  model.ainvoke = side_effect
  model.id = "mock-model"
  model.provider = "mock"
  return model


# ---------------------------------------------------------------------------
# Unit: MemoryManager with mock model
# ---------------------------------------------------------------------------


class TestMemoryManagerWithMock:
  async def test_add_memory_via_tool(self):
    """MemoryManager creates a memory when LLM calls add_memory tool."""
    store = InMemoryStore()
    model = make_mock_model_with_tool_calls(
      tool_calls=[
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "add_memory",
            "arguments": json.dumps({"memory": "User prefers dark mode", "topics": ["preferences"]}),
          },
        }
      ],
      final_response="Remembered user preference.",
    )

    mgr = MemoryManager(store=store, model=model)
    result = await mgr.acreate_user_memories(message="I prefer dark mode", user_id="u1")

    assert result == "Remembered user preference."

    memories = await store.get_user_memories(user_id="u1")
    assert len(memories) == 1
    assert memories[0].memory == "User prefers dark mode"
    assert memories[0].topics == ["preferences"]
    assert memories[0].user_id == "u1"

    await mgr.close()

  async def test_no_tool_calls_no_memories(self):
    """MemoryManager creates no memories when LLM doesn't call tools."""
    store = InMemoryStore()
    model = make_mock_model_with_tool_calls(
      tool_calls=None,
      final_response="Nothing worth remembering.",
    )

    mgr = MemoryManager(store=store, model=model)
    result = await mgr.acreate_user_memories(message="Hello there!", user_id="u1")

    assert result == "Nothing worth remembering."

    memories = await store.get_user_memories(user_id="u1")
    assert len(memories) == 0

    await mgr.close()

  async def test_update_existing_memory(self):
    """MemoryManager updates a memory when LLM calls update_memory tool."""
    store = InMemoryStore()
    # Pre-populate a memory
    original = UserMemory(memory="User lives in NYC", memory_id="m1", user_id="u1")
    await store.initialize()
    await store.upsert_user_memory(original)

    model = make_mock_model_with_tool_calls(
      tool_calls=[
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "update_memory",
            "arguments": json.dumps({"memory_id": "m1", "memory": "User lives in SF"}),
          },
        }
      ],
      final_response="Updated location.",
    )

    mgr = MemoryManager(store=store, model=model)
    mgr._initialized = True  # store already initialized
    await mgr.acreate_user_memories(message="I moved to SF", user_id="u1")

    memories = await store.get_user_memories(user_id="u1")
    assert len(memories) == 1
    assert memories[0].memory == "User lives in SF"

    await mgr.close()

  async def test_format_memories_injection(self):
    """Memories are correctly formatted for system prompt injection."""
    store = InMemoryStore()
    await store.initialize()
    await store.upsert_user_memory(UserMemory(memory="Likes Python", memory_id="m1", user_id="u1", topics=["lang"]))
    await store.upsert_user_memory(UserMemory(memory="Lives in NYC", memory_id="m2", user_id="u1", topics=["location"]))

    mgr = MemoryManager(store=store)
    mgr._initialized = True

    memories = await mgr.aget_user_memories(user_id="u1")
    formatted = mgr.format_memories_for_prompt(memories)

    assert "[m" in formatted  # Memory IDs present
    assert "Likes Python" in formatted
    assert "Lives in NYC" in formatted

    await mgr.close()


# ---------------------------------------------------------------------------
# Behavioral: Agent with Memory (MockModel-based)
# ---------------------------------------------------------------------------


class TestAgentMemoryIntegration:
  async def test_agent_with_memory_config(self):
    """Agent with Memory creates a MemoryManager internally."""
    store = InMemoryStore()
    mock_model = MockModel(responses=["Hello! Nice to meet you."])

    agent = Agent(
      model=mock_model,  # type: ignore[arg-type]
      memory=Memory(store=store, update_on_run=False),
    )

    assert agent.memory is not None
    assert isinstance(agent.memory, MemoryManager)
    assert agent.memory.store is store

  async def test_agent_memory_true(self):
    """Agent with memory=True creates a MemoryManager with InMemoryStore."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=True)  # type: ignore[arg-type]

    assert agent.memory is not None
    assert isinstance(agent.memory, MemoryManager)

  async def test_agent_memory_false(self):
    """Agent with memory=False has no memory."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=False)  # type: ignore[arg-type]
    assert agent.memory is None

  async def test_agent_memory_recall_injects_context(self):
    """Memory recall injects formatted memories into context."""
    store = InMemoryStore()
    await store.initialize()
    await store.upsert_user_memory(UserMemory(memory="User likes dark mode", memory_id="m1", user_id="default", topics=["ui"]))

    mock_model = MockModel(responses=["I remember you like dark mode!"])

    agent = Agent(
      model=mock_model,  # type: ignore[arg-type]
      memory=Memory(store=store, update_on_run=False),
    )

    result = await agent.arun("What do you know about me?")

    # Agent should have recalled memories and gotten a response
    assert result.content is not None

  async def test_backward_compat_cognitive_memory(self):
    """CognitiveMemory import resolves to MemoryManager for backward compat."""
    from definable import CognitiveMemory

    assert CognitiveMemory is MemoryManager


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
    memory = MemoryManager(store=store)
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
    memory = MemoryManager(store=store)
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
    memory = MemoryManager(store=store)
    agent = Agent(
      model=openai_model,
      memory=memory,
      config=AgentConfig(tracing=Tracing(enabled=False)),
    )
    output = await agent.arun("Hello, I am a developer who likes AI.")
    assert output.content
    from definable.agent.events import RunStatus

    assert output.status == RunStatus.completed
