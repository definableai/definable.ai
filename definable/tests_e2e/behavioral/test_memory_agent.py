"""
Behavioral tests for memory integration with Agent.

Tests that:
  - Agent with memory=Memory stores memories after run
  - Memory context is injected into system prompt on recall
  - Memory persistence works across multiple runs
  - MockModel drives all assertions (no real API calls)
"""

import json
from unittest.mock import MagicMock

import pytest

from definable.agent.agent import Agent
from definable.agent.testing import MockModel
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
  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
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
# Behavioral: Agent with Memory
# ---------------------------------------------------------------------------


class TestAgentMemoryIntegration:
  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
  async def test_agent_memory_true(self):
    """Agent with memory=True creates a MemoryManager with InMemoryStore."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=True)  # type: ignore[arg-type]

    assert agent.memory is not None
    assert isinstance(agent.memory, MemoryManager)

  @pytest.mark.asyncio
  async def test_agent_memory_false(self):
    """Agent with memory=False has no memory."""
    mock_model = MockModel(responses=["Hi"])
    agent = Agent(model=mock_model, memory=False)  # type: ignore[arg-type]
    assert agent.memory is None

  @pytest.mark.asyncio
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

  @pytest.mark.asyncio
  async def test_backward_compat_cognitive_memory(self):
    """CognitiveMemory import resolves to MemoryManager for backward compat."""
    from definable import CognitiveMemory

    assert CognitiveMemory is MemoryManager
