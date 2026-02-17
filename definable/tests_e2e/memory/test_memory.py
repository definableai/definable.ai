"""Integration tests for CognitiveMemory."""

import os
import tempfile
import time

import pytest

from definable.agents.testing import AgentTestCase, MockModel
from definable.memory.config import MemoryConfig
from definable.memory.memory import CognitiveMemory
from definable.memory.store.sqlite import SQLiteMemoryStore
from definable.memory.types import Episode
from definable.run.agent import (
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateStartedEvent,
  run_output_event_from_dict,
)


@pytest.fixture
async def cognitive_memory():
  """CognitiveMemory with temporary SQLite store."""
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  store = SQLiteMemoryStore(db_path=db_path)
  config = MemoryConfig(
    distillation_stage_0_age=1.0,
    distillation_stage_1_age=1.0,
    distillation_stage_2_age=1.0,
  )
  memory = CognitiveMemory(store=store, token_budget=500, config=config)
  yield memory
  await memory.close()
  os.unlink(db_path)


class FakeMessage:
  """Minimal message-like object for testing."""

  def __init__(self, role: str, content: str):
    self.role = role
    self.content = content


@pytest.mark.asyncio
class TestCognitiveMemoryStore:
  """Tests for storing messages."""

  async def test_store_messages(self, cognitive_memory):
    messages = [
      FakeMessage("user", "I live in San Francisco and work at Acme Corp."),
      FakeMessage("assistant", "That's great! San Francisco is a wonderful city."),
    ]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    # Verify episodes were stored
    episodes = await cognitive_memory.store.get_episodes(user_id="u1")
    assert len(episodes) == 2

  async def test_store_filters_system_messages(self, cognitive_memory):
    messages = [
      FakeMessage("system", "You are a helpful assistant."),
      FakeMessage("user", "Hello!"),
    ]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    episodes = await cognitive_memory.store.get_episodes(user_id="u1")
    # System messages should be filtered out
    assert len(episodes) == 1
    assert episodes[0].role == "user"

  async def test_store_with_topics(self, cognitive_memory):
    messages = [
      FakeMessage("user", "I'm learning Python programming and machine learning algorithms"),
    ]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    episodes = await cognitive_memory.store.get_episodes(user_id="u1")
    assert len(episodes) == 1
    assert len(episodes[0].topics) > 0


@pytest.mark.asyncio
class TestCognitiveMemoryRecall:
  """Tests for memory recall."""

  async def test_recall_empty_store(self, cognitive_memory):
    result = await cognitive_memory.recall("Hello", user_id="u1", session_id="s1")
    assert result is None

  async def test_recall_with_stored_episodes(self, cognitive_memory):
    # Store some messages
    messages = [
      FakeMessage("user", "I prefer Python programming with type hints"),
      FakeMessage("assistant", "I'll use Python with type hints for all code examples."),
    ]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    # Recall should find relevant memories
    result = await cognitive_memory.recall(
      "Write me some Python code",
      user_id="u1",
      session_id="s1",
    )
    assert result is not None
    assert result.tokens_used > 0
    assert "<memory_context>" in result.context

  async def test_recall_respects_token_budget(self, cognitive_memory):
    # Store many messages
    for i in range(20):
      messages = [
        FakeMessage("user", f"Message {i}: I'm discussing various programming topics in detail"),
      ]
      await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    # Recall with tight budget
    cognitive_memory.token_budget = 50
    result = await cognitive_memory.recall(
      "programming topics",
      user_id="u1",
      session_id="s1",
    )
    if result:
      assert result.tokens_used <= 50


@pytest.mark.asyncio
class TestCognitiveMemoryForget:
  """Tests for memory deletion."""

  async def test_forget_user(self, cognitive_memory):
    messages = [FakeMessage("user", "Remember this fact.")]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="s1")

    episodes = await cognitive_memory.store.get_episodes(user_id="u1")
    assert len(episodes) > 0

    await cognitive_memory.forget(user_id="u1")

    episodes = await cognitive_memory.store.get_episodes(user_id="u1")
    assert len(episodes) == 0

  async def test_forget_session(self, cognitive_memory):
    messages = [FakeMessage("user", "Session-specific data.")]
    await cognitive_memory.store_messages(messages, user_id="u1", session_id="sess-forget")

    await cognitive_memory.forget(session_id="sess-forget")

    episodes = await cognitive_memory.store.get_episodes(session_id="sess-forget")
    assert len(episodes) == 0


@pytest.mark.asyncio
class TestCognitiveMemoryDistillation:
  """Tests for manual distillation trigger."""

  async def test_manual_distillation(self, cognitive_memory):
    # Store episodes that are old enough for distillation
    now = time.time()
    old_time = now - 10  # 10 seconds ago (config has 1s threshold)

    ep = Episode(
      id="ep-manual-dist",
      user_id="u1",
      session_id="s1",
      role="user",
      content="A" * 250,  # Long content will get truncated
      compression_stage=0,
      created_at=old_time,
      last_accessed_at=old_time,
    )
    await cognitive_memory.store.initialize()
    await cognitive_memory.store.store_episode(ep)

    result = await cognitive_memory.run_distillation(user_id="u1")
    assert result.episodes_processed >= 1


@pytest.mark.asyncio
class TestCognitiveMemoryContextManager:
  """Tests for async context manager."""

  async def test_context_manager(self):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
      db_path = f.name

    try:
      store = SQLiteMemoryStore(db_path=db_path)
      memory = CognitiveMemory(store=store, token_budget=500)

      async with memory:
        assert memory._initialized
        messages = [FakeMessage("user", "Hello!")]
        await memory.store_messages(messages, session_id="s1")

      assert not memory._initialized
    finally:
      os.unlink(db_path)


@pytest.mark.asyncio
class TestCognitiveMemoryGracefulDegradation:
  """Tests for graceful failure handling."""

  async def test_recall_survives_store_error(self):
    """Memory recall should return None on errors, not raise."""

    class BrokenStore:
      async def initialize(self):
        raise RuntimeError("Store broken")

      async def close(self):
        pass

    memory = CognitiveMemory(store=BrokenStore(), token_budget=500)  # type: ignore
    result = await memory.recall("test", user_id="u1")
    assert result is None

  async def test_store_survives_error(self):
    """Memory store should log and not raise on errors."""

    class BrokenStore:
      async def initialize(self):
        raise RuntimeError("Store broken")

      async def close(self):
        pass

    memory = CognitiveMemory(store=BrokenStore(), token_budget=500)  # type: ignore
    # Should not raise
    await memory.store_messages(
      [FakeMessage("user", "test")],
      user_id="u1",
      session_id="s1",
    )


@pytest.mark.e2e
class TestMemoryEventSerialization:
  """Tests for memory event serialization round-trips."""

  def test_memory_recall_event_serialization(self):
    """MemoryRecallStarted/Completed events serialize and deserialize correctly."""
    started = MemoryRecallStartedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      query="test query",
    )
    d = started.to_dict()
    assert d["event"] == "MemoryRecallStarted"
    assert d["query"] == "test query"
    reconstructed = run_output_event_from_dict(d)
    assert isinstance(reconstructed, MemoryRecallStartedEvent)
    assert reconstructed.query == "test query"

    completed = MemoryRecallCompletedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      query="test query",
      tokens_used=150,
      chunks_included=3,
      chunks_available=5,
      duration_ms=25.0,
    )
    d = completed.to_dict()
    assert d["event"] == "MemoryRecallCompleted"
    assert d["tokens_used"] == 150
    assert d["chunks_included"] == 3
    assert d["chunks_available"] == 5
    assert d["duration_ms"] == 25.0
    reconstructed = run_output_event_from_dict(d)
    assert isinstance(reconstructed, MemoryRecallCompletedEvent)
    assert reconstructed.tokens_used == 150
    assert reconstructed.chunks_included == 3

  def test_memory_update_event_serialization(self):
    """MemoryUpdateStarted/Completed events with enriched fields serialize correctly."""
    from definable.run.agent import MemoryUpdateCompletedEvent

    started = MemoryUpdateStartedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      message_count=2,
    )
    d = started.to_dict()
    assert d["event"] == "MemoryUpdateStarted"
    assert d["message_count"] == 2
    reconstructed = run_output_event_from_dict(d)
    assert isinstance(reconstructed, MemoryUpdateStartedEvent)
    assert reconstructed.message_count == 2

    completed = MemoryUpdateCompletedEvent(
      run_id="r1",
      session_id="s1",
      agent_id="a1",
      agent_name="test",
      message_count=2,
      duration_ms=10.5,
    )
    d = completed.to_dict()
    assert d["event"] == "MemoryUpdateCompleted"
    assert d["message_count"] == 2
    assert d["duration_ms"] == 10.5


@pytest.mark.asyncio
@pytest.mark.e2e
class TestMemoryEventsIntegration(AgentTestCase):
  """Tests for memory event emission during agent runs."""

  async def test_memory_events_emitted_on_arun(self, cognitive_memory):
    """Agent.arun() emits MemoryRecallStarted/Completed and MemoryUpdateStarted events."""
    # Pre-store some messages so recall returns data
    await cognitive_memory.store_messages(
      [FakeMessage("user", "I like Python programming")],
      user_id="u1",
      session_id="s1",
    )

    mock_model = MockModel(responses=["Sure, I can help with Python!"])
    agent = self.create_agent(model=mock_model, memory=cognitive_memory)

    emitted_events = []
    original_emit = agent._emit

    def capture_emit(event):
      emitted_events.append(event)
      original_emit(event)

    agent._emit = capture_emit

    await agent.arun("Help me with Python", user_id="u1", session_id="s1")

    event_types = [type(e).__name__ for e in emitted_events]
    assert "MemoryRecallStartedEvent" in event_types
    assert "MemoryRecallCompletedEvent" in event_types
    assert "MemoryUpdateStartedEvent" in event_types

    # Verify recall started event
    recall_started = next(e for e in emitted_events if isinstance(e, MemoryRecallStartedEvent))
    assert recall_started.query == "Help me with Python"

    # Verify recall completed event
    recall_completed = next(e for e in emitted_events if isinstance(e, MemoryRecallCompletedEvent))
    assert recall_completed.query == "Help me with Python"
    assert recall_completed.duration_ms is not None
    assert recall_completed.duration_ms >= 0

    # Verify memory update started event
    update_started = next(e for e in emitted_events if isinstance(e, MemoryUpdateStartedEvent))
    assert update_started.message_count == 2  # User message + assistant response

  async def test_memory_events_emitted_on_stream(self, cognitive_memory):
    """Agent.arun_stream() yields memory events in correct order."""
    mock_model = MockModel(responses=["Hello!"])
    agent = self.create_agent(model=mock_model, memory=cognitive_memory)

    events = []
    async for evt in agent.arun_stream("Hello", user_id="u1", session_id="s1"):
      events.append(evt)

    event_types = [type(e).__name__ for e in events]
    assert "MemoryRecallStartedEvent" in event_types
    assert "MemoryRecallCompletedEvent" in event_types
    assert "MemoryUpdateStartedEvent" in event_types

    # Verify ordering: recall before run started, update after run completed
    recall_started_idx = event_types.index("MemoryRecallStartedEvent")
    recall_completed_idx = event_types.index("MemoryRecallCompletedEvent")
    run_started_idx = event_types.index("RunStartedEvent")
    run_completed_idx = event_types.index("RunCompletedEvent")
    update_started_idx = event_types.index("MemoryUpdateStartedEvent")

    assert recall_started_idx < recall_completed_idx
    assert recall_completed_idx < run_started_idx
    assert run_completed_idx < update_started_idx

  async def test_no_memory_events_without_memory(self):
    """No memory events emitted when agent has no memory."""
    mock_model = MockModel(responses=["Hello!"])
    agent = self.create_agent(model=mock_model)

    emitted_events = []
    original_emit = agent._emit

    def capture_emit(event):
      emitted_events.append(event)
      original_emit(event)

    agent._emit = capture_emit

    await agent.arun("Hello")

    event_types = [type(e).__name__ for e in emitted_events]
    assert "MemoryRecallStartedEvent" not in event_types
    assert "MemoryUpdateStartedEvent" not in event_types
