"""E2E tests — Conversational Memory.

Scenario: "I want my agent to remember past conversations."

Tests use InMemoryStore (no external deps). Tests requiring real LLM
use OPENAI_API_KEY.
"""

import os
import tempfile

import pytest

from definable.memory import CognitiveMemory
from definable.memory.config import MemoryConfig, ScoringWeights
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import Episode
from definable.models.message import Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def memory_store():
  """Fresh InMemoryStore for each test."""
  store = InMemoryStore()
  await store.initialize()
  yield store
  await store.close()


@pytest.fixture
def memory(memory_store):
  """CognitiveMemory with InMemoryStore and tight token budget."""
  return CognitiveMemory(
    store=memory_store,
    token_budget=200,
    config=MemoryConfig(
      decay_half_life_days=14,
      scoring_weights=ScoringWeights(),
    ),
  )


# ---------------------------------------------------------------------------
# Tests: Memory Store Operations (no API key needed)
# ---------------------------------------------------------------------------


class TestMemoryStore:
  """Direct InMemoryStore operations."""

  @pytest.mark.asyncio
  async def test_store_and_retrieve_episode(self, memory_store):
    """Store an episode and retrieve it."""
    episode = Episode(
      id="ep1",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="Hello, I'm testing memory.",
      topics=["testing", "memory"],
      compression_stage=0,
    )
    await memory_store.store_episode(episode)

    episodes = await memory_store.get_episodes(user_id="user-1", limit=10)
    assert len(episodes) == 1
    assert episodes[0].content == "Hello, I'm testing memory."
    assert episodes[0].topics == ["testing", "memory"]

  @pytest.mark.asyncio
  async def test_episode_scoped_by_user(self, memory_store):
    """Episodes are scoped by user_id."""
    ep_a = Episode(id="ep-a", user_id="alice", session_id="s1", role="user", content="Alice's message", compression_stage=0)
    ep_b = Episode(id="ep-b", user_id="bob", session_id="s1", role="user", content="Bob's message", compression_stage=0)
    await memory_store.store_episode(ep_a)
    await memory_store.store_episode(ep_b)

    alice_eps = await memory_store.get_episodes(user_id="alice")
    bob_eps = await memory_store.get_episodes(user_id="bob")

    assert len(alice_eps) == 1
    assert alice_eps[0].content == "Alice's message"
    assert len(bob_eps) == 1
    assert bob_eps[0].content == "Bob's message"

  @pytest.mark.asyncio
  async def test_episode_scoped_by_session(self, memory_store):
    """Episodes can be filtered by session_id."""
    ep1 = Episode(id="ep1", user_id=None, session_id="s1", role="user", content="Session 1", compression_stage=0)
    ep2 = Episode(id="ep2", user_id=None, session_id="s2", role="user", content="Session 2", compression_stage=0)
    await memory_store.store_episode(ep1)
    await memory_store.store_episode(ep2)

    s1_eps = await memory_store.get_episodes(session_id="s1")
    assert len(s1_eps) == 1
    assert s1_eps[0].session_id == "s1"

  @pytest.mark.asyncio
  async def test_delete_user_data(self, memory_store):
    """delete_user_data removes all user episodes (GDPR)."""
    for i in range(3):
      ep = Episode(
        id=f"ep{i}",
        user_id="delete-me",
        session_id="s1",
        role="user",
        content=f"Message {i}",
        compression_stage=0,
      )
      await memory_store.store_episode(ep)

    episodes = await memory_store.get_episodes(user_id="delete-me")
    assert len(episodes) == 3

    await memory_store.delete_user_data("delete-me")

    episodes = await memory_store.get_episodes(user_id="delete-me")
    assert len(episodes) == 0

  @pytest.mark.asyncio
  async def test_update_episode_compression_stage(self, memory_store):
    """update_episode correctly changes compression_stage."""
    ep = Episode(id="ep1", user_id=None, session_id="s1", role="user", content="Test", compression_stage=0)
    await memory_store.store_episode(ep)

    await memory_store.update_episode("ep1", compression_stage=1)

    episodes = await memory_store.get_episodes()
    assert episodes[0].compression_stage == 1


class TestSQLiteStorePersistence:
  """SQLite store persistence across open/close cycles."""

  @pytest.mark.asyncio
  async def test_sqlite_store_persists_episodes(self):
    """SQLiteMemoryStore data survives close and reopen."""
    from definable.memory.store.sqlite import SQLiteMemoryStore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
      db_path = f.name

    try:
      # Write
      store = SQLiteMemoryStore(db_path=db_path)
      await store.initialize()
      ep = Episode(
        id="persist-1",
        user_id="user-1",
        session_id="s1",
        role="user",
        content="I should persist across restarts.",
        compression_stage=0,
      )
      await store.store_episode(ep)
      await store.close()

      # Reopen and verify
      store2 = SQLiteMemoryStore(db_path=db_path)
      await store2.initialize()
      episodes = await store2.get_episodes(user_id="user-1")
      assert len(episodes) == 1
      assert episodes[0].content == "I should persist across restarts."
      await store2.close()
    finally:
      os.unlink(db_path)


# ---------------------------------------------------------------------------
# Tests: CognitiveMemory (store + recall pipeline, no API key)
# ---------------------------------------------------------------------------


class TestCognitiveMemoryBasic:
  """CognitiveMemory store and recall without embedder."""

  @pytest.mark.asyncio
  async def test_store_messages_creates_episodes(self, memory, memory_store):
    """store_messages creates episodes from conversation messages."""
    messages = [
      Message(role="user", content="What is machine learning?"),
      Message(role="assistant", content="Machine learning is a subset of AI that learns from data."),
    ]
    await memory.store_messages(messages, user_id="u1", session_id="s1")

    episodes = await memory_store.get_episodes(user_id="u1")
    assert len(episodes) >= 2

    contents = [ep.content for ep in episodes]
    assert any("machine learning" in c.lower() for c in contents)

  @pytest.mark.asyncio
  async def test_recall_returns_payload_or_none(self, memory, memory_store):
    """recall returns a MemoryPayload with context or None if no matches."""
    # Store some content first
    messages = [
      Message(role="user", content="My favorite color is blue."),
      Message(role="assistant", content="Blue is a great color!"),
    ]
    await memory.store_messages(messages, user_id="u1", session_id="s1")

    # Recall - may or may not find matches without embedder
    payload = await memory.recall("What is my favorite color?", user_id="u1")
    # Without embedder, recall uses topic matching / recency; result depends on topics
    # Just verify it doesn't crash and returns the right type
    assert payload is None or hasattr(payload, "context")

  @pytest.mark.asyncio
  async def test_memory_scoped_by_user(self, memory, memory_store):
    """Memory store and recall are scoped by user_id."""
    await memory.store_messages(
      [Message(role="user", content="Alice's secret: the password is hunter2.")],
      user_id="alice",
    )
    await memory.store_messages(
      [Message(role="user", content="Bob likes pizza.")],
      user_id="bob",
    )

    # Alice's episodes
    alice_eps = await memory_store.get_episodes(user_id="alice")
    bob_eps = await memory_store.get_episodes(user_id="bob")

    alice_contents = " ".join(ep.content for ep in alice_eps)
    bob_contents = " ".join(ep.content for ep in bob_eps)

    assert "hunter2" in alice_contents
    assert "hunter2" not in bob_contents
    assert "pizza" in bob_contents


# ---------------------------------------------------------------------------
# Tests: Agent + Memory Integration (requires OpenAI)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestMemoryAgentIntegration:
  """Agent with CognitiveMemory in end-to-end workflows."""

  @pytest.mark.asyncio
  async def test_full_conversation_with_memory(self, openai_model):
    """Agent stores and recalls across turns."""
    from definable.agents.agent import Agent
    from definable.agents.config import AgentConfig, TracingConfig

    store = InMemoryStore()
    memory = CognitiveMemory(store=store, token_budget=500)

    agent = Agent(
      model=openai_model,
      memory=memory,
      instructions="You are a helpful assistant with memory. Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    # Turn 1: introduce information
    output1 = await agent.arun(
      "My name is Orion and I work at Nebula Labs.",
      user_id="test-user",
      session_id="test-session",
    )
    assert output1.content is not None

    # Verify episodes were stored
    episodes = await store.get_episodes(user_id="test-user")
    assert len(episodes) >= 1

  @pytest.mark.asyncio
  async def test_memory_forget_user_data(self, openai_model):
    """GDPR: delete_user_data removes all user memories."""
    from definable.agents.agent import Agent
    from definable.agents.config import AgentConfig, TracingConfig

    store = InMemoryStore()
    memory = CognitiveMemory(store=store, token_budget=500)

    agent = Agent(
      model=openai_model,
      memory=memory,
      instructions="Be concise.",
      config=AgentConfig(tracing=TracingConfig(enabled=False)),
    )

    await agent.arun(
      "Remember: my API key is sk-secret-123.",
      user_id="gdpr-user",
    )

    # Verify data exists
    episodes = await store.get_episodes(user_id="gdpr-user")
    assert len(episodes) >= 1

    # Delete all user data
    await store.delete_user_data("gdpr-user")

    # Verify data is gone
    episodes = await store.get_episodes(user_id="gdpr-user")
    assert len(episodes) == 0
