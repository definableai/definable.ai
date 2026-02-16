"""Tests for Mem0MemoryStore."""

import asyncio
import contextlib
import os
import time

import pytest

from definable.memory.store.mem0 import Mem0MemoryStore
from definable.memory.types import Episode, KnowledgeAtom, Procedure

TEST_USER_IDS = ["user-1", "user-del", "user-keep", "user-remove", "u1", Mem0MemoryStore._SYSTEM_USER_ID]


@pytest.fixture
async def mem0_store():
  """Mem0MemoryStore backed by the hosted Mem0 API."""
  api_key = os.environ.get("MEM0_API_KEY", "")
  if not api_key:
    pytest.skip("MEM0_API_KEY not set")
  store = Mem0MemoryStore(api_key=api_key)
  try:
    await store.initialize()
  except Exception as e:
    pytest.skip(f"Mem0 not available: {e}")

  # Pre-cleanup: remove leftover data from prior runs so tests start clean.
  # Mark users as deleted so _get_all_by_type skips stale API data and
  # only returns entries from the local cache (written during this test).
  for uid in TEST_USER_IDS:
    with contextlib.suppress(Exception):
      await store._client.delete_users(user_id=uid)
    store._deleted_user_ids.add(uid)
  await asyncio.sleep(2)  # Let Mem0 propagate deletes

  yield store

  # Post-cleanup
  for uid in TEST_USER_IDS:
    with contextlib.suppress(Exception):
      await store._client.delete_users(user_id=uid)
  await store.close()


@pytest.mark.mem0
@pytest.mark.asyncio
class TestMem0StoreEpisodes:
  """Tests for episode storage and retrieval."""

  async def test_store_and_retrieve_episode(self, mem0_store):
    now = time.time()
    episode = Episode(
      id="ep-1",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="Hello, I live in San Francisco.",
      topics=["san_francisco", "location"],
      sentiment=0.0,
      token_count=10,
      created_at=now,
      last_accessed_at=now,
    )
    result_id = await mem0_store.store_episode(episode)
    assert result_id == "ep-1"

    episodes = await mem0_store.get_episodes(user_id="user-1")
    assert len(episodes) == 1
    assert episodes[0].id == "ep-1"
    assert episodes[0].content == "Hello, I live in San Francisco."
    assert episodes[0].topics == ["san_francisco", "location"]

  async def test_auto_generate_id(self, mem0_store):
    now = time.time()
    episode = Episode(
      id="",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="Auto ID episode",
      created_at=now,
      last_accessed_at=now,
    )
    result_id = await mem0_store.store_episode(episode)
    assert result_id != ""
    assert len(result_id) == 36  # UUID4 length

  async def test_get_episodes_by_session(self, mem0_store):
    now = time.time()
    for i in range(3):
      episode = Episode(
        id=f"ep-{i}",
        user_id="user-1",
        session_id="sess-A" if i < 2 else "sess-B",
        role="user",
        content=f"Message {i}",
        created_at=now + i,
        last_accessed_at=now + i,
      )
      await mem0_store.store_episode(episode)

    sess_a = await mem0_store.get_episodes(user_id="user-1", session_id="sess-A")
    assert len(sess_a) == 2

    sess_b = await mem0_store.get_episodes(user_id="user-1", session_id="sess-B")
    assert len(sess_b) == 1

  async def test_get_episodes_order_desc(self, mem0_store):
    now = time.time()
    for i in range(3):
      episode = Episode(
        id=f"ep-ord-{i}",
        user_id="user-1",
        session_id="sess-ord",
        role="user",
        content=f"Message {i}",
        created_at=now + i,
        last_accessed_at=now + i,
      )
      await mem0_store.store_episode(episode)

    # Filter by session to isolate from leftover data across runs
    episodes = await mem0_store.get_episodes(user_id="user-1", session_id="sess-ord")
    # Most recent first
    assert episodes[0].id == "ep-ord-2"
    assert episodes[-1].id == "ep-ord-0"

  async def test_update_episode(self, mem0_store):
    now = time.time()
    episode = Episode(
      id="ep-upd",
      user_id="user-1",
      session_id="sess-1",
      role="assistant",
      content="Original content",
      compression_stage=0,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_episode(episode)

    await mem0_store.update_episode("ep-upd", content="Updated content", compression_stage=1)

    episodes = await mem0_store.get_episodes(user_id="user-1")
    assert episodes[0].content == "Updated content"
    assert episodes[0].compression_stage == 1

  async def test_get_episodes_for_distillation(self, mem0_store):
    old_time = time.time() - 7200  # 2 hours ago
    recent_time = time.time()

    for i, (t, stage) in enumerate([(old_time, 0), (recent_time, 0), (old_time, 1)]):
      ep = Episode(
        id=f"ep-dist-{i}",
        user_id="user-1",
        session_id="sess-1",
        role="user",
        content=f"Content {i}",
        compression_stage=stage,
        created_at=t,
        last_accessed_at=t,
      )
      await mem0_store.store_episode(ep)

    # Get stage-0 episodes older than 1 hour ago
    threshold = time.time() - 3600
    result = await mem0_store.get_episodes_for_distillation(stage=0, older_than=threshold)
    assert len(result) == 1
    assert result[0].id == "ep-dist-0"

  async def test_episode_stage_filtering(self, mem0_store):
    now = time.time()
    for i, stage in enumerate([0, 1, 2, 3]):
      ep = Episode(
        id=f"ep-stage-{i}",
        user_id="user-1",
        session_id="sess-1",
        role="user",
        content=f"Stage {stage} content",
        compression_stage=stage,
        created_at=now + i,
        last_accessed_at=now + i,
      )
      await mem0_store.store_episode(ep)

    result = await mem0_store.get_episodes(user_id="user-1", min_stage=1, max_stage=2)
    assert len(result) == 2
    stages = {ep.compression_stage for ep in result}
    assert stages == {1, 2}

  async def test_search_episodes_by_embedding_returns_empty(self, mem0_store):
    results = await mem0_store.search_episodes_by_embedding(
      [0.1, 0.2, 0.3],
      user_id="user-1",
      top_k=5,
    )
    assert results == []


@pytest.mark.mem0
@pytest.mark.asyncio
class TestMem0StoreAtoms:
  """Tests for knowledge atom storage and retrieval."""

  async def test_store_and_retrieve_atom(self, mem0_store):
    now = time.time()
    atom = KnowledgeAtom(
      id="atom-1",
      user_id="user-1",
      subject="user",
      predicate="lives_in",
      object="San Francisco",
      content="user lives in San Francisco",
      confidence=0.9,
      topics=["location"],
      token_count=5,
      created_at=now,
      last_accessed_at=now,
    )
    result_id = await mem0_store.store_atom(atom)
    assert result_id == "atom-1"

    atoms = await mem0_store.get_atoms(user_id="user-1")
    assert len(atoms) == 1
    assert atoms[0].content == "user lives in San Francisco"
    assert atoms[0].confidence == 0.9
    assert atoms[0].topics == ["location"]

  async def test_find_similar_atom(self, mem0_store):
    now = time.time()
    atom = KnowledgeAtom(
      id="atom-sim",
      user_id="user-1",
      subject="user",
      predicate="prefers",
      object="Python",
      content="user prefers Python",
      confidence=0.8,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_atom(atom)

    found = await mem0_store.find_similar_atom("user", "prefers", user_id="user-1")
    assert found is not None
    assert found.id == "atom-sim"

    # Case-insensitive match
    found_upper = await mem0_store.find_similar_atom("User", "Prefers", user_id="user-1")
    assert found_upper is not None
    assert found_upper.id == "atom-sim"

    not_found = await mem0_store.find_similar_atom("user", "dislikes", user_id="user-1")
    assert not_found is None

  async def test_update_atom(self, mem0_store):
    now = time.time()
    atom = KnowledgeAtom(
      id="atom-upd",
      user_id="user-1",
      subject="user",
      predicate="works_at",
      object="Acme",
      content="user works at Acme",
      confidence=0.5,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_atom(atom)

    await mem0_store.update_atom("atom-upd", confidence=0.8, reinforcement_count=2)

    atoms = await mem0_store.get_atoms(user_id="user-1")
    assert atoms[0].confidence == 0.8
    assert atoms[0].reinforcement_count == 2

  async def test_prune_atoms(self, mem0_store):
    now = time.time()
    for i, conf in enumerate([0.05, 0.15, 0.5, 0.9]):
      atom = KnowledgeAtom(
        id=f"atom-pr-{i}",
        user_id="user-1",
        subject="s",
        predicate="p",
        object="o",
        content=f"fact {i}",
        confidence=conf,
        created_at=now,
        last_accessed_at=now,
      )
      await mem0_store.store_atom(atom)

    pruned = await mem0_store.prune_atoms(min_confidence=0.1)
    assert pruned == 1

    remaining = await mem0_store.get_atoms(user_id="user-1", min_confidence=0.0)
    assert len(remaining) == 3

  async def test_atom_source_episode_ids(self, mem0_store):
    now = time.time()
    atom = KnowledgeAtom(
      id="atom-src",
      user_id="user-1",
      subject="user",
      predicate="likes",
      object="coffee",
      content="user likes coffee",
      confidence=0.9,
      source_episode_ids=["ep-1", "ep-2", "ep-3"],
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_atom(atom)

    atoms = await mem0_store.get_atoms(user_id="user-1")
    assert atoms[0].source_episode_ids == ["ep-1", "ep-2", "ep-3"]

  async def test_search_atoms_by_embedding_returns_empty(self, mem0_store):
    results = await mem0_store.search_atoms_by_embedding(
      [0.1, 0.2, 0.3],
      user_id="user-1",
      top_k=5,
    )
    assert results == []


@pytest.mark.mem0
@pytest.mark.asyncio
class TestMem0StoreProcedures:
  """Tests for procedure storage and retrieval."""

  async def test_store_and_retrieve_procedure(self, mem0_store):
    now = time.time()
    proc = Procedure(
      id="proc-1",
      user_id="user-1",
      trigger="user asks for code",
      action="use Python with type hints",
      confidence=0.7,
      created_at=now,
      last_accessed_at=now,
    )
    result_id = await mem0_store.store_procedure(proc)
    assert result_id == "proc-1"

    procs = await mem0_store.get_procedures(user_id="user-1")
    assert len(procs) == 1
    assert procs[0].action == "use Python with type hints"
    assert procs[0].trigger == "user asks for code"

  async def test_find_similar_procedure(self, mem0_store):
    now = time.time()
    proc = Procedure(
      id="proc-sim",
      user_id="user-1",
      trigger="user asks for code review",
      action="check for type safety",
      confidence=0.6,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_procedure(proc)

    found = await mem0_store.find_similar_procedure("review my code", user_id="user-1")
    assert found is not None
    assert found.id == "proc-sim"

    not_found = await mem0_store.find_similar_procedure("tell me a joke", user_id="user-1")
    assert not_found is None

  async def test_update_procedure(self, mem0_store):
    now = time.time()
    proc = Procedure(
      id="proc-upd",
      user_id="user-1",
      trigger="user asks for help",
      action="be concise",
      confidence=0.4,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_procedure(proc)

    await mem0_store.update_procedure("proc-upd", confidence=0.9, observation_count=5)

    procs = await mem0_store.get_procedures(user_id="user-1")
    assert procs[0].confidence == 0.9
    assert procs[0].observation_count == 5

  async def test_procedure_confidence_filtering(self, mem0_store):
    now = time.time()
    for i, conf in enumerate([0.2, 0.5, 0.8]):
      proc = Procedure(
        id=f"proc-conf-{i}",
        user_id="user-1",
        trigger=f"trigger {i}",
        action=f"action {i}",
        confidence=conf,
        created_at=now,
        last_accessed_at=now,
      )
      await mem0_store.store_procedure(proc)

    # Default min_confidence=0.3 should exclude the 0.2 one
    procs = await mem0_store.get_procedures(user_id="user-1")
    assert len(procs) == 2

    # All procedures with min_confidence=0.1
    all_procs = await mem0_store.get_procedures(user_id="user-1", min_confidence=0.1)
    assert len(all_procs) == 3


@pytest.mark.mem0
@pytest.mark.asyncio
class TestMem0StoreTopics:
  """Tests for topic transition storage."""

  async def test_store_and_retrieve_transitions(self, mem0_store):
    # Store transitions multiple times
    for _ in range(5):
      await mem0_store.store_topic_transition("python", "testing", user_id="u1")
    for _ in range(3):
      await mem0_store.store_topic_transition("python", "debugging", user_id="u1")

    transitions = await mem0_store.get_topic_transitions("python", user_id="u1", min_count=3)
    assert len(transitions) == 2

    # Verify probabilities sum to 1.0
    total_prob = sum(t.probability for t in transitions)
    assert abs(total_prob - 1.0) < 1e-6

    # Highest count should be first
    assert transitions[0].to_topic == "testing"
    assert transitions[0].count == 5

  async def test_transitions_min_count_filter(self, mem0_store):
    for _ in range(2):
      await mem0_store.store_topic_transition("python", "rare_topic", user_id="u1")
    for _ in range(5):
      await mem0_store.store_topic_transition("python", "common_topic", user_id="u1")

    # min_count=3 should only return common_topic
    transitions = await mem0_store.get_topic_transitions("python", user_id="u1", min_count=3)
    assert len(transitions) == 1
    assert transitions[0].to_topic == "common_topic"

  async def test_transitions_null_user_id(self, mem0_store):
    for _ in range(4):
      await mem0_store.store_topic_transition("topic_a", "topic_b", user_id=None)

    transitions = await mem0_store.get_topic_transitions("topic_a", user_id=None, min_count=3)
    assert len(transitions) == 1
    assert transitions[0].count == 4


@pytest.mark.mem0
@pytest.mark.asyncio
class TestMem0StoreDeletion:
  """Tests for data deletion."""

  async def test_delete_user_data(self, mem0_store):
    now = time.time()
    episode = Episode(
      id="ep-del",
      user_id="user-del",
      session_id="s1",
      role="user",
      content="test",
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_episode(episode)
    atom = KnowledgeAtom(
      id="atom-del",
      user_id="user-del",
      subject="s",
      predicate="p",
      object="o",
      content="test",
      confidence=0.5,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_atom(atom)
    proc = Procedure(
      id="proc-del",
      user_id="user-del",
      trigger="test trigger",
      action="test action",
      confidence=0.5,
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_procedure(proc)

    await mem0_store.delete_user_data("user-del")
    await asyncio.sleep(2)  # Eventual consistency

    episodes = await mem0_store.get_episodes(user_id="user-del")
    assert len(episodes) == 0
    atoms = await mem0_store.get_atoms(user_id="user-del")
    assert len(atoms) == 0
    procs = await mem0_store.get_procedures(user_id="user-del", min_confidence=0.0)
    assert len(procs) == 0

  async def test_delete_session_data(self, mem0_store):
    now = time.time()
    episode = Episode(
      id="ep-sdel",
      user_id="user-1",
      session_id="sess-del",
      role="user",
      content="test",
      created_at=now,
      last_accessed_at=now,
    )
    await mem0_store.store_episode(episode)

    await mem0_store.delete_session_data("sess-del")
    await asyncio.sleep(2)  # Eventual consistency

    episodes = await mem0_store.get_episodes(user_id="user-1", session_id="sess-del")
    assert len(episodes) == 0

  async def test_delete_preserves_other_users(self, mem0_store):
    now = time.time()
    for uid in ["user-keep", "user-remove"]:
      ep = Episode(
        id=f"ep-{uid}",
        user_id=uid,
        session_id="s1",
        role="user",
        content=f"content for {uid}",
        created_at=now,
        last_accessed_at=now,
      )
      await mem0_store.store_episode(ep)

    await mem0_store.delete_user_data("user-remove")
    await asyncio.sleep(2)  # Eventual consistency

    kept = await mem0_store.get_episodes(user_id="user-keep")
    assert len(kept) == 1
    removed = await mem0_store.get_episodes(user_id="user-remove")
    assert len(removed) == 0
