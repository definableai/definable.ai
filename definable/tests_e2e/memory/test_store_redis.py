"""Tests for RedisMemoryStore."""

import os
import time

import pytest

from definable.memory.store.redis import RedisMemoryStore
from definable.memory.types import Episode, KnowledgeAtom, Procedure


@pytest.fixture
async def redis_store():
  """Temporary RedisMemoryStore for testing."""
  url = os.environ.get("MEMORY_REDIS_URL", "redis://localhost:6379")
  store = RedisMemoryStore(redis_url=url, prefix="test_memory")
  try:
    await store.initialize()
  except Exception:
    pytest.skip("Redis not available")
  yield store
  # Cleanup: delete all test keys
  client = store._client
  keys: list = []
  async for key in client.scan_iter(match="test_memory:*"):
    keys.append(key)
  if keys:
    await client.delete(*keys)
  await store.close()


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreEpisodes:
  """Tests for episode storage and retrieval."""

  async def test_store_and_retrieve_episode(self, redis_store):
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
    result_id = await redis_store.store_episode(episode)
    assert result_id == "ep-1"

    episodes = await redis_store.get_episodes(user_id="user-1")
    assert len(episodes) == 1
    assert episodes[0].id == "ep-1"
    assert episodes[0].content == "Hello, I live in San Francisco."
    assert episodes[0].topics == ["san_francisco", "location"]

  async def test_get_episodes_by_session(self, redis_store):
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
      await redis_store.store_episode(episode)

    sess_a = await redis_store.get_episodes(session_id="sess-A")
    assert len(sess_a) == 2

    sess_b = await redis_store.get_episodes(session_id="sess-B")
    assert len(sess_b) == 1

  async def test_update_episode(self, redis_store):
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
    await redis_store.store_episode(episode)

    await redis_store.update_episode("ep-upd", content="Updated content", compression_stage=1)

    episodes = await redis_store.get_episodes(user_id="user-1")
    assert episodes[0].content == "Updated content"
    assert episodes[0].compression_stage == 1

  async def test_get_episodes_for_distillation(self, redis_store):
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
      await redis_store.store_episode(ep)

    # Get stage-0 episodes older than 1 hour ago
    threshold = time.time() - 3600
    result = await redis_store.get_episodes_for_distillation(stage=0, older_than=threshold)
    assert len(result) == 1
    assert result[0].id == "ep-dist-0"

  async def test_episode_with_embedding(self, redis_store):
    now = time.time()
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    episode = Episode(
      id="ep-emb",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="Test embedding",
      embedding=embedding,
      created_at=now,
      last_accessed_at=now,
    )
    await redis_store.store_episode(episode)

    episodes = await redis_store.get_episodes(user_id="user-1")
    assert episodes[0].embedding == embedding

  async def test_search_episodes_by_embedding(self, redis_store):
    now = time.time()
    episodes = [
      Episode(
        id="ep-vec-1",
        user_id="u1",
        session_id="s1",
        role="user",
        content="Python programming",
        embedding=[0.9, 0.1, 0.0],
        created_at=now,
        last_accessed_at=now,
      ),
      Episode(
        id="ep-vec-2",
        user_id="u1",
        session_id="s1",
        role="user",
        content="Cooking recipes",
        embedding=[0.1, 0.9, 0.0],
        created_at=now,
        last_accessed_at=now,
      ),
      Episode(
        id="ep-vec-3",
        user_id="u1",
        session_id="s1",
        role="user",
        content="Code review",
        embedding=[0.8, 0.2, 0.0],
        created_at=now,
        last_accessed_at=now,
      ),
    ]
    for ep in episodes:
      await redis_store.store_episode(ep)

    # Search with programming-like embedding
    results = await redis_store.search_episodes_by_embedding(
      [0.85, 0.15, 0.0],
      user_id="u1",
      top_k=2,
    )
    assert len(results) == 2
    # Most similar should be first
    assert results[0].id in ("ep-vec-1", "ep-vec-3")

  async def test_auto_generate_id(self, redis_store):
    now = time.time()
    episode = Episode(
      id="",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="No ID provided",
      created_at=now,
      last_accessed_at=now,
    )
    result_id = await redis_store.store_episode(episode)
    assert result_id != ""
    assert len(result_id) == 36  # UUID4 format

  async def test_auto_set_timestamps(self, redis_store):
    episode = Episode(
      id="ep-ts",
      user_id="user-1",
      session_id="sess-1",
      role="user",
      content="Timestamps should be auto-set",
    )
    before = time.time()
    await redis_store.store_episode(episode)
    after = time.time()

    episodes = await redis_store.get_episodes(user_id="user-1")
    assert before <= episodes[0].created_at <= after
    assert before <= episodes[0].last_accessed_at <= after

  async def test_get_episodes_with_stage_filter(self, redis_store):
    now = time.time()
    for i, stage in enumerate([0, 1, 2, 3]):
      ep = Episode(
        id=f"ep-stage-{i}",
        user_id="user-1",
        session_id="sess-1",
        role="user",
        content=f"Stage {stage}",
        compression_stage=stage,
        created_at=now + i,
        last_accessed_at=now + i,
      )
      await redis_store.store_episode(ep)

    result = await redis_store.get_episodes(user_id="user-1", min_stage=1, max_stage=2)
    assert len(result) == 2
    stages = {ep.compression_stage for ep in result}
    assert stages == {1, 2}

  async def test_get_episodes_with_limit(self, redis_store):
    now = time.time()
    for i in range(10):
      ep = Episode(
        id=f"ep-lim-{i}",
        user_id="user-1",
        session_id="sess-1",
        role="user",
        content=f"Message {i}",
        created_at=now + i,
        last_accessed_at=now + i,
      )
      await redis_store.store_episode(ep)

    result = await redis_store.get_episodes(user_id="user-1", limit=3)
    assert len(result) == 3
    # Should be most recent first
    assert result[0].id == "ep-lim-9"


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreAtoms:
  """Tests for knowledge atom storage and retrieval."""

  async def test_store_and_retrieve_atom(self, redis_store):
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
    result_id = await redis_store.store_atom(atom)
    assert result_id == "atom-1"

    atoms = await redis_store.get_atoms(user_id="user-1")
    assert len(atoms) == 1
    assert atoms[0].content == "user lives in San Francisco"
    assert atoms[0].confidence == 0.9

  async def test_find_similar_atom(self, redis_store):
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
    await redis_store.store_atom(atom)

    found = await redis_store.find_similar_atom("user", "prefers", user_id="user-1")
    assert found is not None
    assert found.id == "atom-sim"

    not_found = await redis_store.find_similar_atom("user", "dislikes", user_id="user-1")
    assert not_found is None

  async def test_find_similar_atom_case_insensitive(self, redis_store):
    now = time.time()
    atom = KnowledgeAtom(
      id="atom-ci",
      user_id="user-1",
      subject="User",
      predicate="Lives_In",
      object="NYC",
      content="User lives in NYC",
      confidence=0.9,
      created_at=now,
      last_accessed_at=now,
    )
    await redis_store.store_atom(atom)

    found = await redis_store.find_similar_atom("user", "lives_in", user_id="user-1")
    assert found is not None
    assert found.id == "atom-ci"

  async def test_update_atom(self, redis_store):
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
    await redis_store.store_atom(atom)

    await redis_store.update_atom("atom-upd", confidence=0.8, reinforcement_count=2)

    atoms = await redis_store.get_atoms(user_id="user-1")
    assert atoms[0].confidence == 0.8
    assert atoms[0].reinforcement_count == 2

  async def test_prune_atoms(self, redis_store):
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
      await redis_store.store_atom(atom)

    pruned = await redis_store.prune_atoms(min_confidence=0.1)
    assert pruned == 1

    remaining = await redis_store.get_atoms(user_id="user-1", min_confidence=0.0)
    assert len(remaining) == 3

  async def test_search_atoms_by_embedding(self, redis_store):
    now = time.time()
    atoms = [
      KnowledgeAtom(
        id="a-vec-1",
        user_id="u1",
        subject="s",
        predicate="p",
        object="o",
        content="Python related",
        embedding=[0.9, 0.1, 0.0],
        confidence=0.8,
        created_at=now,
        last_accessed_at=now,
      ),
      KnowledgeAtom(
        id="a-vec-2",
        user_id="u1",
        subject="s",
        predicate="p",
        object="o",
        content="Cooking related",
        embedding=[0.1, 0.9, 0.0],
        confidence=0.8,
        created_at=now,
        last_accessed_at=now,
      ),
    ]
    for a in atoms:
      await redis_store.store_atom(a)

    results = await redis_store.search_atoms_by_embedding(
      [0.85, 0.15, 0.0],
      user_id="u1",
      top_k=1,
    )
    assert len(results) == 1
    assert results[0].id == "a-vec-1"


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreProcedures:
  """Tests for procedure storage and retrieval."""

  async def test_store_and_retrieve_procedure(self, redis_store):
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
    result_id = await redis_store.store_procedure(proc)
    assert result_id == "proc-1"

    procs = await redis_store.get_procedures(user_id="user-1")
    assert len(procs) == 1
    assert procs[0].action == "use Python with type hints"

  async def test_find_similar_procedure(self, redis_store):
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
    await redis_store.store_procedure(proc)

    found = await redis_store.find_similar_procedure("review my code", user_id="user-1")
    assert found is not None
    assert found.id == "proc-sim"

  async def test_update_procedure(self, redis_store):
    now = time.time()
    proc = Procedure(
      id="proc-upd",
      user_id="user-1",
      trigger="user asks for help",
      action="be concise",
      confidence=0.5,
      created_at=now,
      last_accessed_at=now,
    )
    await redis_store.store_procedure(proc)

    await redis_store.update_procedure("proc-upd", confidence=0.9, observation_count=5)

    procs = await redis_store.get_procedures(user_id="user-1")
    assert procs[0].confidence == 0.9
    assert procs[0].observation_count == 5


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreTopics:
  """Tests for topic transition storage."""

  async def test_store_and_retrieve_transitions(self, redis_store):
    # Store transitions multiple times
    for _ in range(5):
      await redis_store.store_topic_transition("python", "testing", user_id="u1")
    for _ in range(3):
      await redis_store.store_topic_transition("python", "debugging", user_id="u1")

    transitions = await redis_store.get_topic_transitions("python", user_id="u1", min_count=3)
    assert len(transitions) == 2

    # Verify probabilities sum to 1.0
    total_prob = sum(t.probability for t in transitions)
    assert abs(total_prob - 1.0) < 1e-6

    # Highest count should be first
    assert transitions[0].to_topic == "testing"
    assert transitions[0].count == 5

  async def test_transitions_below_min_count_filtered(self, redis_store):
    for _ in range(2):
      await redis_store.store_topic_transition("python", "rare_topic", user_id="u1")

    transitions = await redis_store.get_topic_transitions("python", user_id="u1", min_count=3)
    assert len(transitions) == 0


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreDeletion:
  """Tests for data deletion."""

  async def test_delete_user_data(self, redis_store):
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
    await redis_store.store_episode(episode)
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
    await redis_store.store_atom(atom)

    await redis_store.delete_user_data("user-del")

    episodes = await redis_store.get_episodes(user_id="user-del")
    assert len(episodes) == 0
    atoms = await redis_store.get_atoms(user_id="user-del")
    assert len(atoms) == 0

  async def test_delete_session_data(self, redis_store):
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
    await redis_store.store_episode(episode)

    await redis_store.delete_session_data("sess-del")

    episodes = await redis_store.get_episodes(session_id="sess-del")
    assert len(episodes) == 0

  async def test_delete_user_preserves_other_users(self, redis_store):
    now = time.time()
    ep1 = Episode(
      id="ep-keep",
      user_id="user-keep",
      session_id="s1",
      role="user",
      content="keep me",
      created_at=now,
      last_accessed_at=now,
    )
    ep2 = Episode(
      id="ep-remove",
      user_id="user-remove",
      session_id="s2",
      role="user",
      content="remove me",
      created_at=now,
      last_accessed_at=now,
    )
    await redis_store.store_episode(ep1)
    await redis_store.store_episode(ep2)

    await redis_store.delete_user_data("user-remove")

    kept = await redis_store.get_episodes(user_id="user-keep")
    assert len(kept) == 1
    assert kept[0].id == "ep-keep"

    removed = await redis_store.get_episodes(user_id="user-remove")
    assert len(removed) == 0


@pytest.mark.redis
@pytest.mark.asyncio
class TestRedisMemoryStoreContextManager:
  """Tests for async context manager protocol."""

  async def test_context_manager(self):
    url = os.environ.get("MEMORY_REDIS_URL", "redis://localhost:6379")
    try:
      async with RedisMemoryStore(redis_url=url, prefix="test_memory_cm") as store:
        now = time.time()
        ep = Episode(
          id="ep-cm",
          user_id="u1",
          session_id="s1",
          role="user",
          content="context manager test",
          created_at=now,
          last_accessed_at=now,
        )
        await store.store_episode(ep)
        episodes = await store.get_episodes(user_id="u1")
        assert len(episodes) == 1
    except Exception:
      pytest.skip("Redis not available")
