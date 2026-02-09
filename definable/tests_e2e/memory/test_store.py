"""Tests for SQLiteMemoryStore."""

import time

import pytest

from definable.memory.types import Episode, KnowledgeAtom, Procedure


@pytest.mark.asyncio
class TestSQLiteMemoryStoreEpisodes:
  """Tests for episode storage and retrieval."""

  async def test_store_and_retrieve_episode(self, sqlite_store):
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
    result_id = await sqlite_store.store_episode(episode)
    assert result_id == "ep-1"

    episodes = await sqlite_store.get_episodes(user_id="user-1")
    assert len(episodes) == 1
    assert episodes[0].id == "ep-1"
    assert episodes[0].content == "Hello, I live in San Francisco."
    assert episodes[0].topics == ["san_francisco", "location"]

  async def test_get_episodes_by_session(self, sqlite_store):
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
      await sqlite_store.store_episode(episode)

    sess_a = await sqlite_store.get_episodes(session_id="sess-A")
    assert len(sess_a) == 2

    sess_b = await sqlite_store.get_episodes(session_id="sess-B")
    assert len(sess_b) == 1

  async def test_update_episode(self, sqlite_store):
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
    await sqlite_store.store_episode(episode)

    await sqlite_store.update_episode("ep-upd", content="Updated content", compression_stage=1)

    episodes = await sqlite_store.get_episodes(user_id="user-1")
    assert episodes[0].content == "Updated content"
    assert episodes[0].compression_stage == 1

  async def test_get_episodes_for_distillation(self, sqlite_store):
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
      await sqlite_store.store_episode(ep)

    # Get stage-0 episodes older than 1 hour ago
    threshold = time.time() - 3600
    result = await sqlite_store.get_episodes_for_distillation(stage=0, older_than=threshold)
    assert len(result) == 1
    assert result[0].id == "ep-dist-0"

  async def test_episode_with_embedding(self, sqlite_store):
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
    await sqlite_store.store_episode(episode)

    episodes = await sqlite_store.get_episodes(user_id="user-1")
    assert episodes[0].embedding == embedding

  async def test_search_episodes_by_embedding(self, sqlite_store):
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
      await sqlite_store.store_episode(ep)

    # Search with programming-like embedding
    results = await sqlite_store.search_episodes_by_embedding(
      [0.85, 0.15, 0.0],
      user_id="u1",
      top_k=2,
    )
    assert len(results) == 2
    # Most similar should be first
    assert results[0].id in ("ep-vec-1", "ep-vec-3")


@pytest.mark.asyncio
class TestSQLiteMemoryStoreAtoms:
  """Tests for knowledge atom storage and retrieval."""

  async def test_store_and_retrieve_atom(self, sqlite_store):
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
    result_id = await sqlite_store.store_atom(atom)
    assert result_id == "atom-1"

    atoms = await sqlite_store.get_atoms(user_id="user-1")
    assert len(atoms) == 1
    assert atoms[0].content == "user lives in San Francisco"
    assert atoms[0].confidence == 0.9

  async def test_find_similar_atom(self, sqlite_store):
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
    await sqlite_store.store_atom(atom)

    found = await sqlite_store.find_similar_atom("user", "prefers", user_id="user-1")
    assert found is not None
    assert found.id == "atom-sim"

    not_found = await sqlite_store.find_similar_atom("user", "dislikes", user_id="user-1")
    assert not_found is None

  async def test_update_atom(self, sqlite_store):
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
    await sqlite_store.store_atom(atom)

    await sqlite_store.update_atom("atom-upd", confidence=0.8, reinforcement_count=2)

    atoms = await sqlite_store.get_atoms(user_id="user-1")
    assert atoms[0].confidence == 0.8
    assert atoms[0].reinforcement_count == 2

  async def test_prune_atoms(self, sqlite_store):
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
      await sqlite_store.store_atom(atom)

    pruned = await sqlite_store.prune_atoms(min_confidence=0.1)
    assert pruned == 1

    remaining = await sqlite_store.get_atoms(user_id="user-1", min_confidence=0.0)
    assert len(remaining) == 3

  async def test_search_atoms_by_embedding(self, sqlite_store):
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
      await sqlite_store.store_atom(a)

    results = await sqlite_store.search_atoms_by_embedding(
      [0.85, 0.15, 0.0],
      user_id="u1",
      top_k=1,
    )
    assert len(results) == 1
    assert results[0].id == "a-vec-1"


@pytest.mark.asyncio
class TestSQLiteMemoryStoreProcedures:
  """Tests for procedure storage and retrieval."""

  async def test_store_and_retrieve_procedure(self, sqlite_store):
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
    result_id = await sqlite_store.store_procedure(proc)
    assert result_id == "proc-1"

    procs = await sqlite_store.get_procedures(user_id="user-1")
    assert len(procs) == 1
    assert procs[0].action == "use Python with type hints"

  async def test_find_similar_procedure(self, sqlite_store):
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
    await sqlite_store.store_procedure(proc)

    found = await sqlite_store.find_similar_procedure("review my code", user_id="user-1")
    assert found is not None
    assert found.id == "proc-sim"


@pytest.mark.asyncio
class TestSQLiteMemoryStoreTopics:
  """Tests for topic transition storage."""

  async def test_store_and_retrieve_transitions(self, sqlite_store):
    # Store transitions multiple times
    for _ in range(5):
      await sqlite_store.store_topic_transition("python", "testing", user_id="u1")
    for _ in range(3):
      await sqlite_store.store_topic_transition("python", "debugging", user_id="u1")

    transitions = await sqlite_store.get_topic_transitions("python", user_id="u1", min_count=3)
    assert len(transitions) == 2

    # Verify probabilities sum to 1.0
    total_prob = sum(t.probability for t in transitions)
    assert abs(total_prob - 1.0) < 1e-6

    # Highest count should be first
    assert transitions[0].to_topic == "testing"
    assert transitions[0].count == 5


@pytest.mark.asyncio
class TestSQLiteMemoryStoreDeletion:
  """Tests for data deletion."""

  async def test_delete_user_data(self, sqlite_store):
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
    await sqlite_store.store_episode(episode)
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
    await sqlite_store.store_atom(atom)

    await sqlite_store.delete_user_data("user-del")

    episodes = await sqlite_store.get_episodes(user_id="user-del")
    assert len(episodes) == 0
    atoms = await sqlite_store.get_atoms(user_id="user-del")
    assert len(atoms) == 0

  async def test_delete_session_data(self, sqlite_store):
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
    await sqlite_store.store_episode(episode)

    await sqlite_store.delete_session_data("sess-del")

    episodes = await sqlite_store.get_episodes(session_id="sess-del")
    assert len(episodes) == 0
