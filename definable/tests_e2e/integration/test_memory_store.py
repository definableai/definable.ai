"""
Integration tests for memory stores.

Rules:
  - InMemoryStore: pure Python, no API needed — tests always run
  - External stores (Qdrant, Postgres, etc.) require running services
  - Tests verify the full MemoryStore Protocol contract with real data

Real Episode/KnowledgeAtom types (from definable.memory.types):
  Episode: id, user_id, session_id, role (user|assistant), content, created_at, ...
  KnowledgeAtom: id, user_id, subject, predicate, object, content, confidence, ...

Covers:
  - store_episode() / get_episodes() round-trip
  - store_atom() / get_atoms() round-trip
  - initialize() and close() lifecycle
  - Session isolation (episodes from different sessions don't mix)
  - Episode ordering (most recent first by created_at)
"""

import time
import uuid

import pytest

from definable.memory.types import Episode, KnowledgeAtom  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_episode(session_id: str, user_id: str, content: str, role: str = "user") -> Episode:
  return Episode(
    id=str(uuid.uuid4()),
    session_id=session_id,
    user_id=user_id,
    role=role,
    content=content,
  )


def make_atom(user_id: str, subject: str, predicate: str, obj: str) -> KnowledgeAtom:
  return KnowledgeAtom(
    id=str(uuid.uuid4()),
    user_id=user_id,
    subject=subject,
    predicate=predicate,
    object=obj,
    content=f"{subject} {predicate} {obj}",
    confidence=0.9,
  )


# ---------------------------------------------------------------------------
# InMemoryStore — no external deps, always runs
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInMemoryStore:
  """InMemoryStore tests run without any API keys or external services."""

  @pytest.fixture
  def store(self):
    from definable.memory.store.in_memory import InMemoryStore

    return InMemoryStore()

  @pytest.fixture
  def session_id(self):
    return f"test-session-{uuid.uuid4().hex[:8]}"

  @pytest.fixture
  def user_id(self):
    return f"test-user-{uuid.uuid4().hex[:8]}"

  @pytest.mark.asyncio
  async def test_lifecycle_initialize_and_close(self, store):
    """Store can be initialized and closed without error."""
    await store.initialize()
    await store.close()

  @pytest.mark.asyncio
  async def test_close_is_idempotent(self, store):
    """Closing a store twice should not raise."""
    await store.initialize()
    await store.close()
    await store.close()

  @pytest.mark.asyncio
  async def test_store_and_retrieve_episode(self, store, session_id, user_id):
    """store_episode() persists and get_episodes() retrieves."""
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "User said hello to the agent.")
      episode_id = await store.store_episode(ep)
      assert episode_id

      episodes = await store.get_episodes(session_id=session_id, user_id=user_id)
      assert len(episodes) >= 1
      stored_content = [e.content for e in episodes]
      assert "User said hello to the agent." in stored_content
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_store_episode_returns_string_id(self, store, session_id, user_id):
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "Content")
      ep_id = await store.store_episode(ep)
      assert isinstance(ep_id, str)
      assert len(ep_id) > 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_get_episodes_empty_session_returns_empty(self, store):
    """Querying a non-existent session returns empty list."""
    await store.initialize()
    try:
      episodes = await store.get_episodes(session_id="nonexistent-session-xyz")
      assert isinstance(episodes, list)
      assert len(episodes) == 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_session_isolation(self, store):
    """Episodes from session A must not appear in session B queries."""
    await store.initialize()
    try:
      session_a = f"session-a-{uuid.uuid4().hex[:8]}"
      session_b = f"session-b-{uuid.uuid4().hex[:8]}"

      ep_a = make_episode(session_a, "user1", "Secret info for session A")
      await store.store_episode(ep_a)

      episodes_b = await store.get_episodes(session_id=session_b, user_id="user1")
      assert all(e.content != "Secret info for session A" for e in episodes_b)
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_multiple_episodes_retrieved(self, store, session_id, user_id):
    """Multiple episodes in the same session are all retrievable."""
    await store.initialize()
    try:
      for i in range(3):
        ep = make_episode(session_id, user_id, f"Episode number {i}")
        await store.store_episode(ep)

      episodes = await store.get_episodes(session_id=session_id, user_id=user_id)
      assert len(episodes) == 3
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_episodes_ordered_by_recency(self, store, session_id, user_id):
    """get_episodes() returns most recent episodes first (created_at descending)."""
    await store.initialize()
    try:
      for i in range(3):
        ep = make_episode(session_id, user_id, f"Episode {i}")
        ep.created_at = time.time() + i  # Ensure increasing timestamps
        await store.store_episode(ep)

      episodes = await store.get_episodes(session_id=session_id, user_id=user_id)
      timestamps = [e.created_at for e in episodes]
      # Should be sorted descending (most recent first)
      assert timestamps == sorted(timestamps, reverse=True)
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_store_and_retrieve_atom(self, store, user_id):
    """Knowledge atoms (semantic facts) can be stored and retrieved."""
    await store.initialize()
    try:
      atom = make_atom(user_id, "user", "name is", "Alice")
      atom_id = await store.store_atom(atom)
      assert atom_id

      atoms = await store.get_atoms(user_id=user_id)
      assert len(atoms) >= 1
      assert any("Alice" in a.content for a in atoms)
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_get_atoms_empty_returns_empty(self, store):
    await store.initialize()
    try:
      atoms = await store.get_atoms(user_id=f"nonexistent-{uuid.uuid4().hex}")
      assert isinstance(atoms, list)
      assert len(atoms) == 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_atom_confidence_filtering(self, store, user_id):
    """get_atoms() with min_confidence filters out low-confidence atoms."""
    await store.initialize()
    try:
      high_conf = make_atom(user_id, "user", "works at", "Acme")
      high_conf.confidence = 0.9
      low_conf = make_atom(user_id, "user", "maybe lives in", "Paris")
      low_conf.confidence = 0.05  # Below default min_confidence of 0.1

      await store.store_atom(high_conf)
      await store.store_atom(low_conf)

      # Default min_confidence=0.1 should filter out the low-conf atom
      atoms = await store.get_atoms(user_id=user_id, min_confidence=0.1)
      assert all(a.confidence >= 0.1 for a in atoms)
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_delete_session_data(self, store, session_id, user_id):
    """delete_session_data() removes all episodes for that session."""
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "To be deleted")
      await store.store_episode(ep)
      assert len(await store.get_episodes(session_id=session_id)) == 1

      await store.delete_session_data(session_id)
      assert len(await store.get_episodes(session_id=session_id)) == 0
    finally:
      await store.close()

  @pytest.mark.asyncio
  async def test_auto_initializes_on_first_operation(self, store, session_id, user_id):
    """Store should auto-initialize when used without explicit initialize()."""
    ep = make_episode(session_id, user_id, "Auto-init test")
    ep_id = await store.store_episode(ep)
    assert ep_id
    await store.close()
