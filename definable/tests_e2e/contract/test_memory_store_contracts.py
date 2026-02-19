"""
Contract tests: Every MemoryStore implementation must satisfy these.

The MemoryStore Protocol defines the contract:
  - initialize() / close() lifecycle
  - store_episode(episode) → episode_id (str)
  - get_episodes(session_id, user_id, limit) → List[Episode]
  - store_atom(atom) → atom_id (str)
  - get_atoms(user_id, min_confidence, limit) → List[KnowledgeAtom]

Real Episode/KnowledgeAtom types (from definable.memory.types):
  Episode: id, user_id, session_id, role, content, created_at (float unix ts)
  KnowledgeAtom: id, user_id, subject, predicate, object, content, confidence

InMemoryStore always runs (no external deps).
External stores gated by service availability marks.

To add a new MemoryStore: inherit this class and provide a `store` fixture.
CI should verify that every MemoryStore has a corresponding contract test class.
"""

import uuid

import pytest

from definable.memory.types import Episode, KnowledgeAtom  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_episode(session_id: str, user_id: str, content: str) -> Episode:
  return Episode(
    id=str(uuid.uuid4()),
    session_id=session_id,
    user_id=user_id,
    role="user",
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
# Contract definition
# ---------------------------------------------------------------------------


class MemoryStoreContractTests:
  """
  Abstract contract test suite for all MemoryStore implementations.

  Every concrete MemoryStore must pass ALL tests in this class.
  """

  @pytest.fixture
  def store(self):
    raise NotImplementedError("Subclass must provide a store fixture")

  @pytest.fixture
  def session_id(self) -> str:
    return f"contract-session-{uuid.uuid4().hex[:8]}"

  @pytest.fixture
  def user_id(self) -> str:
    return f"contract-user-{uuid.uuid4().hex[:8]}"

  # --- Contract: lifecycle ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_initialize_does_not_raise(self, store):
    await store.initialize()
    await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_close_is_safe_after_initialize(self, store):
    await store.initialize()
    await store.close()  # Must not raise

  # --- Contract: episodes ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_store_episode_returns_string_id(self, store, session_id, user_id):
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "Test content")
      ep_id = await store.store_episode(ep)
      assert isinstance(ep_id, str)
      assert len(ep_id) > 0
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_episodes_returns_list(self, store, session_id, user_id):
    await store.initialize()
    try:
      result = await store.get_episodes(session_id=session_id, user_id=user_id)
      assert isinstance(result, list)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_stored_episode_is_retrievable(self, store, session_id, user_id):
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "Important conversation content")
      await store.store_episode(ep)
      episodes = await store.get_episodes(session_id=session_id, user_id=user_id)
      assert len(episodes) >= 1
      contents = [e.content for e in episodes]
      assert "Important conversation content" in contents
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_episodes_empty_session_returns_empty(self, store):
    await store.initialize()
    try:
      result = await store.get_episodes(session_id="nonexistent-xyz")
      assert isinstance(result, list)
      assert len(result) == 0
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_session_isolation(self, store):
    await store.initialize()
    try:
      sid_a = f"iso-session-a-{uuid.uuid4().hex[:6]}"
      sid_b = f"iso-session-b-{uuid.uuid4().hex[:6]}"
      uid = f"user-{uuid.uuid4().hex[:6]}"

      ep_a = make_episode(sid_a, uid, "Private to session A")
      await store.store_episode(ep_a)

      episodes_b = await store.get_episodes(session_id=sid_b, user_id=uid)
      assert not any(e.content == "Private to session A" for e in episodes_b)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_multiple_episodes_stored_and_retrieved(self, store, session_id, user_id):
    await store.initialize()
    try:
      for i in range(3):
        ep = make_episode(session_id, user_id, f"Episode {i} content")
        await store.store_episode(ep)

      episodes = await store.get_episodes(session_id=session_id, user_id=user_id)
      assert len(episodes) == 3
    finally:
      await store.close()

  # --- Contract: knowledge atoms ---

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_store_atom_returns_string_id(self, store, user_id):
    await store.initialize()
    try:
      atom = make_atom(user_id, "user", "name is", "Bob")
      atom_id = await store.store_atom(atom)
      assert isinstance(atom_id, str)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_get_atoms_returns_list(self, store, user_id):
    await store.initialize()
    try:
      result = await store.get_atoms(user_id=user_id)
      assert isinstance(result, list)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_stored_atom_is_retrievable(self, store, user_id):
    await store.initialize()
    try:
      atom = make_atom(user_id, "user", "prefers", "dark mode")
      await store.store_atom(atom)
      atoms = await store.get_atoms(user_id=user_id)
      assert len(atoms) >= 1
      assert any("dark mode" in a.content for a in atoms)
    finally:
      await store.close()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_episode_has_required_fields(self, store, session_id, user_id):
    """Retrieved Episode must have all expected fields."""
    await store.initialize()
    try:
      ep = make_episode(session_id, user_id, "Field check")
      await store.store_episode(ep)
      episodes = await store.get_episodes(session_id=session_id)
      assert len(episodes) >= 1
      retrieved = episodes[0]
      assert hasattr(retrieved, "id")
      assert hasattr(retrieved, "content")
      assert hasattr(retrieved, "session_id")
      assert hasattr(retrieved, "user_id")
      assert hasattr(retrieved, "role")
    finally:
      await store.close()


# ---------------------------------------------------------------------------
# InMemoryStore — always runs, no external deps
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestInMemoryStoreContract(MemoryStoreContractTests):
  """InMemoryStore satisfies the MemoryStore contract."""

  @pytest.fixture
  def store(self):
    from definable.memory.store.in_memory import InMemoryStore

    return InMemoryStore()

  @pytest.mark.contract
  @pytest.mark.asyncio
  async def test_close_without_initialize_does_not_raise(self, store):
    """InMemoryStore should be resilient to close() without initialize()."""
    await store.close()  # Must not raise
