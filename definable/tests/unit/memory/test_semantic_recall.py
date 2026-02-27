"""Tests for semantic search & smart recall — Phase 2 of semantic memory.

Tests cover:
  - Memory.search() — cosine similarity ranking of atom entries
  - Memory.has_semantic_search property
  - Memory._embed_atoms() — auto-embedding during optimization
  - _cosine_similarity() — vector math
  - Dual-layer recall formatting (STM + LTM)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.memory.manager import Memory, _cosine_similarity
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import MemoryEntry
from definable.model.message import Message


# --- Helpers ---


def _make_embedder(dim: int = 3) -> MagicMock:
  """Create a mock embedder that returns deterministic vectors."""
  embedder = MagicMock()
  embedder.dimensions = dim

  call_count = 0

  async def _embed(text: str):
    nonlocal call_count
    call_count += 1
    # Simple hash-based vector for deterministic results.
    h = hash(text) % 1000
    return [float(h % 7) / 7.0, float(h % 11) / 11.0, float(h % 13) / 13.0]

  embedder.async_get_embedding = AsyncMock(side_effect=_embed)
  return embedder


def _atom(session_id: str = "s1", content: str = "Fact.", vector: list[float] | None = None, **kw) -> MemoryEntry:
  """Create an atom entry."""
  return MemoryEntry(
    session_id=session_id,
    role="atom",
    content=content,
    entry_type="atom",
    lossless_content=content,
    vector=vector,
    **kw,
  )


def _msg(session_id: str = "s1", role: str = "user", content: str = "Hello", **kw) -> MemoryEntry:
  """Create a regular message entry."""
  return MemoryEntry(session_id=session_id, role=role, content=content, **kw)


# --- Cosine Similarity Tests ---


class TestCosineSimilarity:
  def test_identical_vectors(self):
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

  def test_orthogonal_vectors(self):
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

  def test_opposite_vectors(self):
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

  def test_zero_vector(self):
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

  def test_real_vectors(self):
    a = [0.1, 0.2, 0.3]
    b = [0.1, 0.2, 0.3]
    assert _cosine_similarity(a, b) == pytest.approx(1.0)


# --- Memory.search() Tests ---


class TestMemorySearch:
  @pytest.mark.asyncio
  async def test_search_returns_atoms_ranked_by_similarity(self):
    store = InMemoryStore()
    embedder = MagicMock()

    # Query vector
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mem = Memory(store=store, embedder=embedder)
    await mem._ensure_initialized()

    # Add atoms with different vectors (different similarities to [1,0,0]).
    close_atom = _atom(content="Close match", vector=[0.9, 0.1, 0.0])
    far_atom = _atom(content="Far match", vector=[0.0, 0.0, 1.0])
    mid_atom = _atom(content="Mid match", vector=[0.5, 0.5, 0.0])

    for a in [far_atom, mid_atom, close_atom]:
      await store.add(a)

    results = await mem.search("query", session_id="s1")

    assert len(results) == 3
    # Close match should be first (highest cosine similarity to [1,0,0]).
    assert results[0].content == "Close match"
    assert results[2].content == "Far match"

  @pytest.mark.asyncio
  async def test_search_respects_top_k(self):
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mem = Memory(store=store, embedder=embedder, search_top_k=2)
    await mem._ensure_initialized()

    for i in range(5):
      await store.add(_atom(content=f"Fact {i}", vector=[float(i) / 5, 0.1, 0.0]))

    results = await mem.search("query", session_id="s1")
    assert len(results) == 2

  @pytest.mark.asyncio
  async def test_search_filters_superseded(self):
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mem = Memory(store=store, embedder=embedder)
    await mem._ensure_initialized()

    active = _atom(content="Active fact", vector=[0.9, 0.1, 0.0])
    superseded = _atom(content="Old fact", vector=[0.8, 0.2, 0.0], superseded_by="some-id")

    await store.add(active)
    await store.add(superseded)

    results = await mem.search("query", session_id="s1")
    assert len(results) == 1
    assert results[0].content == "Active fact"

  @pytest.mark.asyncio
  async def test_search_without_embedder_returns_chronological(self):
    store = InMemoryStore()
    mem = Memory(store=store)  # No embedder
    await mem._ensure_initialized()

    await store.add(_atom(content="Fact 1", created_at=1.0, updated_at=1.0))
    await store.add(_atom(content="Fact 2", created_at=2.0, updated_at=2.0))
    await store.add(_atom(content="Fact 3", created_at=3.0, updated_at=3.0))

    results = await mem.search("query", session_id="s1", top_k=2)
    assert len(results) == 2
    # Should return last 2 chronologically.
    assert results[0].content == "Fact 2"
    assert results[1].content == "Fact 3"

  @pytest.mark.asyncio
  async def test_search_with_no_atoms(self):
    store = InMemoryStore()
    embedder = MagicMock()
    mem = Memory(store=store, embedder=embedder)
    await mem._ensure_initialized()

    # Only regular messages, no atoms.
    await store.add(_msg(content="Hello"))
    await store.add(_msg(role="assistant", content="Hi"))

    results = await mem.search("query", session_id="s1")
    assert results == []

  @pytest.mark.asyncio
  async def test_search_falls_back_on_embedding_failure(self):
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(side_effect=RuntimeError("API down"))

    mem = Memory(store=store, embedder=embedder)
    await mem._ensure_initialized()

    await store.add(_atom(content="Fact 1", vector=[1.0, 0.0, 0.0]))

    results = await mem.search("query", session_id="s1")
    # Falls back to chronological.
    assert len(results) == 1
    assert results[0].content == "Fact 1"

  @pytest.mark.asyncio
  async def test_search_atoms_without_vectors_fallback(self):
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mem = Memory(store=store, embedder=embedder)
    await mem._ensure_initialized()

    # Atoms without vectors (not yet embedded).
    await store.add(_atom(content="No vector atom 1"))
    await store.add(_atom(content="No vector atom 2"))

    results = await mem.search("query", session_id="s1")
    # Falls back to chronological since no atoms have vectors.
    assert len(results) == 2


# --- has_semantic_search Property ---


class TestHasSemanticSearch:
  def test_true_with_embedder(self):
    mem = Memory(embedder=MagicMock())
    assert mem.has_semantic_search is True

  def test_false_without_embedder(self):
    mem = Memory()
    assert mem.has_semantic_search is False


# --- _embed_atoms Tests ---


class TestEmbedAtoms:
  @pytest.mark.asyncio
  async def test_embeds_atoms_without_vectors(self):
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    mem = Memory(embedder=embedder)

    atom = _atom(content="Test fact")
    msg = _msg(content="Regular message")

    await mem._embed_atoms([atom, msg])

    # Only the atom should have been embedded.
    assert atom.vector == [0.1, 0.2, 0.3]
    assert msg.vector is None
    embedder.async_get_embedding.assert_called_once_with("Test fact")

  @pytest.mark.asyncio
  async def test_skips_atoms_with_existing_vectors(self):
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    mem = Memory(embedder=embedder)

    atom = _atom(content="Already embedded", vector=[0.9, 0.8, 0.7])
    await mem._embed_atoms([atom])

    # Should not re-embed.
    embedder.async_get_embedding.assert_not_called()
    assert atom.vector == [0.9, 0.8, 0.7]

  @pytest.mark.asyncio
  async def test_handles_embedding_failure_gracefully(self):
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(side_effect=RuntimeError("API down"))

    mem = Memory(embedder=embedder)

    atom = _atom(content="Test fact")
    # Should not raise.
    await mem._embed_atoms([atom])
    assert atom.vector is None

  @pytest.mark.asyncio
  async def test_uses_lossless_content_for_embedding(self):
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    mem = Memory(embedder=embedder)

    atom = MemoryEntry(
      session_id="s1",
      role="atom",
      content="short",
      entry_type="atom",
      lossless_content="The full disambiguated lossless content.",
    )
    await mem._embed_atoms([atom])

    embedder.async_get_embedding.assert_called_once_with("The full disambiguated lossless content.")


# --- Optimization with Embedding Tests ---


class TestOptimizationWithEmbedding:
  @pytest.mark.asyncio
  async def test_optimize_embeds_atoms_when_embedder_present(self):
    """When embedder is set, atoms produced by strategy get auto-embedded."""
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

    # Mock model that produces JSON atoms.
    mock_model = MagicMock()
    atoms_json = json.dumps([
      {"lossless_content": "Extracted fact.", "keywords": ["fact"], "entities": [], "persons": [], "topic": "test"},
    ])
    response = MagicMock()
    response.content = atoms_json
    mock_model.ainvoke = AsyncMock(return_value=response)

    from definable.memory.strategies.semantic import SemanticStrategy

    mem = Memory(store=store, strategy=SemanticStrategy(pin_count=1, recent_count=1), embedder=embedder, model=mock_model, max_messages=5)

    # Add enough messages to trigger optimization.
    for i in range(6):
      msg = Message(role="user" if i % 2 == 0 else "assistant", content=f"Msg {i}")
      await mem.add(msg, session_id="s1")

    # Verify atoms were embedded.
    entries = await mem.get_entries("s1")
    atom_entries = [e for e in entries if e.entry_type == "atom"]
    assert len(atom_entries) >= 1
    for atom in atom_entries:
      assert atom.vector == [0.1, 0.2, 0.3]


# --- MemoryEntry Vector Field Tests ---


class TestMemoryEntryVector:
  def test_vector_default_none(self):
    entry = MemoryEntry(session_id="s1")
    assert entry.vector is None

  def test_vector_to_dict_omitted_when_none(self):
    entry = MemoryEntry(session_id="s1")
    d = entry.to_dict()
    assert "vector" not in d

  def test_vector_to_dict_included_when_set(self):
    entry = MemoryEntry(session_id="s1", entry_type="atom", vector=[0.1, 0.2, 0.3])
    d = entry.to_dict()
    assert d["vector"] == [0.1, 0.2, 0.3]

  def test_vector_from_dict(self):
    data = {"session_id": "s1", "vector": [0.1, 0.2]}
    entry = MemoryEntry.from_dict(data)
    assert entry.vector == [0.1, 0.2]

  def test_vector_roundtrip(self):
    entry = MemoryEntry(session_id="s1", entry_type="atom", vector=[0.5, 0.6, 0.7])
    restored = MemoryEntry.from_dict(entry.to_dict())
    assert restored.vector == [0.5, 0.6, 0.7]

  def test_vector_from_dict_missing(self):
    data = {"session_id": "s1"}
    entry = MemoryEntry.from_dict(data)
    assert entry.vector is None


# --- Dual-Layer Recall Format Tests ---


class TestDualLayerRecallFormat:
  @pytest.mark.asyncio
  async def test_semantic_recall_format(self):
    """With embedder, recall produces <long_term_memory> + <short_term_memory> blocks."""
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mem = Memory(store=store, embedder=embedder, recent_count=2)
    await mem._ensure_initialized()

    # Add regular messages.
    await store.add(_msg(content="Hello", created_at=1.0, updated_at=1.0))
    await store.add(_msg(role="assistant", content="Hi there", created_at=2.0, updated_at=2.0))
    await store.add(_msg(content="How are you?", created_at=3.0, updated_at=3.0))
    await store.add(_msg(role="assistant", content="Fine, thanks!", created_at=4.0, updated_at=4.0))

    # Add atoms with vectors.
    await store.add(_atom(content="User name is Alice.", vector=[0.9, 0.1, 0.0], created_at=0.5, updated_at=0.5))
    await store.add(_atom(content="User prefers Python.", vector=[0.8, 0.2, 0.0], created_at=0.6, updated_at=0.6))

    # Simulate what the agent would call.
    entries = await mem.get_entries("s1")
    recent_msgs = [e for e in entries if e.entry_type == "message"][-2:]
    relevant_atoms = await mem.search("what does the user like", session_id="s1")

    # Verify search returns atoms.
    assert len(relevant_atoms) == 2

    # Verify recent messages are the last 2.
    assert len(recent_msgs) == 2
    assert recent_msgs[0].content == "How are you?"
    assert recent_msgs[1].content == "Fine, thanks!"

  @pytest.mark.asyncio
  async def test_chronological_recall_with_atoms_no_embedder(self):
    """Without embedder, atoms show as [Fact]: in chronological dump."""
    store = InMemoryStore()
    mem = Memory(store=store)  # No embedder
    await mem._ensure_initialized()

    await store.add(_msg(content="Hello", created_at=1.0, updated_at=1.0))
    await store.add(_atom(content="User is Alice.", created_at=2.0, updated_at=2.0))
    await store.add(_msg(content="Bye", created_at=3.0, updated_at=3.0))

    entries = await mem.get_entries("s1")

    # Simulate chronological formatting (as agent.py does).
    lines = []
    for e in entries:
      if e.role == "summary":
        lines.append(f"[Summary]: {e.content}")
      elif e.entry_type == "atom":
        lines.append(f"[Fact]: {e.lossless_content or e.content}")
      else:
        lines.append(f"{e.role}: {e.content}")

    assert lines[0] == "user: Hello"
    assert lines[1] == "[Fact]: User is Alice."
    assert lines[2] == "user: Bye"
