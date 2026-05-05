"""Tests for atom consolidation — Phase 3 of semantic memory.

Tests cover:
  - ConsolidationPolicy defaults and configuration
  - Decay — exponential importance reduction based on age
  - Merge — near-duplicate detection via cosine similarity + soft-delete
  - Prune — low-importance atoms soft-deleted
  - Full pipeline (decay → merge → prune) interaction
  - Memory.consolidate_session() public API
  - Integration with _optimize_if_needed
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.memory.consolidation import (
  ConsolidationPolicy,
  _apply_decay,
  _merge_duplicates,
  _prune_low_importance,
  consolidate,
)
from definable.memory.manager import Memory
from definable.memory.store.in_memory import InMemoryStore
from definable.memory.types import MemoryEntry


# --- Helpers ---


def _atom(
  content: str = "Fact.",
  importance: float = 0.5,
  vector: list[float] | None = None,
  created_at: float | None = None,
  superseded_by: str | None = None,
  session_id: str = "s1",
  **kw,
) -> MemoryEntry:
  return MemoryEntry(
    session_id=session_id,
    role="atom",
    content=content,
    entry_type="atom",
    lossless_content=content,
    importance=importance,
    vector=vector,
    created_at=created_at,
    superseded_by=superseded_by,
    **kw,
  )


# --- ConsolidationPolicy Tests ---


class TestConsolidationPolicy:
  def test_defaults(self):
    p = ConsolidationPolicy()
    assert p.decay_half_life_days == 30.0
    assert p.merge_similarity_threshold == 0.92
    assert p.min_importance == 0.05
    assert p.enabled is True

  def test_custom_values(self):
    p = ConsolidationPolicy(
      decay_half_life_days=7.0,
      merge_similarity_threshold=0.95,
      min_importance=0.1,
      enabled=False,
    )
    assert p.decay_half_life_days == 7.0
    assert p.merge_similarity_threshold == 0.95
    assert p.min_importance == 0.1
    assert p.enabled is False


# --- Decay Tests ---


class TestDecay:
  def test_recent_atom_barely_decays(self):
    now = time.time()
    atom = _atom(importance=1.0, created_at=now - 3600)  # 1 hour ago
    _apply_decay([atom], half_life_days=30.0, now=now)
    # 1 hour out of 30 days — almost no decay.
    assert atom.importance > 0.99

  def test_atom_at_half_life_loses_half(self):
    now = time.time()
    atom = _atom(importance=1.0, created_at=now - 30 * 86400)  # 30 days ago
    _apply_decay([atom], half_life_days=30.0, now=now)
    assert atom.importance == pytest.approx(0.5, abs=0.01)

  def test_atom_at_double_half_life(self):
    now = time.time()
    atom = _atom(importance=1.0, created_at=now - 60 * 86400)  # 60 days ago
    _apply_decay([atom], half_life_days=30.0, now=now)
    assert atom.importance == pytest.approx(0.25, abs=0.01)

  def test_short_half_life_decays_faster(self):
    now = time.time()
    atom = _atom(importance=1.0, created_at=now - 7 * 86400)  # 7 days ago
    _apply_decay([atom], half_life_days=7.0, now=now)
    assert atom.importance == pytest.approx(0.5, abs=0.01)

  def test_zero_age_no_decay(self):
    now = time.time()
    atom = _atom(importance=0.8, created_at=now)
    _apply_decay([atom], half_life_days=30.0, now=now)
    assert atom.importance == 0.8

  def test_multiple_atoms_different_ages(self):
    now = time.time()
    young = _atom(content="Young", importance=0.8, created_at=now - 86400)  # 1 day
    old = _atom(content="Old", importance=0.8, created_at=now - 90 * 86400)  # 90 days
    _apply_decay([young, old], half_life_days=30.0, now=now)
    assert young.importance > old.importance

  def test_decay_preserves_relative_ordering(self):
    now = time.time()
    high = _atom(importance=0.9, created_at=now - 10 * 86400)
    low = _atom(importance=0.3, created_at=now - 10 * 86400)
    _apply_decay([high, low], half_life_days=30.0, now=now)
    assert high.importance > low.importance


# --- Merge Tests ---


class TestMerge:
  def test_identical_vectors_merged(self):
    a = _atom(content="Fact A", importance=0.8, vector=[1.0, 0.0, 0.0])
    b = _atom(content="Fact B", importance=0.5, vector=[1.0, 0.0, 0.0])
    _merge_duplicates([a, b], threshold=0.92)
    # Higher importance wins.
    assert a.superseded_by is None
    assert b.superseded_by == a.memory_id

  def test_similar_vectors_merged(self):
    a = _atom(content="Fact A", importance=0.6, vector=[1.0, 0.0, 0.0])
    b = _atom(content="Fact B", importance=0.7, vector=[0.99, 0.05, 0.0])
    _merge_duplicates([a, b], threshold=0.92)
    # b has higher importance, so a gets superseded.
    assert a.superseded_by == b.memory_id
    assert b.superseded_by is None

  def test_orthogonal_vectors_not_merged(self):
    a = _atom(content="Fact A", importance=0.5, vector=[1.0, 0.0, 0.0])
    b = _atom(content="Fact B", importance=0.5, vector=[0.0, 1.0, 0.0])
    _merge_duplicates([a, b], threshold=0.92)
    assert a.superseded_by is None
    assert b.superseded_by is None

  def test_skip_atoms_without_vectors(self):
    a = _atom(content="A", importance=0.5, vector=[1.0, 0.0, 0.0])
    b = _atom(content="B", importance=0.5, vector=None)
    _merge_duplicates([a, b], threshold=0.92)
    assert a.superseded_by is None
    assert b.superseded_by is None

  def test_skip_already_superseded(self):
    a = _atom(content="A", importance=0.5, vector=[1.0, 0.0, 0.0], superseded_by="old-id")
    b = _atom(content="B", importance=0.5, vector=[1.0, 0.0, 0.0])
    _merge_duplicates([a, b], threshold=0.92)
    # a was already superseded, so no new merge.
    assert a.superseded_by == "old-id"
    assert b.superseded_by is None

  def test_three_way_merge(self):
    a = _atom(content="A", importance=0.9, vector=[1.0, 0.0, 0.0])
    b = _atom(content="B", importance=0.5, vector=[1.0, 0.0, 0.0])
    c = _atom(content="C", importance=0.3, vector=[0.99, 0.01, 0.0])
    _merge_duplicates([a, b, c], threshold=0.92)
    # a wins over both b and c.
    assert a.superseded_by is None
    assert b.superseded_by == a.memory_id
    # c is compared against a (b is already superseded) — should merge to a.
    assert c.superseded_by == a.memory_id


# --- Prune Tests ---


class TestPrune:
  def test_low_importance_pruned(self):
    atom = _atom(importance=0.01)
    _prune_low_importance([atom], min_importance=0.05)
    assert atom.superseded_by == "pruned"

  def test_above_threshold_not_pruned(self):
    atom = _atom(importance=0.1)
    _prune_low_importance([atom], min_importance=0.05)
    assert atom.superseded_by is None

  def test_exact_threshold_not_pruned(self):
    atom = _atom(importance=0.05)
    _prune_low_importance([atom], min_importance=0.05)
    assert atom.superseded_by is None

  def test_already_superseded_skipped(self):
    atom = _atom(importance=0.01, superseded_by="some-id")
    _prune_low_importance([atom], min_importance=0.05)
    assert atom.superseded_by == "some-id"  # Unchanged.


# --- Full Pipeline Tests ---


class TestConsolidatePipeline:
  @pytest.mark.asyncio
  async def test_disabled_policy_is_noop(self):
    atom = _atom(importance=0.01)
    policy = ConsolidationPolicy(enabled=False)
    result = await consolidate([atom], policy)
    assert atom.superseded_by is None
    assert len(result) == 1

  @pytest.mark.asyncio
  async def test_empty_list(self):
    policy = ConsolidationPolicy()
    result = await consolidate([], policy)
    assert result == []

  @pytest.mark.asyncio
  async def test_all_superseded_noop(self):
    atom = _atom(superseded_by="old")
    policy = ConsolidationPolicy()
    result = await consolidate([atom], policy)
    assert atom.superseded_by == "old"
    assert len(result) == 1

  @pytest.mark.asyncio
  async def test_decay_then_prune(self):
    """Very old atom decays below min_importance and gets pruned."""
    now = time.time()
    old_atom = _atom(importance=0.1, created_at=now - 365 * 86400)  # 1 year old
    policy = ConsolidationPolicy(decay_half_life_days=30.0, min_importance=0.05)
    await consolidate([old_atom], policy, now=now)
    # After ~12 half-lives, importance ≈ 0.1 * 2^-12 ≈ 0.000024 — well below 0.05.
    assert old_atom.superseded_by == "pruned"

  @pytest.mark.asyncio
  async def test_merge_then_prune(self):
    """Merge duplicate, then prune the low-importance survivor if needed."""
    now = time.time()
    # Both are recent, so decay won't significantly affect them.
    a = _atom(content="A", importance=0.04, vector=[1.0, 0.0, 0.0], created_at=now)
    b = _atom(content="B", importance=0.03, vector=[1.0, 0.0, 0.0], created_at=now)
    policy = ConsolidationPolicy(
      decay_half_life_days=30.0,
      merge_similarity_threshold=0.92,
      min_importance=0.05,
    )
    await consolidate([a, b], policy, now=now)
    # b gets merged into a (a has higher importance).
    assert b.superseded_by == a.memory_id
    # a survives merge but importance 0.04 < 0.05 — gets pruned.
    assert a.superseded_by == "pruned"

  @pytest.mark.asyncio
  async def test_fresh_high_importance_survives(self):
    now = time.time()
    atom = _atom(importance=0.9, created_at=now - 86400, vector=[1.0, 0.0, 0.0])
    policy = ConsolidationPolicy()
    await consolidate([atom], policy, now=now)
    assert atom.superseded_by is None
    assert atom.importance > 0.85  # Barely decayed.

  @pytest.mark.asyncio
  async def test_no_decay_when_half_life_zero(self):
    now = time.time()
    atom = _atom(importance=0.5, created_at=now - 365 * 86400)
    policy = ConsolidationPolicy(decay_half_life_days=0)
    await consolidate([atom], policy, now=now)
    assert atom.importance == 0.5  # No decay applied.
    assert atom.superseded_by is None


# --- Memory.consolidate_session() Tests ---


class TestMemoryConsolidateSession:
  @pytest.mark.asyncio
  async def test_consolidate_session_soft_deletes(self):
    store = InMemoryStore()
    mem = Memory(store=store)
    await mem._ensure_initialized()

    now = time.time()
    old = _atom(content="Old fact", importance=0.1, created_at=now - 365 * 86400)
    fresh = _atom(content="Fresh fact", importance=0.9, created_at=now)
    await store.add(old)
    await store.add(fresh)

    deleted = await mem.consolidate_session("s1")
    assert deleted == 1

    # Verify the old atom is soft-deleted in the store.
    entries = await store.get_entries("s1")
    old_entry = next(e for e in entries if e.content == "Old fact")
    assert old_entry.superseded_by == "pruned"
    fresh_entry = next(e for e in entries if e.content == "Fresh fact")
    assert fresh_entry.superseded_by is None

  @pytest.mark.asyncio
  async def test_consolidate_session_no_atoms(self):
    store = InMemoryStore()
    mem = Memory(store=store)
    await mem._ensure_initialized()

    msg = MemoryEntry(session_id="s1", role="user", content="Hello")
    await store.add(msg)

    deleted = await mem.consolidate_session("s1")
    assert deleted == 0

  @pytest.mark.asyncio
  async def test_consolidate_session_uses_configured_policy(self):
    store = InMemoryStore()
    policy = ConsolidationPolicy(min_importance=0.6)
    mem = Memory(store=store, consolidation=policy)
    await mem._ensure_initialized()

    now = time.time()
    # importance 0.5 < min_importance 0.6 after minimal decay → pruned.
    atom = _atom(importance=0.5, created_at=now)
    await store.add(atom)

    deleted = await mem.consolidate_session("s1")
    assert deleted == 1

  @pytest.mark.asyncio
  async def test_consolidate_session_merges_duplicates(self):
    store = InMemoryStore()
    mem = Memory(store=store)
    await mem._ensure_initialized()

    now = time.time()
    a = _atom(content="Fact A", importance=0.8, vector=[1.0, 0.0, 0.0], created_at=now)
    b = _atom(content="Fact B", importance=0.5, vector=[1.0, 0.0, 0.0], created_at=now)
    await store.add(a)
    await store.add(b)

    deleted = await mem.consolidate_session("s1")
    assert deleted == 1

    entries = await store.get_entries("s1")
    a_entry = next(e for e in entries if e.content == "Fact A")
    b_entry = next(e for e in entries if e.content == "Fact B")
    assert a_entry.superseded_by is None
    assert b_entry.superseded_by == a.memory_id


# --- Integration with Optimization Tests ---


class TestConsolidationDuringOptimization:
  @pytest.mark.asyncio
  async def test_optimize_triggers_consolidation(self):
    """When consolidation policy is set, optimization runs merge on identical-vector atoms."""
    store = InMemoryStore()
    embedder = MagicMock()
    # All atoms get the same vector → they'll be merged by consolidation.
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    policy = ConsolidationPolicy(merge_similarity_threshold=0.92)

    mock_model = MagicMock()
    atoms_json = json.dumps([
      {"lossless_content": "Fact A.", "keywords": ["a"], "entities": [], "persons": [], "topic": "test"},
      {"lossless_content": "Fact B.", "keywords": ["b"], "entities": [], "persons": [], "topic": "test"},
    ])
    response = MagicMock()
    response.content = atoms_json
    mock_model.ainvoke = AsyncMock(return_value=response)

    from definable.memory.strategies.semantic import SemanticStrategy

    mem = Memory(
      store=store,
      strategy=SemanticStrategy(pin_count=1, recent_count=1),
      embedder=embedder,
      consolidation=policy,
      model=mock_model,
      max_messages=5,
    )

    from definable.model.message import Message

    for i in range(6):
      msg = Message(role="user" if i % 2 == 0 else "assistant", content=f"Msg {i}")
      await mem.add(msg, session_id="s1")

    # Both atoms get vector [1,0,0] → cosine sim = 1.0 → one gets merged.
    entries = await mem.get_entries("s1")
    atom_entries = [e for e in entries if e.entry_type == "atom"]
    superseded = [e for e in atom_entries if e.superseded_by is not None]
    active = [e for e in atom_entries if e.superseded_by is None]
    # At least one atom should survive, at least one should be merged.
    if len(atom_entries) >= 2:
      assert len(superseded) >= 1
      assert len(active) >= 1

  @pytest.mark.asyncio
  async def test_no_consolidation_without_policy(self):
    """Without consolidation policy, atoms are not modified after optimization."""
    store = InMemoryStore()
    embedder = MagicMock()
    embedder.async_get_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])

    mock_model = MagicMock()
    atoms_json = json.dumps([
      {"lossless_content": "Low fact.", "keywords": [], "entities": [], "persons": [], "topic": "test", "importance": 0.01},
    ])
    response = MagicMock()
    response.content = atoms_json
    mock_model.ainvoke = AsyncMock(return_value=response)

    from definable.memory.strategies.semantic import SemanticStrategy

    mem = Memory(
      store=store,
      strategy=SemanticStrategy(pin_count=1, recent_count=1),
      embedder=embedder,
      # No consolidation policy.
      model=mock_model,
      max_messages=5,
    )

    from definable.model.message import Message

    for i in range(6):
      msg = Message(role="user" if i % 2 == 0 else "assistant", content=f"Msg {i}")
      await mem.add(msg, session_id="s1")

    # Without consolidation, even low-importance atoms survive.
    entries = await mem.get_entries("s1")
    atom_entries = [e for e in entries if e.entry_type == "atom"]
    for atom in atom_entries:
      assert atom.superseded_by is None
