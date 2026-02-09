"""Tests for progressive distillation engine."""

import time

import pytest

from definable.memory.config import MemoryConfig
from definable.memory.distillation import DistillationResult, _parse_spo, run_distillation
from definable.memory.types import Episode, KnowledgeAtom


class TestParseSPO:
  """Tests for subject-predicate-object parsing."""

  def test_basic_parse(self):
    result = _parse_spo("user lives in San Francisco")
    assert result is not None
    assert result[0] == "user"
    assert result[1] == "lives"
    assert result[2] == "in San Francisco"

  def test_short_string_returns_none(self):
    assert _parse_spo("hello world") is None
    assert _parse_spo("hi") is None
    assert _parse_spo("") is None


@pytest.mark.asyncio
class TestDistillationStage0:
  """Tests for raw -> summary distillation (without LLM)."""

  async def test_stage_0_to_1_truncation(self, sqlite_store):
    """Without a model, distillation truncates long content."""
    now = time.time()
    # Use a config where only stage 0 is old enough to process
    config = MemoryConfig(
      distillation_stage_0_age=1.0,  # 1 second threshold
      distillation_stage_1_age=86400.0,  # 24 hours — won't trigger
      distillation_stage_2_age=604800.0,  # 7 days — won't trigger
    )
    old_time = now - 10  # 10 seconds ago

    long_content = "A" * 300
    episode = Episode(
      id="ep-dist-0",
      user_id="u1",
      session_id="s1",
      role="user",
      content=long_content,
      compression_stage=0,
      created_at=old_time,
      last_accessed_at=old_time,
    )
    await sqlite_store.store_episode(episode)

    result = await run_distillation(sqlite_store, config=config)
    assert result.episodes_processed >= 1

    # Check episode was updated to stage 1
    episodes = await sqlite_store.get_episodes(user_id="u1")
    updated = [e for e in episodes if e.id == "ep-dist-0"]
    assert len(updated) == 1
    assert updated[0].compression_stage == 1
    assert len(updated[0].content) <= 200

  async def test_stage_0_skips_recent(self, sqlite_store, memory_config):
    """Recent episodes should not be distilled."""
    now = time.time()
    episode = Episode(
      id="ep-recent",
      user_id="u1",
      session_id="s1",
      role="user",
      content="Very recent content",
      compression_stage=0,
      created_at=now,
      last_accessed_at=now,
    )
    await sqlite_store.store_episode(episode)

    # Use a config with 1 hour threshold
    cfg = MemoryConfig(distillation_stage_0_age=3600)
    result = await run_distillation(sqlite_store, config=cfg)
    assert result.episodes_processed == 0


@pytest.mark.asyncio
class TestDistillationStage2:
  """Tests for facts -> atoms distillation."""

  async def test_stage_2_to_3_creates_atoms(self, sqlite_store, memory_config):
    now = time.time()
    old_time = now - memory_config.distillation_stage_2_age - 1

    episode = Episode(
      id="ep-facts",
      user_id="u1",
      session_id="s1",
      role="user",
      content="user lives in San Francisco; user prefers Python programming",
      compression_stage=2,
      created_at=old_time,
      last_accessed_at=old_time,
    )
    await sqlite_store.store_episode(episode)

    result = await run_distillation(sqlite_store, config=memory_config)
    assert result.atoms_created >= 1

    # Verify atoms were stored
    atoms = await sqlite_store.get_atoms(user_id="u1", min_confidence=0.0)
    assert len(atoms) >= 1

  async def test_atom_reinforcement(self, sqlite_store, memory_config):
    """Duplicate facts should reinforce existing atoms."""
    now = time.time()
    old_time = now - memory_config.distillation_stage_2_age - 1

    # Pre-create an existing atom
    atom = KnowledgeAtom(
      id="existing-atom",
      user_id="u1",
      subject="user",
      predicate="lives",
      object="in San Francisco",
      content="user lives in San Francisco",
      confidence=0.5,
      reinforcement_count=0,
      created_at=now,
      last_accessed_at=now,
    )
    await sqlite_store.store_atom(atom)

    # Create a fact episode with matching content
    episode = Episode(
      id="ep-reinforce",
      user_id="u1",
      session_id="s1",
      role="user",
      content="user lives in San Francisco",
      compression_stage=2,
      created_at=old_time,
      last_accessed_at=old_time,
    )
    await sqlite_store.store_episode(episode)

    result = await run_distillation(sqlite_store, config=memory_config)
    assert result.atoms_reinforced >= 1

    # Check that the atom's confidence was boosted
    atoms = await sqlite_store.get_atoms(user_id="u1")
    existing = [a for a in atoms if a.id == "existing-atom"]
    assert len(existing) == 1
    assert existing[0].confidence > 0.5


@pytest.mark.asyncio
class TestDistillationResult:
  """Tests for distillation result tracking."""

  async def test_empty_distillation(self, sqlite_store, memory_config):
    result = await run_distillation(sqlite_store, config=memory_config)
    assert isinstance(result, DistillationResult)
    assert result.episodes_processed == 0
    assert result.atoms_created == 0
    assert result.atoms_reinforced == 0
    assert result.atoms_pruned == 0
