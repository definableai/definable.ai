"""Shared fixtures for memory tests."""

import os
import tempfile

import pytest

from definable.memory.config import MemoryConfig, ScoringWeights
from definable.memory.store.sqlite import SQLiteMemoryStore


@pytest.fixture
async def sqlite_store():
  """Temporary SQLiteMemoryStore for testing."""
  with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

  store = SQLiteMemoryStore(db_path=db_path)
  await store.initialize()
  yield store
  await store.close()
  os.unlink(db_path)


@pytest.fixture
def memory_config():
  """Default MemoryConfig for testing."""
  return MemoryConfig(
    decay_half_life_days=14,
    scoring_weights=ScoringWeights(),
    distillation_stage_0_age=1.0,  # 1 second for testing
    distillation_stage_1_age=1.0,
    distillation_stage_2_age=1.0,
    distillation_stage_3_age=1.0,
    distillation_batch_size=10,
  )
