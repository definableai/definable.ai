"""Tests for temporal decay scoring."""

import time


from definable.knowledge.document import Document
from definable.knowledge.scoring.temporal import TemporalDecay


class TestTemporalDecay:
  def test_decay_factor_zero_age(self):
    decay = TemporalDecay(half_life_days=30.0)
    assert decay.decay_factor(0) == 1.0

  def test_decay_factor_one_half_life(self):
    decay = TemporalDecay(half_life_days=30.0)
    factor = decay.decay_factor(30.0)
    assert abs(factor - 0.5) < 0.01

  def test_decay_factor_two_half_lives(self):
    decay = TemporalDecay(half_life_days=30.0)
    factor = decay.decay_factor(60.0)
    assert abs(factor - 0.25) < 0.01

  def test_disabled_returns_unchanged(self):
    decay = TemporalDecay(enabled=False)
    docs = [Document(content="test", reranking_score=1.0)]
    result = decay.apply(docs)
    assert result[0].reranking_score == 1.0

  def test_empty_list(self):
    decay = TemporalDecay()
    assert decay.apply([]) == []

  def test_evergreen_exempt(self):
    now = time.time()
    decay = TemporalDecay(half_life_days=1.0)
    docs = [
      Document(
        content="old but evergreen",
        reranking_score=1.0,
        meta_data={"inserted_at": now - 86400 * 100, "evergreen": True},
      ),
    ]
    result = decay.apply(docs, now=now)
    # Score should be unchanged (evergreen)
    assert result[0].reranking_score == 1.0

  def test_no_timestamp_unchanged(self):
    decay = TemporalDecay(half_life_days=30.0)
    docs = [Document(content="no timestamp", reranking_score=0.8)]
    result = decay.apply(docs)
    assert result[0].reranking_score == 0.8

  def test_old_doc_decayed(self):
    now = time.time()
    decay = TemporalDecay(half_life_days=30.0)
    docs = [
      Document(
        content="old doc",
        reranking_score=1.0,
        meta_data={"inserted_at": now - 86400 * 30},  # 30 days old
      ),
    ]
    result = decay.apply(docs, now=now)
    assert result[0].reranking_score < 0.6  # Should be ~0.5

  def test_sorts_by_decayed_score(self):
    now = time.time()
    decay = TemporalDecay(half_life_days=7.0)
    docs = [
      Document(
        content="old",
        reranking_score=1.0,
        meta_data={"inserted_at": now - 86400 * 30},
      ),
      Document(
        content="recent",
        reranking_score=0.8,
        meta_data={"inserted_at": now - 86400 * 1},
      ),
    ]
    result = decay.apply(docs, now=now)
    # Recent doc should rank higher despite lower base score
    assert result[0].content == "recent"

  def test_uses_created_at_fallback(self):
    now = time.time()
    decay = TemporalDecay(half_life_days=30.0)
    docs = [
      Document(
        content="test",
        reranking_score=1.0,
        meta_data={"created_at": now - 86400 * 30},
      ),
    ]
    result = decay.apply(docs, now=now)
    assert result[0].reranking_score < 0.6
