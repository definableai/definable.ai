"""Tests for scorer and retrieval pipeline."""

import time

import pytest

from definable.memory.config import MemoryConfig, ScoringWeights
from definable.memory.retrieval import _format_candidate, _format_payload, recall_memories
from definable.memory.scorer import ScoredMemory, score_candidate
from definable.memory.topics import extract_topics, predict_next_topics
from definable.memory.types import Episode, KnowledgeAtom, Procedure, TopicTransition


class TestScoringWeights:
  """Tests for ScoringWeights validation."""

  def test_default_weights_sum_to_one(self):
    w = ScoringWeights()
    total = w.semantic_similarity + w.recency + w.access_frequency + w.predicted_need + w.emotional_salience
    assert abs(total - 1.0) < 1e-6

  def test_custom_weights_valid(self):
    w = ScoringWeights(
      semantic_similarity=0.5,
      recency=0.2,
      access_frequency=0.1,
      predicted_need=0.1,
      emotional_salience=0.1,
    )
    assert w.semantic_similarity == 0.5

  def test_weights_must_sum_to_one(self):
    with pytest.raises(ValueError, match="must sum to 1.0"):
      ScoringWeights(
        semantic_similarity=0.5,
        recency=0.5,
        access_frequency=0.5,
        predicted_need=0.0,
        emotional_salience=0.0,
      )


class TestScoreCandidate:
  """Tests for the 5-factor composite scorer."""

  def test_score_recent_episode(self):
    now = time.time()
    episode = Episode(
      id="e1",
      user_id="u1",
      session_id="s1",
      role="user",
      content="test",
      topics=["python"],
      sentiment=0.5,
      last_accessed_at=now,
      access_count=5,
    )
    score = score_candidate(
      episode,
      max_access_count=10,
      config=MemoryConfig(),
    )
    # Recent episode with some access count and sentiment should score > 0
    assert score > 0.0
    assert score <= 1.0

  def test_score_with_semantic_similarity(self):
    now = time.time()
    atom = KnowledgeAtom(
      id="a1",
      user_id="u1",
      subject="s",
      predicate="p",
      object="o",
      content="test",
      confidence=0.8,
      embedding=[0.9, 0.1, 0.0],
      last_accessed_at=now,
      access_count=1,
    )
    score_high = score_candidate(
      atom,
      query_embedding=[0.9, 0.1, 0.0],
      candidate_embedding=atom.embedding,
      max_access_count=1,
    )
    score_low = score_candidate(
      atom,
      query_embedding=[0.0, 0.0, 1.0],
      candidate_embedding=atom.embedding,
      max_access_count=1,
    )
    assert score_high > score_low

  def test_score_with_predicted_topics(self):
    now = time.time()
    episode_matching = Episode(
      id="e1",
      user_id="u1",
      session_id="s1",
      role="user",
      content="test",
      topics=["debugging"],
      last_accessed_at=now,
    )
    episode_nonmatching = Episode(
      id="e2",
      user_id="u1",
      session_id="s1",
      role="user",
      content="test",
      topics=["cooking"],
      last_accessed_at=now,
    )
    score_match = score_candidate(
      episode_matching,
      predicted_topics={"debugging", "testing"},
      max_access_count=1,
    )
    score_nomatch = score_candidate(
      episode_nonmatching,
      predicted_topics={"debugging", "testing"},
      max_access_count=1,
    )
    assert score_match > score_nomatch

  def test_score_old_episode_decays(self):
    now = time.time()
    recent = Episode(
      id="e1",
      user_id="u1",
      session_id="s1",
      role="user",
      content="test",
      last_accessed_at=now,
    )
    old = Episode(
      id="e2",
      user_id="u1",
      session_id="s1",
      role="user",
      content="test",
      last_accessed_at=now - 86400 * 30,  # 30 days ago
    )
    score_recent = score_candidate(recent, max_access_count=1)
    score_old = score_candidate(old, max_access_count=1)
    assert score_recent > score_old


class TestTopicExtraction:
  """Tests for topic extraction."""

  def test_extract_topics_basic(self):
    topics = extract_topics("I want to learn about Python programming and machine learning")
    assert len(topics) > 0
    assert "python" in topics or "programming" in topics or "machine" in topics

  def test_extract_topics_empty(self):
    assert extract_topics("") == []
    assert extract_topics("the a is") == []

  def test_extract_topics_max(self):
    topics = extract_topics("python java rust golang typescript swift kotlin", max_topics=3)
    assert len(topics) <= 3

  def test_predict_next_topics(self):
    transitions = {
      "python": [
        TopicTransition(from_topic="python", to_topic="testing", count=10, probability=0.5),
        TopicTransition(from_topic="python", to_topic="debugging", count=6, probability=0.3),
        TopicTransition(from_topic="python", to_topic="deployment", count=4, probability=0.2),
      ],
    }
    predicted = predict_next_topics(["python"], transitions, min_probability=0.3)
    assert "testing" in predicted
    assert "debugging" in predicted
    assert "deployment" not in predicted  # Below threshold

  def test_predict_excludes_current(self):
    transitions = {
      "python": [
        TopicTransition(from_topic="python", to_topic="python", count=10, probability=0.5),
        TopicTransition(from_topic="python", to_topic="testing", count=6, probability=0.3),
      ],
    }
    predicted = predict_next_topics(["python"], transitions, min_probability=0.3)
    assert "python" not in predicted


class TestFormatting:
  """Tests for memory payload formatting."""

  def test_format_atom_candidate(self):
    atom = KnowledgeAtom(
      id="a1",
      user_id="u1",
      subject="user",
      predicate="lives_in",
      object="SF",
      content="user lives in San Francisco",
      confidence=0.9,
    )
    formatted = _format_candidate(atom)
    assert "user lives in San Francisco" in formatted
    assert "0.90" in formatted

  def test_format_procedure_candidate(self):
    proc = Procedure(
      id="p1",
      user_id="u1",
      trigger="user asks for code",
      action="use Python with type hints",
    )
    formatted = _format_candidate(proc)
    assert "user asks for code" in formatted
    assert "use Python with type hints" in formatted

  def test_format_episode_candidate(self):
    ep = Episode(
      id="e1",
      user_id="u1",
      session_id="s1",
      role="user",
      content="Hello world",
    )
    formatted = _format_candidate(ep)
    assert "[user]" in formatted
    assert "Hello world" in formatted

  def test_format_payload_xml(self):
    items = [
      ScoredMemory(
        memory=KnowledgeAtom(
          id="a1",
          user_id="u1",
          subject="user",
          predicate="lives",
          object="SF",
          content="user lives in SF",
          confidence=0.9,
        ),
        score=0.8,
        token_count=5,
        formatted="- user lives in SF (confidence: 0.90)",
      ),
      ScoredMemory(
        memory=Procedure(
          id="p1",
          user_id="u1",
          trigger="code request",
          action="use Python",
        ),
        score=0.6,
        token_count=5,
        formatted="- When code request: use Python",
      ),
    ]
    payload = _format_payload(items)
    assert "<memory_context>" in payload
    assert "</memory_context>" in payload
    assert "user_profile" in payload or "known_facts" in payload
    assert "<preferences>" in payload


@pytest.mark.asyncio
class TestRecallPipeline:
  """Integration tests for the recall pipeline."""

  async def test_recall_no_memories(self, sqlite_store):
    result = await recall_memories(sqlite_store, "hello", token_budget=500)
    assert result is None

  async def test_recall_with_episodes(self, sqlite_store):
    now = time.time()
    for i in range(5):
      ep = Episode(
        id=f"ep-{i}",
        user_id="u1",
        session_id="s1",
        role="user",
        content=f"I enjoy Python programming and testing code {i}",
        topics=["python", "testing"],
        token_count=10,
        created_at=now - i * 60,
        last_accessed_at=now - i * 60,
      )
      await sqlite_store.store_episode(ep)

    result = await recall_memories(
      sqlite_store,
      "How do I test Python code?",
      token_budget=200,
      user_id="u1",
      session_id="s1",
    )
    assert result is not None
    assert result.tokens_used > 0
    assert result.chunks_included > 0
    assert "<memory_context>" in result.context

  async def test_recall_respects_budget(self, sqlite_store):
    now = time.time()
    # Create episodes with known token counts
    for i in range(20):
      ep = Episode(
        id=f"ep-budget-{i}",
        user_id="u1",
        session_id="s1",
        role="user",
        content=f"Content message number {i} with some filler text to use tokens about testing code",
        topics=["testing"],
        token_count=50,
        created_at=now - i * 60,
        last_accessed_at=now - i * 60,
      )
      await sqlite_store.store_episode(ep)

    result = await recall_memories(
      sqlite_store,
      "testing code",
      token_budget=100,
      user_id="u1",
      session_id="s1",
    )
    assert result is not None
    assert result.tokens_used <= 100
