"""5-factor composite relevance scorer for memory retrieval."""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Union

from definable.memory.config import MemoryConfig, ScoringWeights
from definable.memory.types import Episode, KnowledgeAtom, Procedure


@dataclass
class ScoredMemory:
  """A memory candidate with its computed relevance score."""

  memory: Union[Episode, KnowledgeAtom, Procedure]
  score: float
  token_count: int
  formatted: str  # Pre-formatted string for this memory chunk


def score_candidate(
  candidate: Union[Episode, KnowledgeAtom, Procedure],
  *,
  query_embedding: Optional[List[float]] = None,
  candidate_embedding: Optional[List[float]] = None,
  predicted_topics: Optional[Set[str]] = None,
  max_access_count: int = 1,
  config: Optional[MemoryConfig] = None,
  weights: Optional[ScoringWeights] = None,
) -> float:
  """Compute a 5-factor composite relevance score for a memory candidate.

  Factors:
    1. Semantic similarity (cosine between query and memory embedding)
    2. Recency (exponential decay based on time since last access)
    3. Access frequency (log-normalized access count)
    4. Predicted need (topic overlap with predicted next topics)
    5. Emotional salience (absolute value of sentiment)

  Returns:
      Score between 0.0 and 1.0.
  """
  cfg = config or MemoryConfig()
  w = weights or cfg.scoring_weights
  now = time.time()

  # 1. Semantic similarity
  semantic = 0.0
  if query_embedding and candidate_embedding and len(query_embedding) == len(candidate_embedding):
    semantic = _cosine_similarity(query_embedding, candidate_embedding)
    semantic = max(0.0, semantic)  # Clamp negative similarities

  # 2. Recency
  last_accessed = _get_last_accessed(candidate)
  hours_since = (now - last_accessed) / 3600.0
  decay_rate = math.log(2) / (cfg.decay_half_life_days * 24.0)
  recency = math.exp(-decay_rate * hours_since)

  # 3. Access frequency
  access_count = _get_access_count(candidate)
  safe_max = max(max_access_count, 1)
  frequency = math.log(1 + access_count) / math.log(1 + safe_max)

  # 4. Predicted need
  predicted = 0.0
  if predicted_topics:
    candidate_topics = set(_get_topics(candidate))
    overlap = candidate_topics & predicted_topics
    if overlap:
      predicted = len(overlap) / len(predicted_topics)

  # 5. Emotional salience
  salience = abs(_get_sentiment(candidate))

  # Weighted sum
  score = (
    w.semantic_similarity * semantic
    + w.recency * recency
    + w.access_frequency * frequency
    + w.predicted_need * predicted
    + w.emotional_salience * salience
  )
  return min(1.0, max(0.0, score))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  mag_a = math.sqrt(sum(x * x for x in a))
  mag_b = math.sqrt(sum(x * x for x in b))
  if mag_a == 0.0 or mag_b == 0.0:
    return 0.0
  return dot / (mag_a * mag_b)


def _get_last_accessed(candidate: Union[Episode, KnowledgeAtom, Procedure]) -> float:
  return getattr(candidate, "last_accessed_at", 0.0) or time.time()


def _get_access_count(candidate: Union[Episode, KnowledgeAtom, Procedure]) -> int:
  return getattr(candidate, "access_count", 0) or 0


def _get_topics(candidate: Union[Episode, KnowledgeAtom, Procedure]) -> List[str]:
  return getattr(candidate, "topics", []) or []


def _get_sentiment(candidate: Union[Episode, KnowledgeAtom, Procedure]) -> float:
  return getattr(candidate, "sentiment", 0.0) or 0.0
