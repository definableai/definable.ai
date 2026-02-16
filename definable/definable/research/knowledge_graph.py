"""In-memory fact accumulator with deduplication and contradiction detection."""

import re
from typing import Dict, List, Set

from definable.research.models import CKU, Contradiction, Fact, SourceInfo


def _normalize(text: str) -> Set[str]:
  """Lowercase, strip punctuation, remove stopwords, return word set."""
  stopwords = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "not", "no", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she",
    "they", "them", "their", "our", "my", "your",
  }
  words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
  return words - stopwords


def _jaccard(a: Set[str], b: Set[str]) -> float:
  """Jaccard similarity between two word sets."""
  if not a or not b:
    return 0.0
  return len(a & b) / len(a | b)


_NUMBER_RE = re.compile(r"\b[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|trillion|%|percent))?\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


class KnowledgeGraph:
  """Lightweight in-memory fact accumulator.

  NOT a graph database — just a structured store for facts collected
  during research, with deduplication and contradiction detection.
  """

  def __init__(self) -> None:
    self._facts: List[Fact] = []
    self._sources: Dict[str, SourceInfo] = {}  # url -> SourceInfo
    self._fact_norms: List[Set[str]] = []  # parallel to _facts, for dedup
    self._topic_facts: Dict[str, List[int]] = {}  # topic -> fact indices

  def ingest(self, ckus: List[CKU]) -> int:
    """Ingest CKUs, deduplicating facts.

    Returns the number of new (non-duplicate) facts added.
    """
    new_count = 0
    for cku in ckus:
      # Track source
      url = cku.source_url
      if url not in self._sources:
        self._sources[url] = SourceInfo(
          url=url,
          title=cku.source_title,
          relevance_score=cku.relevance_score,
          fact_count=0,
        )

      topic = cku.query_context

      for fact in cku.facts:
        norm = _normalize(fact.content)
        # Dedup: check Jaccard similarity against existing facts
        is_dup = False
        for existing_norm in self._fact_norms:
          if _jaccard(norm, existing_norm) > 0.8:
            is_dup = True
            break

        if not is_dup:
          idx = len(self._facts)
          self._facts.append(fact)
          self._fact_norms.append(norm)
          self._sources[url].fact_count += 1
          if topic not in self._topic_facts:
            self._topic_facts[topic] = []
          self._topic_facts[topic].append(idx)
          new_count += 1

    return new_count

  def get_facts_by_topic(self, topic: str) -> List[Fact]:
    """Get all facts for a given topic/sub-question."""
    indices = self._topic_facts.get(topic, [])
    return [self._facts[i] for i in indices]

  def get_all_facts(self) -> List[Fact]:
    """Get all unique facts."""
    return list(self._facts)

  def get_contradictions(self) -> List[Contradiction]:
    """Detect contradictions: facts with overlapping entities but different numbers."""
    contradictions: List[Contradiction] = []
    n = len(self._facts)
    for i in range(n):
      for j in range(i + 1, n):
        c = self._check_contradiction(self._facts[i], self._facts[j])
        if c:
          contradictions.append(c)
    return contradictions

  def _check_contradiction(self, a: Fact, b: Fact) -> Contradiction | None:
    """Check if two facts contradict each other.

    Simple heuristic: overlapping entities + same topic area + different numbers.
    """
    # Both must have entities
    entities_a = set(e.lower() for e in a.entities)
    entities_b = set(e.lower() for e in b.entities)
    if not entities_a or not entities_b:
      return None
    # Must share at least one entity
    if not entities_a & entities_b:
      return None

    # Extract numbers from each
    nums_a = set(_NUMBER_RE.findall(a.content))
    nums_b = set(_NUMBER_RE.findall(b.content))
    if not nums_a or not nums_b:
      return None

    # Must have different numbers for the same context
    # Word overlap (excluding numbers) should be high
    words_a = _normalize(a.content) - {n.replace(",", "").strip() for n in nums_a}
    words_b = _normalize(b.content) - {n.replace(",", "").strip() for n in nums_b}
    if _jaccard(words_a, words_b) < 0.3:
      return None

    # Numbers differ
    if nums_a != nums_b:
      return Contradiction(fact_a=a, fact_b=b)

    return None

  def get_sources(self) -> List[SourceInfo]:
    """Get all tracked sources."""
    return list(self._sources.values())

  def fact_count_for_topic(self, topic: str) -> int:
    """Count facts for a given topic."""
    return len(self._topic_facts.get(topic, []))

  @property
  def total_facts(self) -> int:
    return len(self._facts)
