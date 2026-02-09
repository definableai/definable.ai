"""Topic extraction and transition model for predictive pre-loading."""

import re
from collections import Counter
from typing import Dict, List, Optional, Set

from definable.memory.types import TopicTransition

# Common English stop words to filter out
_STOP_WORDS: Set[str] = {
  "a",
  "an",
  "the",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "shall",
  "can",
  "need",
  "dare",
  "ought",
  "and",
  "but",
  "or",
  "nor",
  "not",
  "so",
  "yet",
  "both",
  "either",
  "neither",
  "each",
  "every",
  "all",
  "any",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "only",
  "own",
  "same",
  "than",
  "too",
  "very",
  "just",
  "because",
  "as",
  "until",
  "while",
  "of",
  "at",
  "by",
  "for",
  "with",
  "about",
  "against",
  "between",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "to",
  "from",
  "up",
  "down",
  "in",
  "out",
  "on",
  "off",
  "over",
  "under",
  "again",
  "further",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "what",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "i",
  "me",
  "my",
  "myself",
  "we",
  "our",
  "ours",
  "ourselves",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves",
  "he",
  "him",
  "his",
  "himself",
  "she",
  "her",
  "hers",
  "herself",
  "it",
  "its",
  "itself",
  "they",
  "them",
  "their",
  "theirs",
  "themselves",
  "if",
  "also",
  "like",
  "get",
  "got",
  "go",
  "going",
  "went",
  "come",
  "came",
  "make",
  "made",
  "take",
  "took",
  "know",
  "knew",
  "think",
  "thought",
  "want",
  "say",
  "said",
  "tell",
  "told",
  "use",
  "used",
  "try",
  "tried",
  "really",
  "well",
  "right",
  "much",
  "many",
  "good",
  "bad",
  "great",
  "please",
  "thank",
  "thanks",
  "yes",
  "no",
  "ok",
  "okay",
  "sure",
  "let",
  "lets",
  "help",
  "thing",
  "things",
  "way",
  "something",
}

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def extract_topics(text: str, max_topics: int = 5) -> List[str]:
  """Extract topic keywords from text using word frequency scoring.

  Simple TF-based extraction: frequent words that are not stop words
  and have a minimum length are considered topics.

  Args:
      text: Input text to extract topics from.
      max_topics: Maximum number of topics to return.

  Returns:
      List of topic strings, ordered by frequency.
  """
  if not text:
    return []

  words = _WORD_PATTERN.findall(text.lower())
  # Filter out stop words and very short words
  meaningful = [w for w in words if w not in _STOP_WORDS and len(w) >= 3]

  if not meaningful:
    return []

  # Count frequencies
  counter = Counter(meaningful)

  # Return top topics
  return [word for word, _ in counter.most_common(max_topics)]


def predict_next_topics(
  current_topics: List[str],
  transitions: Dict[str, List[TopicTransition]],
  min_probability: float = 0.3,
) -> List[str]:
  """Predict likely next topics based on transition model.

  Args:
      current_topics: Topics from the current query.
      transitions: Map of from_topic -> list of transitions.
      min_probability: Minimum transition probability to include.

  Returns:
      List of predicted topic strings, ordered by probability.
  """
  predicted: Dict[str, float] = {}

  for topic in current_topics:
    topic_transitions = transitions.get(topic, [])
    for tr in topic_transitions:
      if tr.probability >= min_probability:
        # Take the max probability if a topic appears from multiple sources
        existing = predicted.get(tr.to_topic, 0.0)
        predicted[tr.to_topic] = max(existing, tr.probability)

  # Remove topics that are already current
  current_set = set(current_topics)
  predicted = {t: p for t, p in predicted.items() if t not in current_set}

  # Sort by probability descending
  sorted_topics = sorted(predicted.items(), key=lambda x: x[1], reverse=True)
  return [t for t, _ in sorted_topics]


async def get_transition_map(
  store,
  topics: List[str],
  user_id: Optional[str] = None,
  min_count: int = 3,
) -> Dict[str, List[TopicTransition]]:
  """Build a transition map from the store for given topics.

  Args:
      store: MemoryStore instance.
      topics: Topics to look up transitions for.
      user_id: Optional user ID filter.
      min_count: Minimum transition count to include.

  Returns:
      Dict mapping from_topic to list of TopicTransition.
  """
  transition_map: Dict[str, List[TopicTransition]] = {}
  for topic in topics:
    transitions = await store.get_topic_transitions(topic, user_id=user_id, min_count=min_count)
    if transitions:
      transition_map[topic] = transitions
  return transition_map
