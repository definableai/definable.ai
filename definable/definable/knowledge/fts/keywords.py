"""Keyword extraction from natural language queries for FTS5 matching."""

from __future__ import annotations

import re

# English stop words (common, not exhaustive)
_STOP_WORDS: frozenset[str] = frozenset({
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "with",
  "by",
  "from",
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
  "this",
  "that",
  "these",
  "those",
  "it",
  "its",
  "i",
  "me",
  "my",
  "we",
  "our",
  "you",
  "your",
  "he",
  "she",
  "they",
  "them",
  "their",
  "what",
  "which",
  "who",
  "whom",
  "how",
  "when",
  "where",
  "why",
  "not",
  "no",
  "if",
  "then",
  "than",
  "so",
  "very",
  "just",
  "about",
  "also",
  "into",
  "out",
  "up",
  "down",
  "some",
  "any",
  "all",
  "each",
  "every",
  "both",
  "few",
  "more",
  "most",
  "other",
  "such",
  "only",
  "over",
  "under",
  "between",
  "through",
  "after",
  "before",
  "during",
  "again",
  "here",
  "there",
})

# Pattern to tokenize: keep alphanumeric + hyphens
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][\w-]*[a-zA-Z0-9]|[a-zA-Z0-9]+")


def extract_keywords(query: str, *, max_keywords: int = 10) -> list[str]:
  """Extract meaningful keywords from a natural language query.

  Removes stop words and returns lowercased tokens suitable for
  FTS5 match expressions.

  Args:
    query: The natural language query.
    max_keywords: Maximum number of keywords to extract.

  Returns:
    List of keyword strings.
  """
  tokens = _TOKEN_RE.findall(query.lower())
  keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
  return keywords[:max_keywords]


def keywords_to_fts5_query(keywords: list[str]) -> str:
  """Convert keyword list to an FTS5 match expression.

  Joins with OR for broad matching. Quotes terms that contain
  special characters.

  Args:
    keywords: List of keywords from extract_keywords().

  Returns:
    FTS5-compatible query string.
  """
  if not keywords:
    return ""
  # Quote terms with hyphens
  terms = [f'"{k}"' if "-" in k else k for k in keywords]
  return " OR ".join(terms)
