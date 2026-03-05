"""Query analyzer — classifies queries and builds retrieval plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class QueryType(str, Enum):
  """Classification of query intent."""

  FACTUAL = "factual"  # What is X? / Tell me about Y
  TEMPORAL = "temporal"  # When did X happen?
  CAUSAL = "causal"  # Why did X happen?
  ENTITY = "entity"  # Who/What is related to X?
  PREFERENCE = "preference"  # What do I prefer/like?
  RECALL = "recall"  # What did we discuss about X?
  GENERAL = "general"  # Catch-all


@dataclass
class QueryPlan:
  """Retrieval plan produced by the analyzer."""

  query_type: QueryType = QueryType.GENERAL
  entities: List[str] = field(default_factory=list)
  keywords: List[str] = field(default_factory=list)
  time_references: List[str] = field(default_factory=list)
  use_graph: bool = False
  use_signatures: bool = True
  use_tags: bool = True
  top_k: int = 10


# Rule-based classification patterns
_WHY_RE = re.compile(r"\b(?:why|because|reason|cause)\b", re.IGNORECASE)
_WHEN_RE = re.compile(r"\b(?:when|date|time|last\s+time|recently|ago)\b", re.IGNORECASE)
_WHO_RE = re.compile(r"\b(?:who|person|people|team|member)\b", re.IGNORECASE)
_WHAT_RE = re.compile(r"\b(?:what\s+(?:is|are|was|were)|define|explain|describe)\b", re.IGNORECASE)
_PREF_RE = re.compile(r"\b(?:prefer|like|favorite|style|approach|way\s+I)\b", re.IGNORECASE)
_RECALL_RE = re.compile(r"\b(?:discuss|talk|mention|said|told|remember)\b", re.IGNORECASE)


class QueryAnalyzer:
  """Analyzes queries to produce retrieval plans.

  Uses rule-based classification by default, with optional LLM upgrade
  for ambiguous queries.
  """

  def analyze(self, query: str, top_k: int = 10) -> QueryPlan:
    """Analyze a query and produce a retrieval plan."""
    query_type = self._classify(query)
    entities = self._extract_entities(query)
    keywords = self._extract_keywords(query)
    time_refs = self._extract_time_references(query)

    plan = QueryPlan(
      query_type=query_type,
      entities=entities,
      keywords=keywords,
      time_references=time_refs,
      top_k=top_k,
    )

    # Tune plan based on query type
    if query_type == QueryType.CAUSAL:
      plan.use_graph = True
    elif query_type == QueryType.ENTITY:
      plan.use_graph = True
    elif query_type == QueryType.TEMPORAL:
      plan.use_signatures = False  # Signatures don't encode time well

    return plan

  def _classify(self, query: str) -> QueryType:
    """Classify query intent using regex patterns."""
    if _WHY_RE.search(query):
      return QueryType.CAUSAL
    if _WHEN_RE.search(query):
      return QueryType.TEMPORAL
    if _WHO_RE.search(query):
      return QueryType.ENTITY
    if _PREF_RE.search(query):
      return QueryType.PREFERENCE
    if _RECALL_RE.search(query):
      return QueryType.RECALL
    if _WHAT_RE.search(query):
      return QueryType.FACTUAL
    return QueryType.GENERAL

  def _extract_entities(self, query: str) -> List[str]:
    """Extract potential entity names from query."""
    # CamelCase words
    camel = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", query)
    # Quoted strings
    quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
    return list(dict.fromkeys(camel + quoted))

  def _extract_keywords(self, query: str) -> List[str]:
    """Extract significant keywords (non-stopword tokens)."""
    stopwords = {
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
      "shall",
      "should",
      "may",
      "might",
      "must",
      "can",
      "could",
      "i",
      "me",
      "my",
      "we",
      "our",
      "you",
      "your",
      "he",
      "she",
      "it",
      "they",
      "them",
      "this",
      "that",
      "what",
      "which",
      "who",
      "when",
      "where",
      "why",
      "how",
      "about",
      "of",
      "in",
      "on",
      "at",
      "to",
      "for",
      "with",
      "from",
      "by",
      "and",
      "or",
      "not",
      "but",
      "if",
      "then",
      "so",
      "just",
      "also",
      "very",
    }
    tokens = re.findall(r"\b\w+\b", query.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 2]

  def _extract_time_references(self, query: str) -> List[str]:
    """Extract time references from query."""
    time_patterns = re.findall(
      r"\b(?:yesterday|today|last\s+\w+|this\s+\w+|\d+\s+(?:days?|weeks?|months?)\s+ago)\b",
      query,
      re.IGNORECASE,
    )
    return time_patterns
