"""Retrieval engine — 5-layer cascade for memory recall.

Layer 1: Scratchpad (always retrieved)
Layer 2: Query analysis → plan
Layer 3: Binary signature pre-filter (broad shortlist)
Layer 4: Targeted index search (graph/tags/embedding)
Layer 5: Fusion + reranking
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

from definable.memory.cortex.retrieval.analyzer import QueryAnalyzer
from definable.memory.cortex.retrieval.result import RetrievalResult, ScoredMemory
from definable.utils.log import log_debug

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.index.graph import GraphIndex
  from definable.memory.cortex.index.signature import SignatureBuilder, SignatureIndex
  from definable.memory.cortex.index.tags import TagIndex
  from definable.memory.cortex.store import CortexStore


class RetrievalEngine:
  """5-layer cascade retrieval engine.

  Layers:
    1. Scratchpad — always-retrieved belief state
    2. Query analysis — classify and plan
    3. Binary signature search — fast broad filter
    4. Targeted index search — graph/tags/embedding
    5. Fusion — merge, deduplicate, score, rank
  """

  def __init__(
    self,
    store: "CortexStore",
    config: "CortexConfig",
    embedder: Optional["Embedder"] = None,
    signature_builder: Optional["SignatureBuilder"] = None,
    signature_index: Optional["SignatureIndex"] = None,
    graph_index: Optional["GraphIndex"] = None,
    tag_index: Optional["TagIndex"] = None,
  ):
    self._store = store
    self._config = config
    self._embedder = embedder
    self._sig_builder = signature_builder
    self._sig_index = signature_index
    self._graph_index = graph_index
    self._tag_index = tag_index
    self._analyzer = QueryAnalyzer()

  async def recall(
    self,
    query: str,
    session_id: str = "default",
    user_id: str = "default",
    top_k: Optional[int] = None,
  ) -> RetrievalResult:
    """Execute the 5-layer retrieval cascade.

    Args:
      query: The search query.
      session_id: Session scope.
      user_id: User scope.
      top_k: Max results. Defaults to config.retrieval_top_k.

    Returns:
      RetrievalResult with scored memories and scratchpad context.
    """
    k = top_k or self._config.retrieval_top_k

    # Layer 1: Scratchpad
    scratchpad = await self._store.get_scratchpad(session_id, user_id)
    scratchpad_context = scratchpad.format_for_prompt()

    # Layer 2: Query analysis
    plan = self._analyzer.analyze(query, top_k=k)

    # Layer 3: Binary signature pre-filter
    sig_candidates: Dict[str, float] = {}
    if plan.use_signatures and self._sig_builder and self._sig_index:
      query_sig = self._sig_builder.build(query)
      sig_results = await self._sig_index.search(query_sig, max_distance=384, limit=100)
      for record_id, distance in sig_results:
        # Convert Hamming distance to similarity score (0-1)
        max_dist = self._config.signature_dims
        sig_candidates[record_id] = 1.0 - (distance / max_dist)

    # Layer 4: Targeted index search
    graph_candidates: Dict[str, float] = {}
    tag_candidates: Dict[str, float] = {}
    embedding_candidates: Dict[str, float] = {}

    # Graph traversal for causal/entity queries
    if plan.use_graph and self._graph_index and plan.entities:
      for entity_query in plan.entities:
        # Find records matching entity, then traverse
        if self._tag_index:
          entity_records = await self._tag_index.search_prefix(entity_query.lower())
          for rid in entity_records[:5]:
            bfs_results = await self._graph_index.bfs(rid, max_hops=self._config.graph_max_hops)
            for neighbor_id, depth in bfs_results:
              score = 1.0 / (1.0 + depth)  # Decay by distance
              graph_candidates[neighbor_id] = max(graph_candidates.get(neighbor_id, 0), score)

    # Tag search
    if plan.use_tags and self._tag_index and plan.keywords:
      for kw in plan.keywords:
        tag_results = await self._tag_index.search_prefix(kw)
        for rid in tag_results:
          tag_candidates[rid] = tag_candidates.get(rid, 0) + 0.3

    # Fetch all records once — shared by embedding and keyword search
    all_records = await self._store.get_all_records(user_id=user_id, active_only=True)

    # Embedding similarity
    if self._embedder:
      try:
        query_embedding = await self._embedder.async_get_embedding(query)
        if query_embedding:
          for rec in all_records:
            if rec.embedding:
              sim = _cosine_similarity(query_embedding, rec.embedding)
              embedding_candidates[rec.record_id] = sim
      except Exception:
        pass

    # Keyword matching — always runs as a complementary signal to embeddings.
    # Uses stemmed unigrams + bigrams for better matching.
    keyword_candidates: Dict[str, float] = {}
    query_words = _extract_keywords(query)
    query_bigrams = _extract_bigrams(query)
    if query_words:
      for rec in all_records:
        rec_text_lower = rec.raw_content.lower()
        rec_words = _extract_keywords(rec_text_lower)
        rec_bigrams = _extract_bigrams(rec_text_lower)

        # Unigram overlap (stemmed)
        stemmed_query = {_STEM(w) for w in query_words}
        stemmed_rec = {_STEM(w) for w in rec_words}
        unigram_overlap = len(stemmed_query & stemmed_rec)

        # Bigram overlap (exact phrase matching bonus)
        bigram_overlap = len(query_bigrams & rec_bigrams)

        if unigram_overlap > 0:
          # Weighted: bigrams worth 2x unigrams
          score = (
            (unigram_overlap + bigram_overlap * 2) / (len(query_words) + len(query_bigrams) * 2)
            if (len(query_words) + len(query_bigrams) * 2) > 0
            else 0
          )
          keyword_candidates[rec.record_id] = min(score, 1.0)

    # Layer 5: Fusion — merge all candidate pools
    all_candidate_ids: set[str] = set()
    all_candidate_ids.update(sig_candidates.keys())
    all_candidate_ids.update(graph_candidates.keys())
    all_candidate_ids.update(tag_candidates.keys())
    all_candidate_ids.update(embedding_candidates.keys())
    all_candidate_ids.update(keyword_candidates.keys())

    # If no candidates from indexes, fall back to recent records
    if not all_candidate_ids:
      recent = await self._store.get_records(session_id, user_id, active_only=True, limit=k * 2)
      all_candidate_ids = {r.record_id for r in recent}

    # Score fusion (weighted combination — embedding + keyword are additive)
    fused_scores: Dict[str, float] = {}
    for rid in all_candidate_ids:
      score = 0.0
      if rid in embedding_candidates:
        score += embedding_candidates[rid] * 0.4
      if rid in keyword_candidates:
        score += keyword_candidates[rid] * 0.35
      if rid in sig_candidates:
        score += sig_candidates[rid] * 0.1
      if rid in graph_candidates:
        score += graph_candidates[rid] * 0.1
      if rid in tag_candidates:
        score += tag_candidates[rid] * 0.05
      fused_scores[rid] = score

    # Fetch actual records for top candidates
    sorted_ids = sorted(fused_scores.keys(), key=lambda rid: fused_scores[rid], reverse=True)[: k * 2]

    scored_memories: List[ScoredMemory] = []
    for rid in sorted_ids:
      record = await self._store.get_record(rid)
      if record and record.is_active:
        # Determine which layer contributed most
        source_layer = "fusion"
        if rid in embedding_candidates and embedding_candidates[rid] > 0.3:
          source_layer = "embedding"
        elif rid in graph_candidates:
          source_layer = "graph"
        elif rid in tag_candidates:
          source_layer = "tags"
        elif rid in sig_candidates:
          source_layer = "signature"

        scored_memories.append(
          ScoredMemory(
            record=record,
            score=fused_scores[rid],
            source_layer=source_layer,
          )
        )

    # Final ranking — take top_k
    scored_memories.sort(key=lambda sm: sm.score, reverse=True)
    scored_memories = scored_memories[:k]

    log_debug(
      f"Retrieval: query_type={plan.query_type.value}, candidates={len(all_candidate_ids)}, returned={len(scored_memories)}",
      log_level=2,
    )

    return RetrievalResult(
      query=query,
      memories=scored_memories,
      scratchpad_context=scratchpad_context,
      total_candidates=len(all_candidate_ids),
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  """Compute cosine similarity between two vectors."""
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))
  if norm_a == 0.0 or norm_b == 0.0:
    return 0.0
  return dot / (norm_a * norm_b)


_STOPWORDS = frozenset({
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
  "to",
  "of",
  "in",
  "for",
  "on",
  "with",
  "at",
  "by",
  "from",
  "as",
  "into",
  "about",
  "between",
  "through",
  "and",
  "but",
  "or",
  "not",
  "no",
  "so",
  "if",
  "than",
  "that",
  "this",
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
  "where",
  "when",
  "how",
  "why",
  "all",
  "any",
  "some",
})


def _extract_keywords(text: str) -> set[str]:
  """Extract meaningful keywords from text, filtering stopwords."""
  words = set(text.lower().split())
  return words - _STOPWORDS


def _extract_bigrams(text: str) -> set[str]:
  """Extract bigrams from text (after stopword removal)."""
  words = [w for w in text.lower().split() if w not in _STOPWORDS]
  return {f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)} if len(words) >= 2 else set()


try:
  from nltk.stem import PorterStemmer as _PS

  _STEMMER = _PS()

  def _STEM(w: str) -> str:
    return _STEMMER.stem(w)

except ImportError:

  def _STEM(w: str) -> str:
    return w
