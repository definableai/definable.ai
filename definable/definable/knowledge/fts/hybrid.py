"""Hybrid search — merge vector and full-text search results.

Combines vector similarity scores with BM25 keyword scores using
configurable weights and Reciprocal Rank Fusion (RRF).

Usage::

    from definable.knowledge.fts import HybridSearcher, HybridSearchConfig, FTSIndex

    fts = FTSIndex()
    await fts.initialize()

    searcher = HybridSearcher(fts_index=fts)
    merged = await searcher.merge(vector_results, query="machine learning", limit=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from definable.knowledge.document import Document
from definable.knowledge.fts.index import FTSIndex
from definable.utils.log import log_debug


@dataclass
class HybridSearchConfig:
  """Configuration for hybrid search merging.

  Attributes:
    vector_weight: Weight for vector similarity scores. Default 0.6.
    text_weight: Weight for BM25 text search scores. Default 0.4.
    merge_strategy: ``"rrf"`` (Reciprocal Rank Fusion) or ``"weighted"``
      (direct score combination). RRF is more robust to score scale
      differences.
    rrf_k: RRF smoothing constant. Higher values reduce the impact
      of rank position. Default 60.
    fts_fetch_multiplier: Fetch this many times ``limit`` from FTS
      to ensure good coverage for merging.
  """

  vector_weight: float = 0.6
  text_weight: float = 0.4
  merge_strategy: Literal["rrf", "weighted"] = "rrf"
  rrf_k: int = 60
  fts_fetch_multiplier: int = 3


@dataclass
class HybridSearcher:
  """Merges vector search results with FTS keyword results.

  Given a list of vector-retrieved documents and a query string,
  performs FTS search and merges both result sets.
  """

  fts_index: FTSIndex
  config: HybridSearchConfig = field(default_factory=HybridSearchConfig)

  async def merge(
    self,
    vector_results: List[Document],
    query: str,
    limit: int = 10,
  ) -> List[Document]:
    """Merge vector and FTS results.

    Args:
      vector_results: Documents from vector similarity search.
      query: Original search query for FTS matching.
      limit: Maximum results to return.

    Returns:
      Merged documents sorted by combined score.
    """
    # Fetch FTS results
    fts_limit = limit * self.config.fts_fetch_multiplier
    fts_docs = await self.fts_index.search_documents(query, limit=fts_limit)

    if not fts_docs:
      return vector_results[:limit]

    if not vector_results:
      return fts_docs[:limit]

    if self.config.merge_strategy == "rrf":
      return self._merge_rrf(vector_results, fts_docs, limit)
    return self._merge_weighted(vector_results, fts_docs, limit)

  def _merge_rrf(
    self,
    vector_docs: List[Document],
    fts_docs: List[Document],
    limit: int,
  ) -> List[Document]:
    """Merge using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across retrieval sources.
    """
    k = self.config.rrf_k
    vw = self.config.vector_weight
    tw = self.config.text_weight

    # Build score map keyed by content (since doc IDs may differ)
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Vector results — rank by position
    for rank, doc in enumerate(vector_docs):
      key = doc.content[:200]  # Use content prefix as dedup key
      scores[key] = scores.get(key, 0.0) + vw / (k + rank + 1)
      doc_map[key] = doc

    # FTS results — rank by position
    for rank, doc in enumerate(fts_docs):
      key = doc.content[:200]
      scores[key] = scores.get(key, 0.0) + tw / (k + rank + 1)
      if key not in doc_map:
        doc_map[key] = doc

    # Sort by combined score
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    result = []
    for key in sorted_keys[:limit]:
      doc = doc_map[key]
      doc.reranking_score = scores[key]
      result.append(doc)

    log_debug(f"Hybrid RRF merge: {len(vector_docs)} vector + {len(fts_docs)} FTS → {len(result)} merged")
    return result

  def _merge_weighted(
    self,
    vector_docs: List[Document],
    fts_docs: List[Document],
    limit: int,
  ) -> List[Document]:
    """Merge using weighted score combination.

    Normalizes both score sets to [0, 1] then combines with weights.
    """
    vw = self.config.vector_weight
    tw = self.config.text_weight

    # Normalize vector scores
    v_scores = [doc.reranking_score or 0.0 for doc in vector_docs]
    v_max = max(v_scores) if v_scores else 1.0
    v_min = min(v_scores) if v_scores else 0.0
    v_range = v_max - v_min or 1.0

    # Normalize FTS scores
    f_scores = [doc.reranking_score or 0.0 for doc in fts_docs]
    f_max = max(f_scores) if f_scores else 1.0
    f_min = min(f_scores) if f_scores else 0.0
    f_range = f_max - f_min or 1.0

    # Build score map
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc, raw_score in zip(vector_docs, v_scores):
      key = doc.content[:200]
      normalized = (raw_score - v_min) / v_range
      scores[key] = scores.get(key, 0.0) + vw * normalized
      doc_map[key] = doc

    for doc, raw_score in zip(fts_docs, f_scores):
      key = doc.content[:200]
      normalized = (raw_score - f_min) / f_range
      scores[key] = scores.get(key, 0.0) + tw * normalized
      if key not in doc_map:
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    result = []
    for key in sorted_keys[:limit]:
      doc = doc_map[key]
      doc.reranking_score = scores[key]
      result.append(doc)

    log_debug(f"Hybrid weighted merge: {len(vector_docs)} vector + {len(fts_docs)} FTS → {len(result)} merged")
    return result
