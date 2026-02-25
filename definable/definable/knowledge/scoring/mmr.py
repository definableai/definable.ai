"""Maximal Marginal Relevance (MMR) — diversity-aware re-ranking.

Prevents near-duplicate results by balancing relevance with diversity.
Higher lambda favors relevance; lower lambda favors diversity.

Usage::

    from definable.knowledge.scoring import MMRConfig, mmr_rerank

    config = MMRConfig(lambda_param=0.7)
    diverse = mmr_rerank(query_embedding, documents, config=config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from definable.knowledge.document import Document


@dataclass
class MMRConfig:
  """Configuration for MMR diversity ranking.

  Attributes:
    lambda_param: Balance between relevance (1.0) and diversity (0.0).
      Default 0.5 provides equal weight.
    enabled: Whether MMR is active.
  """

  lambda_param: float = 0.5
  enabled: bool = True


def _cosine_similarity(a: List[float], b: List[float]) -> float:
  """Compute cosine similarity between two vectors.

  Falls back to dot-product-based computation for efficiency.
  Returns 0.0 if either vector is zero-length.
  """
  if len(a) != len(b) or not a:
    return 0.0

  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(x * x for x in b))

  if norm_a == 0.0 or norm_b == 0.0:
    return 0.0
  return dot / (norm_a * norm_b)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
  """Compute Jaccard similarity between two texts (token-level).

  Used as fallback when embeddings are not available.
  """
  tokens_a = set(text_a.lower().split())
  tokens_b = set(text_b.lower().split())
  if not tokens_a and not tokens_b:
    return 0.0
  intersection = tokens_a & tokens_b
  union = tokens_a | tokens_b
  return len(intersection) / len(union) if union else 0.0


def mmr_rerank(
  query_embedding: Optional[List[float]],
  documents: List[Document],
  *,
  config: Optional[MMRConfig] = None,
  top_k: Optional[int] = None,
) -> List[Document]:
  """Re-rank documents using Maximal Marginal Relevance.

  MMR iteratively selects documents that maximize:
  ``MMR = lambda * relevance - (1 - lambda) * max_sim_to_selected``

  This balances relevance (similarity to query) with diversity
  (dissimilarity to already-selected documents).

  Args:
    query_embedding: The query's embedding vector. If None, falls back
      to Jaccard similarity on text.
    documents: Documents to re-rank. Should have ``embedding`` and/or
      ``reranking_score`` attributes.
    config: MMR configuration. Defaults to lambda=0.5.
    top_k: Maximum results to return. Defaults to all documents.

  Returns:
    Re-ranked documents in MMR selection order.
  """
  cfg = config or MMRConfig()
  if not cfg.enabled or len(documents) <= 1:
    return documents

  k = top_k or len(documents)
  lam = cfg.lambda_param

  # Pre-compute relevance scores
  use_embeddings = query_embedding is not None and all(doc.embedding is not None for doc in documents)

  relevance_scores: list[float] = []
  for doc in documents:
    if use_embeddings and query_embedding is not None and doc.embedding is not None:
      relevance_scores.append(_cosine_similarity(query_embedding, doc.embedding))
    elif doc.reranking_score is not None:
      relevance_scores.append(doc.reranking_score)
    else:
      relevance_scores.append(0.0)

  # Greedy MMR selection
  selected_indices: list[int] = []
  remaining = set(range(len(documents)))

  for _ in range(min(k, len(documents))):
    best_idx = -1
    best_score = -float("inf")

    for idx in remaining:
      # Relevance term
      rel = relevance_scores[idx]

      # Diversity term: max similarity to any already-selected doc
      max_sim = 0.0
      for sel_idx in selected_indices:
        idx_emb = documents[idx].embedding
        sel_emb = documents[sel_idx].embedding
        if use_embeddings and idx_emb is not None and sel_emb is not None:
          sim = _cosine_similarity(idx_emb, sel_emb)
        else:
          sim = _jaccard_similarity(documents[idx].content, documents[sel_idx].content)
        max_sim = max(max_sim, sim)

      # MMR score
      mmr_score = lam * rel - (1 - lam) * max_sim

      if mmr_score > best_score:
        best_score = mmr_score
        best_idx = idx

    if best_idx < 0:
      break

    selected_indices.append(best_idx)
    remaining.discard(best_idx)

  return [documents[i] for i in selected_indices]
