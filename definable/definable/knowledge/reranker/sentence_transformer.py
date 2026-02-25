"""Sentence Transformer reranker — open-source cross-encoder reranking."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from definable.knowledge.document import Document
from definable.knowledge.reranker.base import Reranker
from definable.utils.log import log_error
from pydantic import ConfigDict


class SentenceTransformerReranker(Reranker):
  """Rerank documents using a sentence-transformers cross-encoder model.

  Uses the ``sentence-transformers`` library for local, open-source
  reranking with no API calls. The cross-encoder scores query-document
  pairs directly for high-quality relevance ranking.

  Args:
      model: Cross-encoder model name. Default "cross-encoder/ms-marco-MiniLM-L-6-v2".
      top_n: Maximum number of documents to return. None = return all.
      device: Device for inference ("cpu", "cuda", "mps"). Default None (auto).
      batch_size: Batch size for scoring. Default 32.

  Example::

      from definable.reranker import SentenceTransformerReranker
      reranker = SentenceTransformerReranker()
      reranked = reranker.rerank("quantum computing", documents)
  """

  model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_n: Optional[int] = None
  device: Optional[str] = None
  batch_size: int = 32
  _cross_encoder: Optional[Any] = None

  model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

  @property
  def cross_encoder(self) -> Any:
    """Lazy-load the cross-encoder model."""
    if self._cross_encoder is not None:
      return self._cross_encoder

    try:
      from sentence_transformers import CrossEncoder
    except ImportError:
      raise ImportError("`sentence-transformers` not installed. Run: pip install sentence-transformers")

    kwargs: Dict[str, Any] = {}
    if self.device:
      kwargs["device"] = self.device

    encoder = CrossEncoder(self.model, **kwargs)
    object.__setattr__(self, "_cross_encoder", encoder)
    return encoder

  def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Internal rerank using cross-encoder scores."""
    if not documents:
      return []

    # Build query-document pairs
    pairs = [[query, doc.content] for doc in documents]

    # Score pairs
    scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)

    # Assign scores to documents
    scored_docs: List[Document] = []
    for doc, score in zip(documents, scores):
      doc.reranking_score = float(score)
      scored_docs.append(doc)

    # Sort by score descending
    scored_docs.sort(key=lambda d: d.reranking_score if d.reranking_score is not None else float("-inf"), reverse=True)

    # Apply top_n limit
    if self.top_n is not None and self.top_n > 0:
      scored_docs = scored_docs[: self.top_n]

    return scored_docs

  def rerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Rerank documents by relevance to query."""
    try:
      return self._rerank(query=query, documents=documents)
    except Exception as e:
      log_error(f"SentenceTransformerReranker error: {e}. Returning original documents.")
      return documents

  async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Async rerank via thread pool (cross-encoder is CPU/GPU bound)."""
    try:
      loop = asyncio.get_event_loop()
      return await loop.run_in_executor(None, self._rerank, query, documents)
    except Exception as e:
      log_error(f"SentenceTransformerReranker async error: {e}. Returning original documents.")
      return documents
