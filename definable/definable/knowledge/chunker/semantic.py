"""Semantic chunker — splits text on embedding-based similarity boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from definable.knowledge.chunker.base import Chunker
from definable.knowledge.document import Document
from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.knowledge.embedder.base import Embedder


@dataclass
class SemanticChunker(Chunker):
  """Split text at semantic boundaries using embedding similarity.

  Sentences that are semantically similar stay together. A new chunk
  starts when the cosine similarity between consecutive sentence
  groups drops below a threshold.

  Args:
      chunk_size: Target chunk size in characters (soft limit).
      chunk_overlap: Overlap between consecutive chunks (characters).
      embedder: Embedder to compute sentence embeddings. If None,
               falls back to sentence-length heuristic.
      similarity_threshold: Cosine similarity threshold (0.0-1.0).
               Below this value, a new chunk boundary is created.
               Default 0.5.
      sentence_window: Number of sentences to average for comparison.
               Default 1.
      min_sentences: Minimum sentences per chunk. Default 2.

  Example::

      from definable.embedder import OpenAIEmbedder

      chunker = SemanticChunker(
          embedder=OpenAIEmbedder(),
          similarity_threshold=0.4,
      )
      chunks = chunker.chunk(Document(content=long_text))
  """

  embedder: Optional["Embedder"] = None
  similarity_threshold: float = 0.5
  sentence_window: int = 1
  min_sentences: int = 2

  def chunk(self, document: Document) -> List[Document]:
    text = document.content
    if not text:
      return []

    sentences = self._split_sentences(text)
    if len(sentences) <= self.min_sentences:
      return [document]

    if self.embedder is not None:
      boundaries = self._find_semantic_boundaries(sentences)
    else:
      # Fallback: split by size only
      boundaries = self._find_size_boundaries(sentences)

    raw_chunks = self._build_chunks(sentences, boundaries)

    # Create Document objects
    parent_id = document.id or str(uuid4())
    result: List[Document] = []
    for i, chunk_text in enumerate(raw_chunks):
      chunk_text = chunk_text.strip()
      if not chunk_text:
        continue
      result.append(
        Document(
          content=chunk_text,
          name=f"{document.name}_chunk_{i}" if document.name else f"chunk_{i}",
          source=document.source,
          source_type=document.source_type,
          parent_id=parent_id,
          chunk_index=i,
          chunk_total=len(raw_chunks),
          meta_data={**document.meta_data, "chunk_index": i, "chunk_total": len(raw_chunks), "chunker": "semantic"},
        )
      )

    # Fix chunk_total after filtering empties
    for doc in result:
      doc.chunk_total = len(result)
      doc.meta_data["chunk_total"] = len(result)

    return result

  def _split_sentences(self, text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    import re

    # Split on sentence endings followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    return [s.strip() for s in parts if s.strip()]

  def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
    """Find chunk boundaries using embedding similarity."""
    assert self.embedder is not None

    # Get embeddings for all sentences
    try:
      embeddings: List[List[float]] = []
      for s in sentences:
        emb = self.embedder.get_embedding(s)
        embeddings.append(emb)
    except Exception as e:
      log_warning(f"SemanticChunker: embedding failed ({e}), falling back to size-based splitting")
      return self._find_size_boundaries(sentences)

    if not embeddings or not embeddings[0]:
      return self._find_size_boundaries(sentences)

    # Compute similarities between consecutive windows
    boundaries: List[int] = []
    window = self.sentence_window

    for i in range(window, len(sentences)):
      # Average embeddings for left and right windows
      left = self._average_embeddings(embeddings[max(0, i - window) : i])
      right = self._average_embeddings(embeddings[i : min(len(sentences), i + window)])

      sim = self._cosine_similarity(left, right)
      if sim < self.similarity_threshold:
        boundaries.append(i)

    return boundaries

  def _find_size_boundaries(self, sentences: List[str]) -> List[int]:
    """Fallback: split by size when no embedder available."""
    boundaries: List[int] = []
    current_size = 0
    for i, s in enumerate(sentences):
      current_size += len(s) + 1
      if current_size >= self.chunk_size and i > 0:
        boundaries.append(i)
        current_size = len(s)
    return boundaries

  def _build_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
    """Build chunk strings from sentences and boundary indices."""
    chunks: List[str] = []
    start = 0
    for b in boundaries:
      if b > start:
        chunk = " ".join(sentences[start:b])
        chunks.append(chunk)
        start = b
    # Last chunk
    if start < len(sentences):
      chunks.append(" ".join(sentences[start:]))

    # Enforce min_sentences: merge tiny chunks
    merged: List[str] = []
    for chunk in chunks:
      if merged and len(chunk.split(". ")) < self.min_sentences:
        merged[-1] = merged[-1] + " " + chunk
      else:
        merged.append(chunk)

    return merged

  @staticmethod
  def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Compute element-wise average of embedding vectors."""
    if not embeddings:
      return []
    n = len(embeddings)
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
      for j in range(dim):
        avg[j] += emb[j]
    return [x / n for x in avg]

  @staticmethod
  def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
      return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
      return 0.0
    return dot / (norm_a * norm_b)
