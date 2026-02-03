"""In-memory vector database for development and testing."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from definable.knowledge.document import Document
from definable.knowledge.vector_dbs.base import VectorDB


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
  """Calculate cosine similarity between two vectors."""
  dot_product = sum(a * b for a, b in zip(vec1, vec2))
  norm1 = math.sqrt(sum(a * a for a in vec1))
  norm2 = math.sqrt(sum(b * b for b in vec2))
  if norm1 == 0 or norm2 == 0:
    return 0.0
  return dot_product / (norm1 * norm2)


@dataclass
class InMemoryVectorDB(VectorDB):
  """In-memory vector store using pure Python for similarity search."""

  _documents: Dict[str, Document] = field(default_factory=dict)
  _embeddings: Dict[str, List[float]] = field(default_factory=dict)

  def add(self, documents: List[Document]) -> List[str]:
    """Add documents to the vector store."""
    ids: List[str] = []
    for doc in documents:
      doc_id = doc.id or str(uuid4())
      doc.id = doc_id
      self._documents[doc_id] = doc
      if doc.embedding:
        self._embeddings[doc_id] = doc.embedding
      ids.append(doc_id)
    return ids

  async def aadd(self, documents: List[Document]) -> List[str]:
    """Async add documents to the vector store."""
    return self.add(documents)

  def search(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Search for similar documents by embedding vector."""
    if not self._embeddings:
      return []

    scores: List[tuple[str, float]] = []

    for doc_id, embedding in self._embeddings.items():
      # Apply filter if provided
      if filter:
        doc = self._documents.get(doc_id)
        if doc and not self._match_filter(doc, filter):
          continue

      similarity = _cosine_similarity(query_embedding, embedding)
      scores.append((doc_id, similarity))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)

    results: List[Document] = []
    for doc_id, score in scores[:top_k]:
      doc = self._documents[doc_id]
      doc.reranking_score = float(score)
      results.append(doc)

    return results

  async def asearch(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Async search for similar documents."""
    return self.search(query_embedding, top_k, filter)

  def delete(self, ids: List[str]) -> None:
    """Delete documents by IDs."""
    for doc_id in ids:
      self._documents.pop(doc_id, None)
      self._embeddings.pop(doc_id, None)

  async def adelete(self, ids: List[str]) -> None:
    """Async delete documents by IDs."""
    self.delete(ids)

  def clear(self) -> None:
    """Clear all documents from the collection."""
    self._documents.clear()
    self._embeddings.clear()

  def count(self) -> int:
    """Return number of documents in collection."""
    return len(self._documents)

  def _match_filter(self, doc: Document, filters: Dict[str, Any]) -> bool:
    """Check if document matches the filter criteria."""
    for key, value in filters.items():
      doc_value = doc.meta_data.get(key)
      if doc_value != value:
        return False
    return True
