"""In-memory vector database implementation.

Simple dict-based storage with numpy cosine similarity search.
No external dependencies beyond numpy.
"""

import asyncio
from hashlib import md5
from typing import Any, Dict, List, Optional, Union

import numpy as np

from definable.knowledge.document import Document
from definable.knowledge.embedder import Embedder
from definable.utils.log import log_debug, log_info, log_warning
from definable.vectordb.base import VectorDB
from definable.vectordb.distance import Distance


class InMemoryVectorDB(VectorDB):
  """Pure-Python in-memory vector database using numpy cosine similarity.

  Stores documents in a dict keyed by document ID. Supports vector search
  via cosine similarity (default), L2 distance, or max inner product.

  The embedder is used for **search query embedding only**. Documents
  should arrive pre-embedded from Knowledge or the caller.

  Args:
      name: Name for this vector database instance.
      description: Optional description.
      doc_id: Optional custom ID.
      embedder: Embedder for embedding search queries. Defaults to OpenAIEmbedder.
      distance: Distance metric for similarity search.
  """

  def __init__(
    self,
    name: Optional[str] = None,
    description: Optional[str] = None,
    doc_id: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    distance: Distance = Distance.cosine,
    # Backward compat: accept but ignore dimensions kwarg
    dimensions: Optional[int] = None,
    **kwargs,
  ):
    super().__init__(doc_id=doc_id, name=name, description=description)

    # Embedder for embedding search queries
    if embedder is None:
      from definable.knowledge.embedder.openai import OpenAIEmbedder

      embedder = OpenAIEmbedder()
      log_debug("Embedder not provided, using OpenAIEmbedder as default.")
    self.embedder: Embedder = embedder

    self.distance: Distance = distance

    # Storage: doc_id -> record dict
    self._store: Dict[str, Dict[str, Any]] = {}

    if dimensions is not None:
      log_warning("InMemoryVectorDB: 'dimensions' kwarg is deprecated and ignored.")

  def create(self) -> None:
    """No-op for in-memory store."""
    log_debug("InMemoryVectorDB.create() — no-op")

  async def async_create(self) -> None:
    """No-op for in-memory store."""
    log_debug("InMemoryVectorDB.async_create() — no-op")

  def name_exists(self, name: str) -> bool:
    """Check if a document with the given name exists."""
    return any(r.get("name") == name for r in self._store.values())

  async def async_name_exists(self, name: str) -> bool:  # type: ignore[override]
    return self.name_exists(name)

  def id_exists(self, doc_id: str) -> bool:
    return doc_id in self._store

  def content_hash_exists(self, content_hash: str) -> bool:
    return any(r.get("content_hash") == content_hash for r in self._store.values())

  def insert(
    self,
    content_hash_or_docs: Union[str, List[Document]],
    documents: Optional[List[Document]] = None,
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Insert pre-embedded documents into the store.

    Supports two calling conventions:
      insert(content_hash, documents)
      insert(documents)  # auto-generates content_hash
    """
    direct_insert = False
    if isinstance(content_hash_or_docs, list):
      # Called as insert(documents) — auto-generate content_hash, preserve doc IDs
      documents = content_hash_or_docs
      combined = "".join(doc.content for doc in documents)
      content_hash = md5(combined.encode()).hexdigest()
      direct_insert = True
    else:
      content_hash = content_hash_or_docs
      if documents is None:
        raise ValueError("documents must be provided when content_hash is given")

    for doc in documents:
      cleaned_content = doc.content.replace("\x00", "\ufffd")
      if direct_insert and doc.id:
        # Preserve user-provided IDs for direct insert
        doc_id = doc.id
      else:
        base_id = doc.id or md5(cleaned_content.encode()).hexdigest()
        doc_id = md5(f"{base_id}_{content_hash}".encode()).hexdigest()

      meta_data = doc.meta_data.copy() if doc.meta_data else {}
      if filters:
        meta_data.update(filters)

      self._store[doc_id] = {
        "id": doc_id,
        "name": doc.name,
        "content": cleaned_content,
        "embedding": doc.embedding,
        "meta_data": meta_data,
        "usage": doc.usage,
        "content_hash": content_hash,
        "content_id": doc.content_id,
      }
    log_info(f"Inserted {len(documents)} documents")

  async def async_insert(
    self,
    content_hash_or_docs: Union[str, List[Document]],
    documents: Optional[List[Document]] = None,
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.insert(content_hash_or_docs, documents, filters)

  async def ainsert(
    self,
    content_hash_or_docs: Union[str, List[Document]],
    documents: Optional[List[Document]] = None,
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Override base ainsert to preserve direct-insert semantics."""
    self.insert(content_hash_or_docs, documents, filters)

  def upsert_available(self) -> bool:
    return True

  def upsert(
    self,
    content_hash: str,
    documents: List[Document],
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Upsert by deleting existing content_hash docs, then inserting."""
    if self.content_hash_exists(content_hash):
      self._delete_by_content_hash(content_hash)
    self.insert(content_hash, documents, filters)

  async def async_upsert(
    self,
    content_hash: str,
    documents: List[Document],
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.upsert(content_hash, documents, filters)

  def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
      return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

  def _l2_distance(self, a: List[float], b: List[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a_arr - b_arr))

  def _inner_product(self, a: List[float], b: List[float]) -> float:
    """Compute inner product between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.dot(a_arr, b_arr))

  def search(
    self,
    query: str,
    limit: int = 5,
    filters: Optional[Union[Dict[str, Any], list]] = None,
  ) -> List[Document]:
    """Search for documents by embedding the query and computing similarity."""
    query_embedding = self.embedder.get_embedding(query)
    if query_embedding is None:
      log_warning(f"Failed to get embedding for query: {query}")  # type: ignore[unreachable]
      return []

    # Filter candidates that have embeddings
    candidates = []
    for record in self._store.values():
      if record.get("embedding") is None:
        continue

      # Apply dict filters
      if filters and isinstance(filters, dict):
        meta = record.get("meta_data", {})
        if not all(meta.get(k) == v for k, v in filters.items()):
          continue

      candidates.append(record)

    if not candidates:
      return []

    # Score each candidate
    scored = []
    for record in candidates:
      embedding = record["embedding"]
      if self.distance == Distance.cosine:
        score = self._cosine_similarity(query_embedding, embedding)
      elif self.distance == Distance.l2:
        # Lower distance = better, so negate for sorting
        score = -self._l2_distance(query_embedding, embedding)
      elif self.distance == Distance.max_inner_product:
        score = self._inner_product(query_embedding, embedding)
      else:
        score = self._cosine_similarity(query_embedding, embedding)  # type: ignore[unreachable]
      scored.append((record, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build results
    results = []
    for record, _ in scored[:limit]:
      results.append(
        Document(
          id=record["id"],
          name=record.get("name"),
          content=record.get("content", ""),
          meta_data=record.get("meta_data", {}),
          embedding=record.get("embedding"),
          usage=record.get("usage"),
          content_id=record.get("content_id"),
        )
      )

    log_info(f"Found {len(results)} documents")
    return results

  async def async_search(
    self,
    query: str,
    limit: int = 5,
    filters: Optional[Union[Dict[str, Any], list]] = None,
  ) -> List[Document]:
    """Async search — delegates to sync since everything is in-memory."""
    return await asyncio.to_thread(self.search, query, limit, filters)

  def drop(self) -> None:
    """Clear all stored documents."""
    self._store.clear()
    log_info("InMemoryVectorDB dropped (all data cleared)")

  async def async_drop(self) -> None:
    self.drop()

  def exists(self) -> bool:
    """Always returns True for in-memory store."""
    return True

  async def async_exists(self) -> bool:
    return True

  def get_count(self) -> int:
    """Get the number of stored documents."""
    return len(self._store)

  def optimize(self) -> None:
    """No-op for in-memory store."""
    pass

  def delete(self) -> bool:
    """Delete all documents."""
    self._store.clear()
    return True

  def delete_by_id(self, doc_id: str) -> bool:
    if doc_id in self._store:
      del self._store[doc_id]
      return True
    return False

  def delete_by_name(self, name: str) -> bool:
    ids_to_delete = [k for k, v in self._store.items() if v.get("name") == name]
    for id_ in ids_to_delete:
      del self._store[id_]
    return len(ids_to_delete) > 0

  def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
    ids_to_delete = []
    for k, v in self._store.items():
      meta = v.get("meta_data", {})
      if all(meta.get(mk) == mv for mk, mv in metadata.items()):
        ids_to_delete.append(k)
    for id_ in ids_to_delete:
      del self._store[id_]
    return len(ids_to_delete) > 0

  def delete_by_content_id(self, content_id: str) -> bool:
    ids_to_delete = [k for k, v in self._store.items() if v.get("content_id") == content_id]
    for id_ in ids_to_delete:
      del self._store[id_]
    return len(ids_to_delete) > 0

  def _delete_by_content_hash(self, content_hash: str) -> bool:
    ids_to_delete = [k for k, v in self._store.items() if v.get("content_hash") == content_hash]
    for id_ in ids_to_delete:
      del self._store[id_]
    return len(ids_to_delete) > 0

  def get_supported_search_types(self) -> List[str]:
    return ["vector"]

  # Backward compat alias
  def add(self, documents: List[Document]) -> None:
    """Deprecated: use insert() instead."""
    log_warning("InMemoryVectorDB.add() is deprecated; use insert() instead.")
    from definable.utils.string import generate_id

    self.insert(generate_id(), documents)
