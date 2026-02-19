from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from definable.knowledge.document import Document
from definable.knowledge.embedder import Embedder
from definable.utils.log import log_warning
from definable.utils.string import generate_id


class VectorDB(ABC):
  """Base class for Vector Databases.

  Subclasses must implement all @abstractmethod methods.
  Convenience aliases (count, ainsert, asearch) delegate to
  the canonical names (get_count, async_insert, async_search).
  """

  # Subclasses should set this in __init__; default is None.
  embedder: Optional[Embedder] = None

  def __init__(self, *, doc_id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None):
    """Initialize base VectorDB.

    Args:
        doc_id: Optional custom ID. If not provided, an id will be generated.
        name: Optional name for the vector database.
        description: Optional description for the vector database.
    """
    if name is None:
      name = self.__class__.__name__

    self.name = name
    self.description = description
    # Last resort fallback to generate id from name if ID not specified
    self.id = doc_id or generate_id(name)

  @abstractmethod
  def create(self) -> None:
    raise NotImplementedError

  @abstractmethod
  async def async_create(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def name_exists(self, name: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def async_name_exists(self, name: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def id_exists(self, doc_id: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def content_hash_exists(self, content_hash: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
    raise NotImplementedError

  @abstractmethod
  async def async_insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
    raise NotImplementedError

  def upsert_available(self) -> bool:
    return False

  @abstractmethod
  def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
    raise NotImplementedError

  @abstractmethod
  async def async_upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
    raise NotImplementedError

  @abstractmethod
  def search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
    raise NotImplementedError

  @abstractmethod
  async def async_search(self, query: str, limit: int = 5, filters: Optional[Any] = None) -> List[Document]:
    raise NotImplementedError

  @abstractmethod
  def drop(self) -> None:
    raise NotImplementedError

  @abstractmethod
  async def async_drop(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def exists(self) -> bool:
    raise NotImplementedError

  @abstractmethod
  async def async_exists(self) -> bool:
    raise NotImplementedError

  def optimize(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def delete(self) -> bool:
    raise NotImplementedError

  @abstractmethod
  def delete_by_id(self, doc_id: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def delete_by_name(self, name: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
    raise NotImplementedError

  def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
    """
    Update the metadata for documents with the given content_id.

    Default implementation logs a warning. Subclasses should override this method
    to provide their specific implementation.

    Args:
        content_id (str): The content ID to update
        metadata (Dict[str, Any]): The metadata to update
    """
    log_warning(f"{self.__class__.__name__}.update_metadata() is not implemented. Metadata update for content_id '{content_id}' was skipped.")

  @abstractmethod
  def delete_by_content_id(self, content_id: str) -> bool:
    raise NotImplementedError

  @abstractmethod
  def get_supported_search_types(self) -> List[str]:
    raise NotImplementedError

  # ── Convenience aliases (used by Knowledge layer) ────────────────────

  def get_count(self) -> int:
    """Return the number of stored documents. Override in subclasses."""
    return 0

  def count(self) -> int:
    """Alias for get_count()."""
    return self.get_count()

  async def ainsert(
    self,
    content_hash_or_docs: "str | List[Document]",
    documents: Optional[List[Document]] = None,
    filters: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Alias for async_insert(). Accepts (content_hash, docs) or (docs)."""
    if isinstance(content_hash_or_docs, list):
      from hashlib import md5

      docs = content_hash_or_docs
      combined = "".join(doc.content for doc in docs)
      content_hash = md5(combined.encode()).hexdigest()
      await self.async_insert(content_hash, docs, filters)
    else:
      if documents is None:
        raise ValueError("documents must be provided when content_hash is given")
      await self.async_insert(content_hash_or_docs, documents, filters)

  async def asearch(
    self,
    query: str,
    limit: int = 5,
    filters: Optional[Any] = None,
  ) -> List[Document]:
    """Alias for async_search()."""
    return await self.async_search(query, limit, filters)
