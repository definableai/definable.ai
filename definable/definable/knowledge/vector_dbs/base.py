"""Base class for vector database implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from definable.knowledge.document import Document


@dataclass
class VectorDB(ABC):
  """Base class for vector database implementations."""

  collection_name: str = "default"
  dimensions: int = 1536

  @abstractmethod
  def add(self, documents: List[Document]) -> List[str]:
    """Add documents to the vector store. Returns list of IDs."""
    ...

  @abstractmethod
  async def aadd(self, documents: List[Document]) -> List[str]:
    """Async add documents to the vector store."""
    ...

  @abstractmethod
  def search(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Search for similar documents by embedding vector."""
    ...

  @abstractmethod
  async def asearch(
    self,
    query_embedding: List[float],
    top_k: int = 10,
    filter: Optional[Dict[str, Any]] = None,  # noqa: A002
  ) -> List[Document]:
    """Async search for similar documents."""
    ...

  @abstractmethod
  def delete(self, ids: List[str]) -> None:
    """Delete documents by IDs."""
    ...

  @abstractmethod
  async def adelete(self, ids: List[str]) -> None:
    """Async delete documents by IDs."""
    ...

  @abstractmethod
  def clear(self) -> None:
    """Clear all documents from the collection."""
    ...

  def count(self) -> int:
    """Return number of documents in collection."""
    return 0
