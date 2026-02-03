"""Base class for reranker implementations."""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, ConfigDict

from definable.knowledge.document import Document


class Reranker(BaseModel, ABC):
  """Base class for reranker implementations."""

  model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

  @abstractmethod
  def rerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Synchronously rerank documents by relevance to query."""
    ...

  @abstractmethod
  async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
    """Asynchronously rerank documents by relevance to query."""
    ...
