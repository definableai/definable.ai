"""Base class for text chunkers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from definable.knowledge.document import Document


@dataclass
class Chunker(ABC):
  """Base class for text chunkers."""

  chunk_size: int = 1000
  chunk_overlap: int = 200

  @abstractmethod
  def chunk(self, document: Document) -> List[Document]:
    """Split document into chunks."""
    ...

  def chunk_many(self, documents: List[Document]) -> List[Document]:
    """Split multiple documents into chunks."""
    chunks: List[Document] = []
    for doc in documents:
      chunks.extend(self.chunk(doc))
    return chunks
