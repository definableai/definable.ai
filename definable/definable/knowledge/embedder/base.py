"""Base class for embedder implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Embedder(ABC):
  """Base class for embedder implementations."""

  dimensions: int = 1536
  batch_size: int = 100

  @abstractmethod
  def get_embedding(self, text: str) -> List[float]:
    """Get embedding vector for text."""
    ...

  @abstractmethod
  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Get embedding vector and usage statistics."""
    ...

  @abstractmethod
  async def async_get_embedding(self, text: str) -> List[float]:
    """Async get embedding vector."""
    ...

  @abstractmethod
  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Async get embedding vector and usage statistics."""
    ...
