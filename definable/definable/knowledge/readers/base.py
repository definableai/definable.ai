"""Base class for document readers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from definable.knowledge.document import Document


@dataclass
class ReaderConfig:
  """Configuration for reader behavior."""

  encoding: str = "utf-8"
  max_file_size: Optional[int] = None  # bytes
  timeout: Optional[float] = 30.0  # seconds for URL-based readers
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reader(ABC):
  """Base class for document readers."""

  config: Optional[ReaderConfig] = None

  def __post_init__(self) -> None:
    if self.config is None:
      self.config = ReaderConfig()

  @abstractmethod
  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read source and return list of documents."""
    ...

  @abstractmethod
  async def aread(self, source: Union[str, Path]) -> List[Document]:
    """Async read source and return list of documents."""
    ...

  def can_read(self, source: Union[str, Path]) -> bool:
    """Check if this reader can handle the source."""
    return False
