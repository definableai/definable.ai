"""Abstract base class for format parsers.

Parsers are sync-only, stateless functions: bytes → list[ContentBlock].
They never do I/O. The async boundary lives in BaseReader.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set

from definable.reader.models import ContentBlock, ReaderConfig


class BaseParser(ABC):
  """Base class for all format parsers."""

  @abstractmethod
  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    """Parse raw bytes into content blocks."""
    ...

  @abstractmethod
  def supported_mime_types(self) -> List[str]:
    """Return the MIME types this parser can handle."""
    ...

  @abstractmethod
  def supported_extensions(self) -> Set[str]:
    """Return the file extensions this parser can handle (e.g., {'.pdf'})."""
    ...

  def can_parse(self, mime_type: Optional[str] = None, extension: Optional[str] = None) -> bool:
    """Check whether this parser can handle the given format.

    Matches by MIME type first, then falls back to file extension.
    """
    if mime_type and mime_type in self.supported_mime_types():
      return True
    if extension and extension in self.supported_extensions():
      return True
    return False
