"""Google reader stub — placeholder for future Google-based document processing.

Intended to support Google's Document AI and Gemini vision
capabilities for document understanding. Not yet implemented.
"""

from typing import List, Set

from definable.media import File
from definable.readers.models import ReaderOutput


class GoogleReader:
  """Placeholder for Google-based document reader.

  Will support document processing via Google Document AI and
  Gemini vision capabilities when implemented.
  """

  def can_read(self, file: File) -> bool:
    return False

  def read(self, file: File) -> ReaderOutput:
    raise NotImplementedError("GoogleReader is not yet implemented")

  async def aread(self, file: File) -> ReaderOutput:
    raise NotImplementedError("GoogleReader is not yet implemented")

  def supported_mime_types(self) -> List[str]:
    return []

  def supported_extensions(self) -> Set[str]:
    return set()
