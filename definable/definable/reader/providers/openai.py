"""OpenAI reader stub — placeholder for future OpenAI-based document processing.

Intended to support OpenAI's file processing and vision capabilities
for document understanding. Not yet implemented.
"""

from typing import List, Set

from definable.media import File
from definable.reader.models import ReaderOutput


class OpenAIReader:
  """Placeholder for OpenAI-based document reader.

  Will support document processing via OpenAI's vision and
  file understanding capabilities when implemented.
  """

  def can_read(self, file: File) -> bool:
    return False

  def read(self, file: File) -> ReaderOutput:
    raise NotImplementedError("OpenAIReader is not yet implemented")

  async def aread(self, file: File) -> ReaderOutput:
    raise NotImplementedError("OpenAIReader is not yet implemented")

  def supported_mime_types(self) -> List[str]:
    return []

  def supported_extensions(self) -> Set[str]:
    return set()
