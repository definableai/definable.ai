"""Anthropic reader stub — placeholder for future Anthropic-based document processing.

Intended to support Anthropic's PDF and document understanding
capabilities. Not yet implemented.
"""

from typing import List, Set

from definable.media import File
from definable.readers.models import ReaderOutput


class AnthropicReader:
  """Placeholder for Anthropic-based document reader.

  Will support document processing via Anthropic's PDF understanding
  and vision capabilities when implemented.
  """

  def can_read(self, file: File) -> bool:
    return False

  def read(self, file: File) -> ReaderOutput:
    raise NotImplementedError("AnthropicReader is not yet implemented")

  async def aread(self, file: File) -> ReaderOutput:
    raise NotImplementedError("AnthropicReader is not yet implemented")

  def supported_mime_types(self) -> List[str]:
    return []

  def supported_extensions(self) -> Set[str]:
    return set()
