"""AI-provider-backed readers that handle their own API I/O.

Provider readers implement the ProviderReader protocol and can be
registered with BaseReader for format-specific AI processing.
"""

from typing import List, Protocol, Set, runtime_checkable

from definable.media import File
from definable.readers.models import ReaderOutput


@runtime_checkable
class ProviderReader(Protocol):
  """Protocol for AI-provider readers that handle their own I/O."""

  def can_read(self, file: File) -> bool: ...

  def read(self, file: File) -> ReaderOutput: ...

  async def aread(self, file: File) -> ReaderOutput: ...

  def supported_mime_types(self) -> List[str]: ...

  def supported_extensions(self) -> Set[str]: ...


__all__ = ["ProviderReader"]
