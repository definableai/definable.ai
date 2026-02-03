"""Text file reader implementation."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Union

from definable.knowledge.document import Document
from definable.knowledge.readers.base import Reader


@dataclass
class TextReader(Reader):
  """Reader for plain text files."""

  supported_extensions: Set[str] = field(default_factory=lambda: {".txt", ".text", ".md", ".rst", ".csv", ".log"})

  def __post_init__(self) -> None:
    super().__post_init__()

  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read a text file and return as Document."""
    path = Path(source)

    if not path.exists():
      raise FileNotFoundError(f"File not found: {path}")

    if self.config.max_file_size:
      file_size = path.stat().st_size
      if file_size > self.config.max_file_size:
        raise ValueError(f"File size {file_size} exceeds max {self.config.max_file_size}")

    content = path.read_text(encoding=self.config.encoding)

    return [
      Document(
        content=content,
        name=path.name,
        source=str(path.absolute()),
        source_type="file",
        size=len(content),
        meta_data={
          "file_type": path.suffix.lstrip("."),
          "file_size": path.stat().st_size,
          **self.config.metadata,
        },
      )
    ]

  async def aread(self, source: Union[str, Path]) -> List[Document]:
    """Async read a text file."""
    try:
      import aiofiles  # type: ignore[import-untyped]
    except ImportError:
      # Fallback to sync read in executor
      loop = asyncio.get_event_loop()
      return await loop.run_in_executor(None, self.read, source)

    path = Path(source)

    if not path.exists():
      raise FileNotFoundError(f"File not found: {path}")

    if self.config.max_file_size:
      file_size = path.stat().st_size
      if file_size > self.config.max_file_size:
        raise ValueError(f"File size {file_size} exceeds max {self.config.max_file_size}")

    async with aiofiles.open(path, encoding=self.config.encoding) as f:
      content = await f.read()

    return [
      Document(
        content=content,
        name=path.name,
        source=str(path.absolute()),
        source_type="file",
        size=len(content),
        meta_data={
          "file_type": path.suffix.lstrip("."),
          "file_size": path.stat().st_size,
          **self.config.metadata,
        },
      )
    ]

  def can_read(self, source: Union[str, Path]) -> bool:
    """Check if this reader can handle the source."""
    path = Path(source)
    return path.suffix.lower() in self.supported_extensions
