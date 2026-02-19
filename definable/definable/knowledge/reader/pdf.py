"""PDF file reader implementation."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from definable.knowledge.document import Document
from definable.knowledge.reader.base import Reader


@dataclass
class PDFReader(Reader):
  """Reader for PDF files. Requires pypdf."""

  extract_images: bool = False
  page_separator: str = "\n\n---\n\n"

  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read a PDF file and return as Document."""
    assert self.config is not None
    try:
      from pypdf import PdfReader as PyPDFReader
    except ImportError:
      raise ImportError("pypdf not installed. Run: pip install pypdf")

    path = Path(source)

    if not path.exists():
      raise FileNotFoundError(f"File not found: {path}")

    reader = PyPDFReader(path)
    pages: List[str] = []

    for page in reader.pages:
      text = page.extract_text()
      if text:
        pages.append(text)

    content = self.page_separator.join(pages)

    return [
      Document(
        content=content,
        name=path.name,
        source=str(path.absolute()),
        source_type="file",
        size=len(content),
        meta_data={
          "file_type": "pdf",
          "page_count": len(pages),
          "file_size": path.stat().st_size,
          **self.config.metadata,
        },
      )
    ]

  async def aread(self, source: Union[str, Path]) -> List[Document]:
    """Async read a PDF file."""
    # pypdf doesn't have async support, use executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.read, source)

  def can_read(self, source: Union[str, Path]) -> bool:
    """Check if this reader can handle the source."""
    path = Path(source)
    return path.suffix.lower() == ".pdf"
