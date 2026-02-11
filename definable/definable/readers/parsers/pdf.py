"""PDF parser — extracts text from PDFs using pypdf.

Requires optional dependency: ``pypdf>=4.0.0``
"""

import io
from typing import List, Set

from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser


def _import_pypdf():
  try:
    import pypdf

    return pypdf
  except ImportError as e:
    raise ImportError(
      "PDFParser requires the 'pypdf' package. Install it with: pip install 'pypdf>=4.0.0' or: pip install 'definable-ai[readers]'"
    ) from e


class PDFParser(BaseParser):
  """Parses PDF files using pypdf (local, no API calls).

  Extracts text from all pages, one ContentBlock per page.
  """

  def __init__(self, page_separator: str = "\n\n") -> None:
    _import_pypdf()
    self.page_separator = page_separator

  def supported_mime_types(self) -> List[str]:
    return ["application/pdf"]

  def supported_extensions(self) -> Set[str]:
    return {".pdf"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    pypdf = _import_pypdf()
    reader = pypdf.PdfReader(io.BytesIO(data))
    blocks: List[ContentBlock] = []
    for i, page in enumerate(reader.pages):
      text = page.extract_text() or ""
      if text.strip():
        blocks.append(
          ContentBlock(
            content_type="text",
            content=text,
            mime_type="application/pdf",
            page_number=i + 1,
          )
        )
    return blocks
