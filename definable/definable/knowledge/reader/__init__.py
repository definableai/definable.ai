from definable.knowledge.reader.base import Reader, ReaderConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.reader.pdf import PDFReader
  from definable.knowledge.reader.text import TextReader
  from definable.knowledge.reader.url import URLReader

__all__ = [
  "Reader",
  "ReaderConfig",
  # Implementations (lazy-loaded)
  "PDFReader",
  "TextReader",
  "URLReader",
]


def __getattr__(name: str):
  if name == "TextReader":
    from definable.knowledge.reader.text import TextReader

    return TextReader
  if name == "PDFReader":
    from definable.knowledge.reader.pdf import PDFReader

    return PDFReader
  if name == "URLReader":
    from definable.knowledge.reader.url import URLReader

    return URLReader
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
