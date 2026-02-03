from definable.knowledge.readers.base import Reader, ReaderConfig

__all__ = [
  "Reader",
  "ReaderConfig",
]


def __getattr__(name: str):
  if name == "TextReader":
    from definable.knowledge.readers.text import TextReader
    return TextReader
  if name == "PDFReader":
    from definable.knowledge.readers.pdf import PDFReader
    return PDFReader
  if name == "URLReader":
    from definable.knowledge.readers.url import URLReader
    return URLReader
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
