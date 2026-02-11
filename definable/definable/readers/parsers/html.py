"""HTML parser — extracts text from HTML documents using stdlib html.parser.

No optional dependencies required.
"""

from html.parser import HTMLParser as _StdlibHTMLParser
from typing import List, Set

from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser

# Tags whose content should be skipped
_SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}


class _TextExtractor(_StdlibHTMLParser):
  """Extracts visible text from HTML."""

  def __init__(self) -> None:
    super().__init__()
    self._parts: List[str] = []
    self._skip_depth: int = 0

  def handle_starttag(self, tag: str, attrs: list) -> None:
    if tag.lower() in _SKIP_TAGS:
      self._skip_depth += 1

  def handle_endtag(self, tag: str) -> None:
    if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
      self._skip_depth -= 1

  def handle_data(self, data: str) -> None:
    if self._skip_depth == 0:
      text = data.strip()
      if text:
        self._parts.append(text)

  def get_text(self) -> str:
    return "\n".join(self._parts)


class HTMLParser(BaseParser):
  """Parses HTML files by stripping tags and extracting visible text.

  Uses the stdlib html.parser — no external dependencies.
  """

  def supported_mime_types(self) -> List[str]:
    return ["text/html", "application/xhtml+xml"]

  def supported_extensions(self) -> Set[str]:
    return {".html", ".htm", ".xhtml"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    encoding = config.encoding if config else "utf-8"
    html_text = data.decode(encoding, errors="replace")
    extractor = _TextExtractor()
    extractor.feed(html_text)
    text = extractor.get_text()
    return [
      ContentBlock(
        content_type="text",
        content=text,
        mime_type="text/html",
      )
    ]
