"""ParserRegistry — format-to-parser mapping with priority-based dispatch.

Maps MIME types and extensions to parsers. Higher priority wins for the
same format. Built-in parsers register at priority 0, user-registered at 100.
"""

from typing import List, Optional

from definable.readers.parsers.base_parser import BaseParser
from definable.utils.log import log_debug


class ParserRegistry:
  """Central registry that maps formats to the appropriate parser.

  Priority-based: higher priority values win for the same format.
  Built-in parsers register at priority 0. User-registered parsers
  default to priority 100.

  Args:
    include_defaults: If True, auto-register built-in parsers with
      graceful fallback for missing optional deps.
  """

  def __init__(self, include_defaults: bool = True) -> None:
    self.include_defaults = include_defaults
    self._entries: List[tuple[BaseParser, int]] = []
    if include_defaults:
      self._register_defaults()

  def register(self, parser: BaseParser, priority: int = 100) -> "ParserRegistry":
    """Register a parser with the given priority.

    Returns self for method chaining.
    """
    self._entries.append((parser, priority))
    # Sort by priority descending so highest-priority parsers are checked first
    self._entries.sort(key=lambda e: e[1], reverse=True)
    return self

  def get_parser(
    self,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
  ) -> Optional[BaseParser]:
    """Return the highest-priority parser for the given format, or None."""
    for parser, _priority in self._entries:
      if parser.can_parse(mime_type, extension):
        return parser
    return None

  @property
  def parsers(self) -> List[BaseParser]:
    """Return all registered parsers in priority order."""
    return [parser for parser, _priority in self._entries]

  def _register_defaults(self) -> None:
    """Register built-in parsers at priority 0."""
    # Text parser — always available (no optional deps)
    from definable.readers.parsers.text import TextParser

    self._entries.append((TextParser(), 0))

    # PDF parser — optional dep: pypdf
    try:
      from definable.readers.parsers.pdf import PDFParser

      self._entries.append((PDFParser(), 0))
      log_debug("PDFParser registered (pypdf available)")
    except ImportError:
      log_debug("PDFParser not registered (pypdf not installed)")

    # DOCX parser — optional dep: python-docx
    try:
      from definable.readers.parsers.docx import DocxParser

      self._entries.append((DocxParser(), 0))
      log_debug("DocxParser registered (python-docx available)")
    except ImportError:
      log_debug("DocxParser not registered (python-docx not installed)")

    # PPTX parser — optional dep: python-pptx
    try:
      from definable.readers.parsers.pptx import PptxParser

      self._entries.append((PptxParser(), 0))
      log_debug("PptxParser registered (python-pptx available)")
    except ImportError:
      log_debug("PptxParser not registered (python-pptx not installed)")

    # XLSX parser — optional dep: openpyxl
    try:
      from definable.readers.parsers.xlsx import XlsxParser

      self._entries.append((XlsxParser(), 0))
      log_debug("XlsxParser registered (openpyxl available)")
    except ImportError:
      log_debug("XlsxParser not registered (openpyxl not installed)")

    # ODS parser — optional dep: odfpy
    try:
      from definable.readers.parsers.ods import OdsParser

      self._entries.append((OdsParser(), 0))
      log_debug("OdsParser registered (odfpy available)")
    except ImportError:
      log_debug("OdsParser not registered (odfpy not installed)")

    # RTF parser — optional dep: striprtf
    try:
      from definable.readers.parsers.rtf import RtfParser

      self._entries.append((RtfParser(), 0))
      log_debug("RtfParser registered (striprtf available)")
    except ImportError:
      log_debug("RtfParser not registered (striprtf not installed)")

    # HTML parser — stdlib, always available
    from definable.readers.parsers.html import HTMLParser

    self._entries.append((HTMLParser(), 0))
    log_debug("HTMLParser registered")

    # Image parser — passthrough, always available
    from definable.readers.parsers.image import ImageParser

    self._entries.append((ImageParser(), 0))
    log_debug("ImageParser registered")

    # Audio parser — passthrough, always available
    from definable.readers.parsers.audio import AudioParser

    self._entries.append((AudioParser(), 0))
    log_debug("AudioParser registered")

    # Sort by priority descending
    self._entries.sort(key=lambda e: e[1], reverse=True)
