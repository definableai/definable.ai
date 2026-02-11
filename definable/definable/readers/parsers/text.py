"""Text file parser — handles plain text and code files.

No optional dependencies required.
"""

from typing import List, Optional, Set

from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser

_DEFAULT_TEXT_EXTENSIONS: Set[str] = {
  ".txt",
  ".md",
  ".csv",
  ".json",
  ".xml",
  ".html",
  ".htm",
  ".log",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".rst",
  # Code files
  ".py",
  ".js",
  ".ts",
  ".jsx",
  ".tsx",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".cs",
  ".go",
  ".rs",
  ".rb",
  ".php",
  ".swift",
  ".kt",
  ".scala",
  ".r",
  ".sql",
  ".sh",
  ".bash",
  ".zsh",
  ".ps1",
  ".bat",
  ".cmd",
  ".lua",
  ".perl",
  ".pl",
  ".css",
  ".scss",
  ".sass",
  ".less",
  ".vue",
  ".svelte",
}

_DEFAULT_TEXT_MIME_TYPES: List[str] = [
  "text/plain",
  "text/html",
  "text/css",
  "text/csv",
  "text/xml",
  "text/markdown",
  "text/x-python",
  "text/javascript",
  "text/md",
  "application/json",
  "application/xml",
  "application/x-javascript",
  "application/x-python",
  "application/x-yaml",
]


class TextParser(BaseParser):
  """Parses text-based files by decoding bytes to string.

  Handles plain text, markdown, CSV, JSON, XML, HTML, code files, and more.
  """

  def supported_mime_types(self) -> List[str]:
    return list(_DEFAULT_TEXT_MIME_TYPES)

  def supported_extensions(self) -> Set[str]:
    return set(_DEFAULT_TEXT_EXTENSIONS)

  def can_parse(self, mime_type: Optional[str] = None, extension: Optional[str] = None) -> bool:
    # Also match any text/* MIME type not explicitly listed
    if mime_type and mime_type.startswith("text/"):
      return True
    return super().can_parse(mime_type, extension)

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    encoding = config.encoding if config else "utf-8"
    text = data.decode(encoding, errors="replace")
    return [
      ContentBlock(
        content_type="text",
        content=text,
        mime_type=mime_type or "text/plain",
      )
    ]
