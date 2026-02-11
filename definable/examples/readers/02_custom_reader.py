"""Custom parser example — Create a custom format parser.

Shows how to build a custom BaseParser for a specific file format.
Register it with a ParserRegistry and it takes priority over
built-in parsers for the same formats.
"""

from typing import List, Set

from definable.media import File
from definable.readers.base import BaseReader
from definable.readers.models import ContentBlock, ReaderConfig
from definable.readers.parsers.base_parser import BaseParser
from definable.readers.registry import ParserRegistry


class MarkdownParser(BaseParser):
  """Example: Custom parser that adds markdown-specific metadata."""

  def supported_mime_types(self) -> List[str]:
    return ["text/markdown"]

  def supported_extensions(self) -> Set[str]:
    return {".md"}

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    encoding = config.encoding if config else "utf-8"
    text = data.decode(encoding, errors="replace")
    # Count headings for metadata
    headings = [line for line in text.split("\n") if line.startswith("#")]
    return [
      ContentBlock(
        content_type="text",
        content=text,
        mime_type="text/markdown",
        metadata={"heading_count": len(headings)},
      )
    ]


# Usage: override default text parser for .md files with custom parser
registry = ParserRegistry()
registry.register(MarkdownParser(), priority=200)  # Higher priority = wins over defaults
reader = BaseReader(registry=registry)

# Test it
md_file = File(
  content=b"# Title\n\nSome content.\n\n## Section\n\nMore content.",
  filename="readme.md",
  mime_type="text/markdown",
)

result = reader.read(md_file)
print(f"Content: {result.content[:100]}")
print(f"Filename: {result.filename}")
print(f"Error: {result.error}")
