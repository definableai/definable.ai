"""Core types for the readers module.

Defines ContentBlock (multimodal output), ReaderOutput (replaces ReaderResult),
and ReaderConfig (replaces FileReaderConfig).
"""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class ContentBlock:
  """A single block of content extracted from a file.

  Supports text, images, tables, audio, and raw binary content.
  """

  content_type: Literal["text", "image", "table", "audio", "raw"]
  content: str | bytes
  mime_type: str | None = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  page_number: int | None = None

  def as_text(self) -> str:
    """Return a text representation of this block."""
    if isinstance(self.content, str):
      return self.content
    if self.content_type == "image":
      return f"[image: {self.mime_type or 'unknown'}]"
    if self.content_type == "audio":
      return f"[audio: {self.mime_type or 'unknown'}]"
    return f"[binary: {len(self.content)} bytes]"

  def as_message_content(self) -> dict:
    """Convert to OpenAI-format message content part."""
    if self.content_type in ("text", "table"):
      return {"type": "text", "text": self.as_text()}

    if self.content_type == "image" and isinstance(self.content, bytes):
      mime = self.mime_type or "image/png"
      b64 = base64.b64encode(self.content).decode("ascii")
      return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
      }

    if self.content_type == "audio" and isinstance(self.content, bytes):
      mime = self.mime_type or "audio/mpeg"
      b64 = base64.b64encode(self.content).decode("ascii")
      return {
        "type": "input_audio",
        "input_audio": {"data": b64, "format": mime},
      }

    return {"type": "text", "text": self.as_text()}


@dataclass
class ReaderOutput:
  """Output of a file reader — extracted content blocks plus metadata.

  Replaces ReaderResult with multimodal-aware output.
  """

  filename: str
  blocks: List[ContentBlock] = field(default_factory=list)
  mime_type: str | None = None
  page_count: int | None = None
  word_count: int | None = None
  truncated: bool = False
  error: str | None = None
  metadata: Dict[str, Any] = field(default_factory=dict)

  def as_text(self, separator: str = "\n\n") -> str:
    """Join all block text representations."""
    return separator.join(block.as_text() for block in self.blocks)

  def as_messages(self) -> List[dict]:
    """All blocks as OpenAI-format content parts."""
    return [block.as_message_content() for block in self.blocks]

  @property
  def content(self) -> str:
    """Backwards-compatible text content property."""
    return self.as_text()


@dataclass
class ReaderConfig:
  """Per-reader configuration. Replaces FileReaderConfig."""

  max_file_size: int | None = None
  max_content_length: int | None = None
  encoding: str = "utf-8"
  timeout: float | None = 30.0
