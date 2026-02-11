"""BaseReader — orchestrator that resolves files to parsed content.

Replaces the old FileReader ABC and FileReaderRegistry. This is the
primary public API for the readers module.

Flow: File → get_file_bytes → check size → detect(mime) → registry.get_parser(mime, ext)
     → parser.parse(bytes) → build ReaderOutput
"""

import asyncio
from typing import Any, List, Optional

from definable.media import File
from definable.readers.detector import detect
from definable.readers.models import ContentBlock, ReaderConfig, ReaderOutput
from definable.readers.registry import ParserRegistry
from definable.readers.utils import (
  extract_extension as extract_file_extension,
  get_file_bytes_async,
  get_file_bytes_sync,
  get_filename,
)
from definable.utils.log import log_warning


class BaseReader:
  """Orchestrator that resolves files to parsed content.

  Detects file format, dispatches to the appropriate parser, and
  returns a ReaderOutput with ContentBlock list.

  Args:
    config: Optional reader configuration (max_file_size, etc.).
    registry: ParserRegistry for format-to-parser mapping.
      If None, creates a default registry.
  """

  def __init__(
    self,
    config: Optional[ReaderConfig] = None,
    registry: Optional[ParserRegistry] = None,
  ) -> None:
    self.config = config
    self.registry = registry or ParserRegistry()

  def register(self, parser: Any, priority: int = 100) -> "BaseReader":
    """Register a parser with the given priority. Returns self for chaining."""
    self.registry.register(parser, priority)
    return self

  def get_reader(self, file: File) -> Optional[Any]:
    """Return the parser that can handle *file*, or None.

    Backwards-compatible alias for get_parser().
    """
    return self.get_parser(file)

  def get_parser(self, file: File) -> Optional[Any]:
    """Return the parser that can handle *file*, or None."""
    mime_type, ext = self._detect_format(file)
    return self.registry.get_parser(mime_type, ext)

  def read(self, file: File) -> ReaderOutput:
    """Synchronously read a single file."""
    try:
      raw = get_file_bytes_sync(
        file,
        encoding=self._encoding,
        timeout=self._timeout,
      )
      return self._process(file, raw)
    except Exception as e:
      return self._make_error_output(file, str(e))

  async def aread(self, file: File) -> ReaderOutput:
    """Asynchronously read a single file."""
    try:
      raw = await get_file_bytes_async(
        file,
        encoding=self._encoding,
        timeout=self._timeout,
      )
      return self._process(file, raw)
    except Exception as e:
      return self._make_error_output(file, str(e))

  async def aread_all(self, files: List[File]) -> List[ReaderOutput]:
    """Concurrently read multiple files via ``asyncio.gather``."""
    return list(await asyncio.gather(*(self.aread(f) for f in files)))

  def _process(self, file: File, raw: bytes) -> ReaderOutput:
    """Core processing: size check → detect → parse → build output."""
    filename = get_filename(file)

    # Check file size
    self._check_file_size(raw)

    # Detect format
    ext = extract_file_extension(file)
    mime_type = detect(
      data=raw,
      filename=file.filename or file.name,
      filepath=str(file.filepath) if file.filepath else None,
      url=file.url,
      mime_type=file.mime_type,
    )

    # Find parser
    parser = self.registry.get_parser(mime_type, ext)
    if parser is None:
      log_warning(f"No parser found for file: {filename} (mime={mime_type}, ext={ext})")
      return ReaderOutput(
        filename=filename,
        mime_type=mime_type,
        error=f"No parser available for this file type (mime={mime_type})",
      )

    # Parse
    blocks = parser.parse(data=raw, mime_type=mime_type, config=self.config)

    # Truncate if needed
    truncated = False
    if self.config and self.config.max_content_length:
      blocks, truncated = self._truncate_blocks(blocks, self.config.max_content_length)

    # Build output
    text = "\n\n".join(b.as_text() for b in blocks)
    word_count = len(text.split()) if text else 0
    page_count = self._count_pages(blocks)

    return ReaderOutput(
      filename=filename,
      blocks=blocks,
      mime_type=mime_type,
      page_count=page_count,
      word_count=word_count,
      truncated=truncated,
    )

  def _detect_format(self, file: File) -> tuple[str | None, str | None]:
    """Detect format for a file (for get_parser, before reading bytes)."""
    ext = extract_file_extension(file)
    mime_type = file.mime_type
    if not mime_type:
      mime_type = detect(
        filename=file.filename or file.name,
        filepath=str(file.filepath) if file.filepath else None,
        url=file.url,
      )
    return mime_type, ext

  def _check_file_size(self, raw: bytes) -> None:
    """Raise ValueError if file exceeds max_file_size."""
    max_size = self.config.max_file_size if self.config else None
    if max_size and len(raw) > max_size:
      raise ValueError(f"File size {len(raw)} exceeds max_file_size {max_size}")

  @staticmethod
  def _truncate_blocks(blocks: List[ContentBlock], max_length: int) -> tuple[List[ContentBlock], bool]:
    """Truncate text content across blocks to fit max_content_length."""
    total = 0
    result: List[ContentBlock] = []
    truncated = False

    for block in blocks:
      text = block.as_text()
      remaining = max_length - total
      if remaining <= 0:
        truncated = True
        break
      if len(text) > remaining:
        # Truncate this block's content
        if isinstance(block.content, str):
          truncated_block = ContentBlock(
            content_type=block.content_type,
            content=block.content[:remaining],
            mime_type=block.mime_type,
            metadata=block.metadata,
            page_number=block.page_number,
          )
          result.append(truncated_block)
        else:
          result.append(block)
        truncated = True
        break
      result.append(block)
      total += len(text)

    return result, truncated

  @staticmethod
  def _count_pages(blocks: List[ContentBlock]) -> int | None:
    """Count pages from block page_number metadata."""
    page_numbers = [b.page_number for b in blocks if b.page_number is not None]
    if page_numbers:
      return max(page_numbers)
    return None

  def _make_error_output(self, file: File, error: str) -> ReaderOutput:
    """Create an error ReaderOutput."""
    filename = get_filename(file)
    log_warning(f"Reader error for {filename}: {error}")
    return ReaderOutput(
      filename=filename,
      mime_type=file.mime_type,
      error=error,
    )

  @property
  def _encoding(self) -> str:
    return self.config.encoding if self.config else "utf-8"

  @property
  def _timeout(self) -> float | None:
    return self.config.timeout if self.config else 30.0


# Backwards-compatible aliases
FileReader = BaseReader
FileReaderConfig = ReaderConfig
ReaderResult = ReaderOutput
FileReaderRegistry = BaseReader
