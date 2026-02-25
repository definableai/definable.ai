"""Markdown-aware chunker that preserves document structure."""

import re
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from definable.knowledge.chunker.base import Chunker
from definable.knowledge.document import Document

# Heading pattern: # to ######
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class MarkdownChunker(Chunker):
  """Split Markdown documents preserving structural boundaries.

  Splits on headings first (configurable depth), then falls back to
  paragraph boundaries within large sections. Code blocks and lists
  are kept intact when possible.

  Args:
      chunk_size: Target chunk size in characters.
      chunk_overlap: Overlap between consecutive chunks (characters).
      max_heading_depth: Maximum heading depth to split on (1-6). Default 3.
      preserve_code_blocks: Keep fenced code blocks intact. Default True.

  Example::

      chunker = MarkdownChunker(chunk_size=1000, max_heading_depth=2)
      chunks = chunker.chunk(Document(content=markdown_text))
  """

  max_heading_depth: int = 3
  preserve_code_blocks: bool = True
  _code_placeholder: str = field(default="__CODE_BLOCK_{i}__", repr=False)

  def chunk(self, document: Document) -> List[Document]:
    text = document.content
    if not text:
      return []

    # Extract and protect code blocks
    code_blocks: List[str] = []
    if self.preserve_code_blocks:
      text, code_blocks = self._extract_code_blocks(text)

    # Split into sections by headings
    sections = self._split_by_headings(text)

    # Build chunks from sections
    raw_chunks = self._merge_sections(sections)

    # Restore code blocks
    if code_blocks:
      raw_chunks = [self._restore_code_blocks(c, code_blocks) for c in raw_chunks]

    # Create Document objects
    parent_id = document.id or str(uuid4())
    result: List[Document] = []
    for i, chunk_text in enumerate(raw_chunks):
      chunk_text = chunk_text.strip()
      if not chunk_text:
        continue
      result.append(
        Document(
          content=chunk_text,
          name=f"{document.name}_chunk_{i}" if document.name else f"chunk_{i}",
          source=document.source,
          source_type=document.source_type,
          parent_id=parent_id,
          chunk_index=i,
          chunk_total=len(raw_chunks),
          meta_data={**document.meta_data, "chunk_index": i, "chunk_total": len(raw_chunks), "chunker": "markdown"},
        )
      )

    # Fix chunk_total after filtering empties
    for doc in result:
      doc.chunk_total = len(result)
      doc.meta_data["chunk_total"] = len(result)

    return result

  def _extract_code_blocks(self, text: str) -> tuple[str, List[str]]:
    """Replace fenced code blocks with placeholders."""
    blocks: List[str] = []
    pattern = re.compile(r"```[\s\S]*?```", re.DOTALL)

    def _replace(m: re.Match[str]) -> str:
      blocks.append(m.group(0))
      return self._code_placeholder.format(i=len(blocks) - 1)

    cleaned = pattern.sub(_replace, text)
    return cleaned, blocks

  def _restore_code_blocks(self, text: str, blocks: List[str]) -> str:
    """Restore code blocks from placeholders."""
    for i, block in enumerate(blocks):
      text = text.replace(self._code_placeholder.format(i=i), block)
    return text

  def _split_by_headings(self, text: str) -> List[tuple[str, str]]:
    """Split text into (heading, body) pairs by heading boundaries.

    Returns a list of (heading_text, body_text) tuples.
    The first entry may have an empty heading (preamble text).
    """
    sections: List[tuple[str, str]] = []
    heading_pattern = re.compile(rf"^(#{{1,{self.max_heading_depth}}})\s+(.+)$", re.MULTILINE)

    matches = list(heading_pattern.finditer(text))
    if not matches:
      return [("", text)]

    # Preamble before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
      sections.append(("", preamble))

    for i, m in enumerate(matches):
      heading = m.group(0)
      start = m.end()
      end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
      body = text[start:end].strip()
      sections.append((heading, body))

    return sections

  def _merge_sections(self, sections: List[tuple[str, str]]) -> List[str]:
    """Convert sections into chunks, splitting large sections further.

    Each heading section becomes its own chunk. Sections exceeding
    chunk_size are split by paragraphs.
    """
    chunks: List[str] = []

    for heading, body in sections:
      section_text = f"{heading}\n\n{body}" if heading else body

      if len(section_text) > self.chunk_size:
        sub_chunks = self._split_large_section(heading, body)
        chunks.extend(sub_chunks)
      else:
        chunks.append(section_text)

    return chunks

  def _split_large_section(self, heading: str, body: str) -> List[str]:
    """Split a large section by paragraphs."""
    paragraphs = re.split(r"\n\n+", body)
    chunks: List[str] = []
    current = heading or ""

    for para in paragraphs:
      combined = f"{current}\n\n{para}" if current else para
      if len(combined) <= self.chunk_size:
        current = combined
      else:
        if current:
          chunks.append(current)
        # If a single paragraph exceeds chunk_size, force-split it
        if len(para) > self.chunk_size:
          step = max(1, self.chunk_size - self.chunk_overlap)
          for i in range(0, len(para), step):
            chunks.append(para[i : i + self.chunk_size])
          current = ""
        else:
          current = para

    if current:
      chunks.append(current)

    return chunks
