"""Recursive text chunker implementation."""
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from definable.knowledge.chunkers.base import Chunker
from definable.knowledge.document import Document


@dataclass
class RecursiveChunker(Chunker):
  """Recursively split text using multiple separators."""

  separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

  def chunk(self, document: Document) -> List[Document]:
    """Split document into chunks using recursive splitting."""
    text = document.content

    if not text:
      return []

    chunks = self._split_recursive(text, self.separators)

    # Create Document objects
    parent_id = document.id or str(uuid4())
    result: List[Document] = []

    for i, chunk_text in enumerate(chunks):
      result.append(
        Document(
          content=chunk_text,
          name=f"{document.name}_chunk_{i}" if document.name else f"chunk_{i}",
          source=document.source,
          source_type=document.source_type,
          parent_id=parent_id,
          chunk_index=i,
          chunk_total=len(chunks),
          meta_data={**document.meta_data, "chunk_index": i, "chunk_total": len(chunks)},
        )
      )

    return result

  def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
    """Recursively split text using separators in order."""
    if not separators:
      # No more separators, just split by chunk_size
      return self._split_by_size(text)

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
      # Empty separator means split by characters
      return self._split_by_size(text)

    if separator not in text:
      # This separator doesn't exist, try next one
      return self._split_recursive(text, remaining_separators)

    # Split by current separator
    parts = text.split(separator)
    chunks: List[str] = []
    current_chunk = ""

    for part in parts:
      test_chunk = current_chunk + (separator if current_chunk else "") + part

      if len(test_chunk) <= self.chunk_size:
        current_chunk = test_chunk
      else:
        if current_chunk:
          chunks.append(current_chunk)

        # If part is still too large, recursively split with remaining separators
        if len(part) > self.chunk_size:
          sub_chunks = self._split_recursive(part, remaining_separators)
          chunks.extend(sub_chunks[:-1])
          current_chunk = sub_chunks[-1] if sub_chunks else ""
        else:
          current_chunk = part

    if current_chunk:
      chunks.append(current_chunk)

    # Apply overlap
    if self.chunk_overlap > 0 and len(chunks) > 1:
      overlapped: List[str] = [chunks[0]]
      for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
        overlapped.append(overlap + chunks[i])
      return overlapped

    return chunks

  def _split_by_size(self, text: str) -> List[str]:
    """Split text into chunks of chunk_size."""
    if len(text) <= self.chunk_size:
      return [text] if text else []

    chunks: List[str] = []
    for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
      chunk = text[i : i + self.chunk_size]
      if chunk:
        chunks.append(chunk)

    return chunks
