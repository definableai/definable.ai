"""Text chunker implementation."""
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from definable.knowledge.chunkers.base import Chunker
from definable.knowledge.document import Document


@dataclass
class TextChunker(Chunker):
  """Simple text chunker that splits by separators."""

  separator: str = "\n\n"
  keep_separator: bool = False

  def chunk(self, document: Document) -> List[Document]:
    """Split document into chunks."""
    text = document.content

    if not text:
      return []

    # Split by separator
    if self.separator:
      parts = text.split(self.separator)
    else:
      parts = [text]

    # Merge parts to fit chunk_size
    chunks: List[str] = []
    current_chunk = ""

    for part in parts:
      # If adding this part would exceed chunk_size
      separator_len = len(self.separator) if self.keep_separator and current_chunk else 0
      if current_chunk and len(current_chunk) + separator_len + len(part) > self.chunk_size:
        chunks.append(current_chunk)
        current_chunk = part
      else:
        if current_chunk:
          if self.keep_separator:
            current_chunk += self.separator + part
          else:
            current_chunk += self.separator + part
        else:
          current_chunk = part

    if current_chunk:
      chunks.append(current_chunk)

    # Handle overlap by including end of previous chunk
    if self.chunk_overlap > 0 and len(chunks) > 1:
      overlapped_chunks: List[str] = [chunks[0]]
      for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        overlap_text = prev_chunk[-self.chunk_overlap :] if len(prev_chunk) > self.chunk_overlap else prev_chunk
        overlapped_chunks.append(overlap_text + chunks[i])
      chunks = overlapped_chunks

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
