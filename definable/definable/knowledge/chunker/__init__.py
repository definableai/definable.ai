from definable.knowledge.chunker.base import Chunker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.chunker.recursive import RecursiveChunker
  from definable.knowledge.chunker.text import TextChunker

__all__ = [
  "Chunker",
  # Implementations (lazy-loaded)
  "RecursiveChunker",
  "TextChunker",
]


def __getattr__(name: str):
  if name == "TextChunker":
    from definable.knowledge.chunker.text import TextChunker

    return TextChunker
  if name == "RecursiveChunker":
    from definable.knowledge.chunker.recursive import RecursiveChunker

    return RecursiveChunker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
