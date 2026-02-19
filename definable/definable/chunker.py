"""Top-level chunker re-exports for convenience.

Usage:
    from definable.chunker import RecursiveChunker, TextChunker, Chunker
"""

from typing import TYPE_CHECKING

from definable.knowledge.chunker import Chunker

if TYPE_CHECKING:
  from definable.knowledge.chunker.recursive import RecursiveChunker
  from definable.knowledge.chunker.text import TextChunker

__all__ = [
  "Chunker",
  "RecursiveChunker",
  "TextChunker",
]


def __getattr__(name: str):
  if name == "RecursiveChunker":
    from definable.knowledge.chunker.recursive import RecursiveChunker

    return RecursiveChunker
  if name == "TextChunker":
    from definable.knowledge.chunker.text import TextChunker

    return TextChunker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
