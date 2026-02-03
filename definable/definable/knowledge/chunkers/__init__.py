from definable.knowledge.chunkers.base import Chunker

__all__ = [
  "Chunker",
]


def __getattr__(name: str):
  if name == "TextChunker":
    from definable.knowledge.chunkers.text import TextChunker
    return TextChunker
  if name == "RecursiveChunker":
    from definable.knowledge.chunkers.recursive import RecursiveChunker
    return RecursiveChunker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
