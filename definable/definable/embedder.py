"""Top-level embedder re-exports for convenience.

Usage:
    from definable.embedder import OpenAIEmbedder, VoyageAIEmbedder, Embedder
"""

from typing import TYPE_CHECKING

from definable.knowledge.embedder import Embedder

if TYPE_CHECKING:
  from definable.knowledge.embedder.openai import OpenAIEmbedder
  from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

__all__ = [
  "Embedder",
  "OpenAIEmbedder",
  "VoyageAIEmbedder",
]


def __getattr__(name: str):
  if name == "OpenAIEmbedder":
    from definable.knowledge.embedder.openai import OpenAIEmbedder

    return OpenAIEmbedder
  if name == "VoyageAIEmbedder":
    from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

    return VoyageAIEmbedder
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
