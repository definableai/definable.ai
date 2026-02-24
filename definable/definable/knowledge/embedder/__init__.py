from definable.knowledge.embedder.base import Embedder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.embedder.openai import OpenAIEmbedder
  from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

__all__ = [
  "Embedder",
  # Implementations (lazy-loaded)
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
