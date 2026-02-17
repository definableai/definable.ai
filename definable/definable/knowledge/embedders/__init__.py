from definable.knowledge.embedders.base import Embedder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.embedders.openai import OpenAIEmbedder
  from definable.knowledge.embedders.voyageai import VoyageAIEmbedder

__all__ = [
  "Embedder",
  "OpenAIEmbedder",
  "VoyageAIEmbedder",
]


def __getattr__(name: str):
  if name == "OpenAIEmbedder":
    from definable.knowledge.embedders.openai import OpenAIEmbedder

    return OpenAIEmbedder
  if name == "VoyageAIEmbedder":
    from definable.knowledge.embedders.voyageai import VoyageAIEmbedder

    return VoyageAIEmbedder
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
