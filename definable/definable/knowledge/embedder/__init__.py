from definable.knowledge.embedder.base import Embedder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.embedder.google import GoogleEmbedder
  from definable.knowledge.embedder.mistral import MistralEmbedder
  from definable.knowledge.embedder.openai import OpenAIEmbedder
  from definable.knowledge.embedder.voyageai import VoyageAIEmbedder

__all__ = [
  "Embedder",
  # Implementations (lazy-loaded)
  "FallbackEmbedder",  # noqa: F822
  "GoogleEmbedder",
  "MistralEmbedder",
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
  if name == "FallbackEmbedder":
    from definable.knowledge.embedder.fallback import FallbackEmbedder

    return FallbackEmbedder
  if name == "GoogleEmbedder":
    from definable.knowledge.embedder.google import GoogleEmbedder

    return GoogleEmbedder
  if name == "MistralEmbedder":
    from definable.knowledge.embedder.mistral import MistralEmbedder

    return MistralEmbedder
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
