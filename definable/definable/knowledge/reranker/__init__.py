from definable.knowledge.reranker.base import Reranker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.reranker.cohere import CohereReranker

__all__ = [
  "Reranker",
  # Implementations (lazy-loaded)
  "CohereReranker",
]


def __getattr__(name: str):
  if name == "CohereReranker":
    from definable.knowledge.reranker.cohere import CohereReranker

    return CohereReranker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
