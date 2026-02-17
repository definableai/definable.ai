from definable.knowledge.rerankers.base import Reranker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.rerankers.cohere import CohereReranker

__all__ = [
  "Reranker",
  "CohereReranker",
]


def __getattr__(name: str):
  if name == "CohereReranker":
    from definable.knowledge.rerankers.cohere import CohereReranker

    return CohereReranker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
