"""Top-level reranker re-exports for convenience.

Usage:
    from definable.reranker import CohereReranker, Reranker
"""

from typing import TYPE_CHECKING

from definable.knowledge.reranker import Reranker

if TYPE_CHECKING:
  from definable.knowledge.reranker.cohere import CohereReranker

__all__ = [
  "Reranker",
  "CohereReranker",
]


def __getattr__(name: str):
  if name == "CohereReranker":
    from definable.knowledge.reranker.cohere import CohereReranker

    return CohereReranker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
