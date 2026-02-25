from definable.knowledge.reranker.base import Reranker

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.reranker.cohere import CohereReranker
  from definable.knowledge.reranker.sentence_transformer import SentenceTransformerReranker

__all__ = [
  "Reranker",
  # Implementations (lazy-loaded)
  "CohereReranker",
  "SentenceTransformerReranker",
]


def __getattr__(name: str):
  if name == "CohereReranker":
    from definable.knowledge.reranker.cohere import CohereReranker

    return CohereReranker
  if name == "SentenceTransformerReranker":
    from definable.knowledge.reranker.sentence_transformer import SentenceTransformerReranker

    return SentenceTransformerReranker
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
