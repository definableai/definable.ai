from definable.knowledge.vector_dbs.base import VectorDB

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.knowledge.vector_dbs.memory import InMemoryVectorDB
  from definable.knowledge.vector_dbs.pgvector import PgVectorDB

__all__ = [
  "VectorDB",
  # Implementations (lazy-loaded)
  "InMemoryVectorDB",
  "PgVectorDB",
]


def __getattr__(name: str):
  if name == "InMemoryVectorDB":
    from definable.knowledge.vector_dbs.memory import InMemoryVectorDB

    return InMemoryVectorDB
  if name == "PgVectorDB":
    from definable.knowledge.vector_dbs.pgvector import PgVectorDB

    return PgVectorDB
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
