"""Vector database implementations.

Provides a unified VectorDB interface with built-in embedder support.
Each backend embeds queries internally during search.
Documents arrive pre-embedded from the Knowledge layer.

Quick Start:
  from definable.vectordb import InMemoryVectorDB
  from definable.knowledge import Document

  db = InMemoryVectorDB(name="my_collection")
  db.create()
  db.insert("hash123", [Document(content="Hello world")])
  results = db.search("greeting", limit=5)

Available Backends:
  - InMemoryVectorDB — Pure Python, no external deps (beyond numpy)
  - PgVector — PostgreSQL + pgvector
  - Qdrant — Qdrant vector search engine
  - ChromaDb — ChromaDB
  - MongoDb — MongoDB Atlas vector search
  - RedisDB — Redis with RediSearch
  - PineconeDb — Pinecone managed service
"""

from typing import TYPE_CHECKING

from definable.vectordb.base import VectorDB
from definable.vectordb.distance import Distance
from definable.vectordb.search import SearchType

if TYPE_CHECKING:
  from definable.vectordb.chroma.chromadb import ChromaDb
  from definable.vectordb.memory.memory import InMemoryVectorDB
  from definable.vectordb.mongodb.mongodb import MongoDb
  from definable.vectordb.pgvector.pgvector import PgVector
  from definable.vectordb.pineconedb.pineconedb import PineconeDb
  from definable.vectordb.qdrant.qdrant import Qdrant
  from definable.vectordb.redis.redisdb import RedisDB

__all__ = [
  # Base
  "VectorDB",
  "Distance",
  "SearchType",
  # Implementations (lazy-loaded)
  "InMemoryVectorDB",
  "PgVector",
  "Qdrant",
  "ChromaDb",
  "MongoDb",
  "RedisDB",
  "PineconeDb",
]


def __getattr__(name: str):
  if name == "InMemoryVectorDB":
    from definable.vectordb.memory.memory import InMemoryVectorDB

    return InMemoryVectorDB

  if name == "PgVector":
    from definable.vectordb.pgvector.pgvector import PgVector

    return PgVector

  if name == "Qdrant":
    from definable.vectordb.qdrant.qdrant import Qdrant

    return Qdrant

  if name == "ChromaDb":
    from definable.vectordb.chroma.chromadb import ChromaDb

    return ChromaDb

  if name == "MongoDb":
    from definable.vectordb.mongodb.mongodb import MongoDb

    return MongoDb

  if name == "RedisDB":
    from definable.vectordb.redis.redisdb import RedisDB

    return RedisDB

  if name == "PineconeDb":
    from definable.vectordb.pineconedb.pineconedb import PineconeDb

    return PineconeDb

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
