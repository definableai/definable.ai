"""Memory store implementations."""

from definable.memory.store.base import MemoryStore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.store.chroma import ChromaMemoryStore
  from definable.memory.store.in_memory import InMemoryStore
  from definable.memory.store.mem0 import Mem0MemoryStore
  from definable.memory.store.mongodb import MongoMemoryStore
  from definable.memory.store.pinecone import PineconeMemoryStore
  from definable.memory.store.postgres import PostgresMemoryStore
  from definable.memory.store.qdrant import QdrantMemoryStore
  from definable.memory.store.redis import RedisMemoryStore
  from definable.memory.store.sqlite import SQLiteMemoryStore

__all__ = [
  "MemoryStore",
  # Store implementations (lazy-loaded)
  "ChromaMemoryStore",
  "InMemoryStore",
  "Mem0MemoryStore",
  "MongoMemoryStore",
  "PineconeMemoryStore",
  "PostgresMemoryStore",
  "QdrantMemoryStore",
  "RedisMemoryStore",
  "SQLiteMemoryStore",
]

_LAZY_IMPORTS = {
  "SQLiteMemoryStore": ("definable.memory.store.sqlite", "SQLiteMemoryStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
  "PostgresMemoryStore": ("definable.memory.store.postgres", "PostgresMemoryStore"),
  "RedisMemoryStore": ("definable.memory.store.redis", "RedisMemoryStore"),
  "QdrantMemoryStore": ("definable.memory.store.qdrant", "QdrantMemoryStore"),
  "ChromaMemoryStore": ("definable.memory.store.chroma", "ChromaMemoryStore"),
  "MongoMemoryStore": ("definable.memory.store.mongodb", "MongoMemoryStore"),
  "PineconeMemoryStore": ("definable.memory.store.pinecone", "PineconeMemoryStore"),
  "Mem0MemoryStore": ("definable.memory.store.mem0", "Mem0MemoryStore"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
