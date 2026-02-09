"""Memory store implementations."""

from definable.memory.store.base import MemoryStore

__all__ = [
  "MemoryStore",
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
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
