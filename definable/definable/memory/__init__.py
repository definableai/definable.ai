"""Cognitive memory system for Definable AI agents.

Provides multi-tier memory with token-budget-aware retrieval,
progressive distillation, and predictive pre-loading.

Quick Start:
    from definable.memory import CognitiveMemory, SQLiteMemoryStore

    memory = CognitiveMemory(
        store=SQLiteMemoryStore("./memory.db"),
        token_budget=500,
    )

    # Use with Agent:
    agent = Agent(model=model, memory=memory)
"""

from definable.memory.config import MemoryConfig, ScoringWeights
from definable.memory.memory import CognitiveMemory
from definable.memory.store.base import MemoryStore
from definable.memory.types import Episode, KnowledgeAtom, MemoryPayload, Procedure, TopicTransition

__all__ = [
  # Core
  "CognitiveMemory",
  "MemoryConfig",
  "ScoringWeights",
  # Protocol
  "MemoryStore",
  # Types
  "Episode",
  "KnowledgeAtom",
  "Procedure",
  "TopicTransition",
  "MemoryPayload",
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
