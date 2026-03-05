"""Session-history memory system for Definable AI agents.

The memory system stores conversation history per session with
configurable optimization strategies.

Composition: Memory = Strategy + Store + Optional[Embedder].

Strategies:
    - SummarizeStrategy: pin + summarize-middle + recent (default).
    - SemanticStrategy: extract atomic, self-contained facts from conversation.

Quick Start:
    from definable.memory import Memory, SQLiteStore

    memory = Memory(store=SQLiteStore("./memory.db"))
    agent = Agent(model=model, memory=memory)

    # With semantic extraction:
    from definable.memory import Memory, SemanticStrategy
    memory = Memory(strategy=SemanticStrategy())
    agent = Agent(model=model, memory=memory)
"""

from definable.memory.manager import Memory
from definable.memory.store.base import MemoryStore
from definable.memory.strategies.base import MemoryStrategy
from definable.memory.types import MemoryEntry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.consolidation import ConsolidationPolicy
  from definable.memory.cortex.cortex import CortexMemory
  from definable.memory.store.file import FileStore
  from definable.memory.store.in_memory import InMemoryStore
  from definable.memory.store.sqlite import SQLiteStore
  from definable.memory.strategies.semantic import SemanticStrategy
  from definable.memory.strategies.summarize import SummarizeStrategy

__all__ = [
  # Core
  "Memory",
  "MemoryEntry",
  # Protocol
  "MemoryStore",
  "MemoryStrategy",
  # Consolidation
  "ConsolidationPolicy",
  # Store implementations (lazy-loaded)
  "InMemoryStore",
  "SQLiteStore",
  "FileStore",
  # Strategies (lazy-loaded)
  "SummarizeStrategy",
  "SemanticStrategy",
  # Cortex (lazy-loaded)
  "CortexMemory",
]

_LAZY_IMPORTS = {
  "SQLiteStore": ("definable.memory.store.sqlite", "SQLiteStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
  "FileStore": ("definable.memory.store.file", "FileStore"),
  "SummarizeStrategy": ("definable.memory.strategies.summarize", "SummarizeStrategy"),
  "SemanticStrategy": ("definable.memory.strategies.semantic", "SemanticStrategy"),
  "ConsolidationPolicy": ("definable.memory.consolidation", "ConsolidationPolicy"),
  "CortexMemory": ("definable.memory.cortex.cortex", "CortexMemory"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
