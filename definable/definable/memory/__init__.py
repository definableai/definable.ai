"""Session-history memory system for Definable AI agents.

The memory system stores conversation history per session with
auto-summarization when history exceeds a configurable threshold.

Quick Start:
    from definable.memory import Memory, SQLiteStore

    memory = Memory(store=SQLiteStore("./memory.db"))

    # Use with Agent — snaps in directly, no config wrapper needed:
    agent = Agent(model=model, memory=memory)
"""

from definable.memory.manager import Memory
from definable.memory.store.base import MemoryStore
from definable.memory.types import MemoryEntry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.store.file import FileStore
  from definable.memory.store.in_memory import InMemoryStore
  from definable.memory.store.sqlite import SQLiteStore
  from definable.memory.strategies.summarize import SummarizeStrategy

__all__ = [
  # Core
  "Memory",
  "MemoryEntry",
  # Protocol
  "MemoryStore",
  # Store implementations (lazy-loaded)
  "InMemoryStore",
  "SQLiteStore",
  "FileStore",
  # Strategies
  "SummarizeStrategy",
]

_LAZY_IMPORTS = {
  "SQLiteStore": ("definable.memory.store.sqlite", "SQLiteStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
  "FileStore": ("definable.memory.store.file", "FileStore"),
  "SummarizeStrategy": ("definable.memory.strategies.summarize", "SummarizeStrategy"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
