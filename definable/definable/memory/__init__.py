"""Agentic memory system for Definable AI agents.

The memory system uses an LLM-driven approach: Memory calls the model
with tools (add_memory, update_memory, delete_memory) and the model decides what
facts about the user are worth remembering.

Quick Start:
    from definable.memory import Memory, SQLiteStore

    memory = Memory(store=SQLiteStore("./memory.db"))

    # Use with Agent — snaps in directly, no config wrapper needed:
    agent = Agent(model=model, memory=memory)
"""

from definable.memory.manager import Memory, MemoryManager
from definable.memory.store.base import MemoryStore
from definable.memory.types import UserMemory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.store.in_memory import InMemoryStore
  from definable.memory.store.postgres import PostgresStore
  from definable.memory.store.sqlite import SQLiteStore

__all__ = [
  # Core
  "Memory",
  "MemoryManager",  # backward compat alias
  "UserMemory",
  # Protocol
  "MemoryStore",
  # Store implementations (lazy-loaded)
  "InMemoryStore",
  "SQLiteStore",
  "PostgresStore",
]

_LAZY_IMPORTS = {
  "SQLiteStore": ("definable.memory.store.sqlite", "SQLiteStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
  "PostgresStore": ("definable.memory.store.postgres", "PostgresStore"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
