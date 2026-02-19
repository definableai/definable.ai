"""Memory store implementations."""

from definable.memory.store.base import MemoryStore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.store.file import FileStore
  from definable.memory.store.in_memory import InMemoryStore
  from definable.memory.store.sqlite import SQLiteStore

__all__ = [
  "MemoryStore",
  "InMemoryStore",
  "SQLiteStore",
  "FileStore",
]

_LAZY_IMPORTS = {
  "SQLiteStore": ("definable.memory.store.sqlite", "SQLiteStore"),
  "InMemoryStore": ("definable.memory.store.in_memory", "InMemoryStore"),
  "FileStore": ("definable.memory.store.file", "FileStore"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
