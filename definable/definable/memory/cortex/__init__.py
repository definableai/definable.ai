"""Cortex — next-generation memory layer for Definable AI agents.

Cortex goes beyond retrieval — it learns the user through multi-representation
ingestion, 5-layer retrieval cascade, cascade-aware updates, and behavioral learning.

Quick Start:
    from definable.memory.cortex import CortexMemory

    memory = CortexMemory()
    agent = Agent(model=model, memory=memory)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.memory.cortex.config import CortexConfig
  from definable.memory.cortex.cortex import CortexMemory
  from definable.memory.cortex.record.scratchpad import Scratchpad
  from definable.memory.cortex.record.types import (
    Edge,
    EdgeType,
    Fact,
    MemoryRecord,
    MemorySource,
    NarrativeEpisode,
  )
  from definable.memory.cortex.store import CortexStore

__all__ = [
  "CortexMemory",
  "CortexConfig",
  "CortexStore",
  "MemoryRecord",
  "NarrativeEpisode",
  "Fact",
  "Edge",
  "EdgeType",
  "MemorySource",
  "Scratchpad",
]

_LAZY_IMPORTS = {
  "CortexMemory": ("definable.memory.cortex.cortex", "CortexMemory"),
  "CortexConfig": ("definable.memory.cortex.config", "CortexConfig"),
  "CortexStore": ("definable.memory.cortex.store", "CortexStore"),
  "MemoryRecord": ("definable.memory.cortex.record.types", "MemoryRecord"),
  "NarrativeEpisode": ("definable.memory.cortex.record.types", "NarrativeEpisode"),
  "Fact": ("definable.memory.cortex.record.types", "Fact"),
  "Edge": ("definable.memory.cortex.record.types", "Edge"),
  "EdgeType": ("definable.memory.cortex.record.types", "EdgeType"),
  "MemorySource": ("definable.memory.cortex.record.types", "MemorySource"),
  "Scratchpad": ("definable.memory.cortex.record.scratchpad", "Scratchpad"),
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
  if name in _LAZY_IMPORTS:
    module_path, class_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
