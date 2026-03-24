"""Tool-based memory system for Definable AI agents.

The LLM is the memory manager. No extraction pipeline. The agent decides
what to store, what to evict, and when to recall — via tool calls.

Two tiers:
  - Working memory: always loaded into the prompt (scratchpad)
  - Archived memory: indexed, on-demand (search → fetch)

Quick Start:
    from definable.memory.v2 import Memory, SQLiteStore

    agent = Agent(model="openai/gpt-4o", memory=Memory(store=SQLiteStore("./memory.db")))
"""

from definable.memory.v2.memory import Memory
from definable.memory.v2.models import IndexEntry, MemoryEntry, WorkingMemory
from definable.memory.v2.stores.base import MemoryStore
from definable.memory.v2.stores.sqlite import SQLiteStore

__all__ = [
  "Memory",
  "MemoryStore",
  "SQLiteStore",
  "WorkingMemory",
  "IndexEntry",
  "MemoryEntry",
]
