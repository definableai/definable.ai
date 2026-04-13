"""Tool-based memory system for Definable AI agents.

The LLM is the memory manager. No extraction pipeline. The agent decides
what to store, what to evict, and when to recall — via tool calls.

Two tiers:
  - Working memory: always loaded into the prompt (scratchpad)
  - Archived memory: indexed, on-demand (search -> fetch)

Enterprise features:
  - Per-user SQLite files for tenant isolation
  - GDPR delete/export
  - Background consolidation (dedupe, prune, TTL)
  - WM versioning and rollback
  - Recency-weighted search with access tracking
  - Framework-level validation (WM structure, summary quality)

Quick Start:
    from definable.memory.v2 import Memory, SQLiteStore

    agent = Agent(model="openai/gpt-4o", memory=Memory(store=SQLiteStore("./memory.db")))

Per-user isolation:
    from definable.memory.v2 import Memory, PerUserSQLiteStore

    agent = Agent(model="openai/gpt-4o", memory=Memory(store=PerUserSQLiteStore("./memory_data")))
"""

from definable.memory.v2.memory import Memory
from definable.memory.v2.models import (
  ConsolidationReport,
  IndexEntry,
  MemoryEntry,
  MemoryStats,
  WarmMemory,
  WorkingMemory,
  WorkingMemorySnapshot,
)
from definable.memory.v2.stores.base import MemoryStore
from definable.memory.v2.stores.per_user_sqlite import PerUserSQLiteStore
from definable.memory.v2.stores.sqlite import SQLiteStore

__all__ = [
  "Memory",
  "MemoryStore",
  "SQLiteStore",
  "PerUserSQLiteStore",
  "WorkingMemory",
  "WarmMemory",
  "WorkingMemorySnapshot",
  "IndexEntry",
  "MemoryEntry",
  "MemoryStats",
  "ConsolidationReport",
]
