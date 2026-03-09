# Memory

> Session-history memory with auto-summarization for Definable AI agents.

The Memory system stores conversation history per session. Messages are stored as `MemoryEntry` objects in pluggable backends. When history exceeds a configurable threshold and a model is available, the middle section is automatically summarized to keep context windows manageable.

## Quick Start

```python
import asyncio
from definable.memory import Memory, InMemoryStore, SQLiteStore
from definable.model.message import Message


async def main():
  # Default: InMemoryStore (created automatically)
  memory = Memory()

  # Add messages
  await memory.add(Message(role="user", content="What is Python?"), session_id="s1")
  await memory.add(Message(role="assistant", content="A programming language."), session_id="s1")

  # Retrieve entries
  entries = await memory.get_entries("s1")
  print(f"Stored {len(entries)} entries")  # 2

  # Get as Message objects (ready for agent context)
  messages = await memory.get_context_messages("s1")
  print(messages[0].role, messages[0].content)  # user What is Python?

  await memory.close()


asyncio.run(main())
```

## Architecture

```
Memory (manager)
  │
  ├── store: MemoryStore (protocol)
  │     ├── InMemoryStore ─── dict of lists, ephemeral
  │     ├── SQLiteStore ───── aiosqlite, persistent
  │     └── FileStore ─────── JSONL files, human-readable
  │
  ├── model: Model (optional) ─── for auto-summarization
  │
  └── strategies/
        └── SummarizeStrategy ─── pin + summarize-middle + keep-recent
```

### Module Structure

```
memory/
├── __init__.py         # Public API: Memory, MemoryEntry, stores, SummarizeStrategy
├── manager.py          # Memory class (the main entry point)
├── types.py            # MemoryEntry dataclass
├── store/
│   ├── base.py         # MemoryStore protocol
│   ├── in_memory.py    # InMemoryStore
│   ├── sqlite.py       # SQLiteStore (requires aiosqlite)
│   └── file.py         # FileStore (JSONL)
└── strategies/
    ├── base.py         # MemoryStrategy ABC
    └── summarize.py    # SummarizeStrategy
```

## API Reference

### Memory

The main entry point. Snaps directly into Agent — no config wrapper needed.

```python
from definable.memory import Memory

memory = Memory(
  store=None,  # Backend store. None → InMemoryStore (auto-created)
  model=None,  # LLM for summarization. None → uses agent's model at runtime
  enabled=True,  # Whether memory is active
  max_messages=100,  # Threshold for auto-optimization
  pin_count=2,  # Initial messages to preserve during optimization
  recent_count=5,  # Recent messages to preserve during optimization
  description=None,  # Description shown in agent layer guide
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `await memory.add(message, session_id, user_id)` | Add a Message to session memory |
| `get_entries` | `await memory.get_entries(session_id, user_id, limit)` | Get raw MemoryEntry objects |
| `get_context_messages` | `await memory.get_context_messages(session_id, user_id)` | Get entries as Message objects |
| `update` | `await memory.update(memory_id, content)` | Update an entry's content |
| `delete` | `await memory.delete(memory_id)` | Delete a single entry |
| `clear` | `await memory.clear(session_id)` | Clear all entries for a session |
| `close` | `await memory.close()` | Close the underlying store |

### MemoryEntry

The data object stored by all backends.

```python
from definable.memory import MemoryEntry

entry = MemoryEntry(
  session_id="sess-1",  # Required — session scope
  memory_id=None,  # Auto-generated UUID if not provided
  user_id="default",  # User scope
  role="user",  # "user" | "assistant" | "tool" | "system" | "summary"
  content="Hello, world!",  # Text content
  message_data=None,  # Full serialized message (preserves tool_calls)
  created_at=None,  # Auto-set to time.time()
  updated_at=None,  # Auto-set to time.time()
)

# Serialization
d = entry.to_dict()
restored = MemoryEntry.from_dict(d)
```

### MemoryStore Protocol

All stores implement this async protocol:

```python
from definable.memory import MemoryStore


class MemoryStore(Protocol):
  async def initialize(self) -> None: ...
  async def close(self) -> None: ...
  async def add(self, entry: MemoryEntry) -> None: ...
  async def get_entries(self, session_id, user_id="default", limit=None) -> list[MemoryEntry]: ...
  async def get_entry(self, memory_id) -> MemoryEntry | None: ...
  async def update(self, entry: MemoryEntry) -> None: ...
  async def delete(self, memory_id) -> None: ...
  async def delete_session(self, session_id, user_id=None) -> None: ...
  async def count(self, session_id, user_id="default") -> int: ...
```

## Store Implementations

### InMemoryStore

Ephemeral, in-process storage. Best for testing and short-lived processes.

```python
import asyncio
from definable.memory import InMemoryStore, MemoryEntry


async def main():
  store = InMemoryStore()
  await store.initialize()

  await store.add(MemoryEntry(session_id="s1", role="user", content="Hello"))
  await store.add(MemoryEntry(session_id="s1", role="assistant", content="Hi!"))

  entries = await store.get_entries("s1")
  print(len(entries))  # 2

  count = await store.count("s1")
  print(count)  # 2

  await store.close()


asyncio.run(main())
```

### SQLiteStore

Persistent storage via aiosqlite. Auto-creates tables on first use.

```python
import asyncio
from definable.memory import SQLiteStore, MemoryEntry


async def main():
  store = SQLiteStore("./my_memory.db")  # or None → .definable/memory.db
  await store.initialize()

  await store.add(MemoryEntry(session_id="s1", role="user", content="Remember this"))

  entries = await store.get_entries("s1")
  print(entries[0].content)  # "Remember this"

  await store.close()


asyncio.run(main())
```

> **Requires:** `pip install aiosqlite`

### FileStore

JSONL file-based storage. Human-readable, good for debugging.

```python
import asyncio
from definable.memory import FileStore, MemoryEntry


async def main():
  store = FileStore("./memory_data")  # or None → .definable/memory/
  await store.initialize()

  await store.add(MemoryEntry(session_id="chat1", user_id="alice", role="user", content="Hi"))

  entries = await store.get_entries("chat1", "alice")
  print(entries[0].content)  # "Hi"

  await store.close()


asyncio.run(main())
```

**Directory structure:**
```
memory_data/
  chat1/
    alice.jsonl      ← one JSON line per MemoryEntry
    default.jsonl
  chat2/
    alice.jsonl
```

## Context Manager Support

All stores and Memory itself support `async with`:

```python
import asyncio
from definable.memory import Memory
from definable.model.message import Message


async def main():
  async with Memory() as memory:
    await memory.add(Message(role="user", content="Hello"), session_id="s1")
    entries = await memory.get_entries("s1")
    print(len(entries))  # 1
  # store automatically closed


asyncio.run(main())
```

## Agent Integration

### Basic Usage

```python
from definable.agent import Agent
from definable.memory import Memory

# Quick: auto-creates InMemoryStore
agent = Agent(model="openai/gpt-4o-mini", memory=True)

# Explicit store
agent = Agent(
  model="openai/gpt-4o-mini",
  memory=Memory(store=SQLiteStore("./agent_memory.db")),
)
```

### Multi-Turn Conversations

Memory automatically injects past messages into agent context:

```python
from definable.agent import Agent
from definable.memory import Memory, SQLiteStore

agent = Agent(
  model="openai/gpt-4o-mini",
  memory=Memory(store=SQLiteStore("./chat.db")),
)

# First turn
r1 = await agent.arun("My name is Alice", session_id="s1")
# Second turn — agent remembers the name
r2 = await agent.arun("What's my name?", session_id="s1")
```

### Auto-Summarization

When conversation history exceeds `max_messages`, Memory automatically summarizes the middle section (preserving the first `pin_count` and last `recent_count` messages):

```python
memory = Memory(
  store=SQLiteStore("./chat.db"),
  max_messages=50,  # Trigger summarization at 50 entries
  pin_count=2,  # Keep first 2 messages
  recent_count=5,  # Keep last 5 messages
)
# model is set automatically from agent.model at runtime
```

```
Before optimization (55 entries):
  [msg1, msg2, msg3, msg4, ..., msg50, msg51, msg52, msg53, msg54, msg55]
   ├─ pin ─┤  ├────────── summarize ──────────┤  ├───── keep recent ─────┤

After optimization (8 entries):
  [msg1, msg2, summary, msg51, msg52, msg53, msg54, msg55]
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `session_id` alone doesn't maintain history | You need Memory or pass `messages=r1.messages` |
| `memory=True` creates InMemoryStore | Data is lost when process exits. Use SQLiteStore for persistence |
| `SQLiteStore` requires aiosqlite | `pip install aiosqlite` |
| Auto-summarization requires a model | Set on Memory or it uses the Agent's model automatically |

## Related Modules

- **[Agent](../agent/README.md)** — Memory snaps into Agent via `memory=` parameter
- **[Knowledge](../../knowledge/README.md)** — Long-term document storage (different from session memory)
- **[Model](../../model/README.md)** — Required for auto-summarization
