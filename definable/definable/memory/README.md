# memory

Multi-tier cognitive memory with retrieval, distillation, and topic prediction.

## Installation

The default SQLite backend has no extra dependencies. Other backends require optional extras:

```bash
pip install 'definable[postgres-memory]'   # PostgreSQL (asyncpg + pgvector)
pip install 'definable[redis-memory]'      # Redis
pip install 'definable[qdrant-memory]'     # Qdrant
pip install 'definable[chroma-memory]'     # Chroma
pip install 'definable[mongodb-memory]'    # MongoDB (motor)
pip install 'definable[pinecone-memory]'   # Pinecone
pip install 'definable[mem0-memory]'      # Mem0
```

## Quick Start

```python
from definable.agents import Agent
from definable.memory import CognitiveMemory, SQLiteMemoryStore
from definable.knowledge import OpenAIEmbedder

memory = CognitiveMemory(
  store=SQLiteMemoryStore(db_path="./memory.db"),
  embedder=OpenAIEmbedder(),
  token_budget=500,
)

agent = Agent(
  model=model,
  memory=memory,
  instructions="You remember past conversations.",
)

# Memory is auto-recalled before each run and auto-stored after.
response = agent.run("Remember that I prefer dark mode.")
response = agent.run("What are my preferences?")
```

## Module Structure

```
memory/
├── __init__.py        # Public API (backends lazy-loaded)
├── memory.py          # CognitiveMemory orchestrator
├── config.py          # MemoryConfig, ScoringWeights
├── types.py           # Episode, KnowledgeAtom, Procedure, TopicTransition, MemoryPayload
├── retrieval.py       # recall_memories() — 5-path retrieval pipeline
├── distillation.py    # run_distillation() — progressive compression (stages 0-3)
├── scorer.py          # 5-factor composite relevance scorer
├── topics.py          # Topic extraction and transition prediction
└── store/
    ├── __init__.py    # MemoryStore protocol (backends lazy-loaded)
    ├── base.py        # MemoryStore Protocol
    ├── _utils.py      # cosine_similarity utility
    ├── sqlite.py      # SQLiteMemoryStore
    ├── in_memory.py   # InMemoryStore
    ├── postgres.py    # PostgresMemoryStore
    ├── redis.py       # RedisMemoryStore
    ├── qdrant.py      # QdrantMemoryStore
    ├── chroma.py      # ChromaMemoryStore
    ├── mongodb.py     # MongoMemoryStore
    ├── pinecone.py    # PineconeMemoryStore
    └── mem0.py        # Mem0MemoryStore
```

## API Reference

### CognitiveMemory

```python
from definable.memory import CognitiveMemory
```

The main orchestrator for memory operations.

```python
memory = CognitiveMemory(
  store=SQLiteMemoryStore(),
  token_budget=500,                  # Max tokens for recall context
  embedder=OpenAIEmbedder(),         # Optional; enables semantic search
  distillation_model=model,          # Optional; enables LLM-powered distillation
  config=MemoryConfig(),             # Optional; fine-tune scoring and distillation
)
```

| Method | Description |
|--------|-------------|
| `recall(query, user_id=, session_id=)` | Retrieve relevant memories as `MemoryPayload` |
| `store_messages(messages, user_id=, session_id=)` | Store conversation messages as episodes |
| `run_distillation(user_id=)` | Compress old episodes into knowledge atoms |
| `forget(user_id=, session_id=)` | Delete user or session data |
| `close()` | Close the underlying store |

### MemoryConfig

```python
from definable.memory import MemoryConfig, ScoringWeights
```

| Field | Default | Description |
|-------|---------|-------------|
| `decay_half_life_days` | `14.0` | Recency decay half-life |
| `scoring_weights` | `ScoringWeights()` | Weight for each scoring factor |
| `distillation_stage_0_age` | `3600.0` | Age (seconds) before stage 0 -> 1 |
| `distillation_stage_1_age` | `86400.0` | Age before stage 1 -> 2 |
| `distillation_stage_2_age` | `604800.0` | Age before stage 2 -> 3 |
| `distillation_stage_3_age` | `2592000.0` | Age before stage 3 -> archive |
| `retrieval_top_k` | `20` | Candidates per retrieval path |
| `recent_episodes_limit` | `5` | Recent session episodes to include |

**ScoringWeights** — 5-factor composite relevance scoring:

| Factor | Default | Description |
|--------|---------|-------------|
| `semantic_similarity` | `0.35` | Cosine similarity to query embedding |
| `recency` | `0.25` | Exponential decay from last access |
| `access_frequency` | `0.15` | How often the memory is accessed |
| `predicted_need` | `0.15` | Topic transition prediction overlap |
| `emotional_salience` | `0.10` | Absolute sentiment value |

### Types

```python
from definable.memory import (
  Episode,           # A conversation turn (raw -> summary -> facts -> atoms)
  KnowledgeAtom,     # An extracted fact as subject-predicate-object triple
  Procedure,         # A learned behavioral pattern (trigger -> action)
  TopicTransition,   # A transition between topics with probability
  MemoryPayload,     # Result of recall — context string + token count
)
```

### Store Backends

```python
from definable.memory import MemoryStore  # Protocol
```

All backends implement the `MemoryStore` protocol with methods for episodes, atoms, procedures, topics, vector search, and deletion.

| Backend | Constructor | Features |
|---------|-------------|----------|
| `SQLiteMemoryStore` | `(db_path="./memory.db")` | Default; aiosqlite; in-Python vector search |
| `InMemoryStore` | `()` | Dev/test; no persistence |
| `PostgresMemoryStore` | `(db_url=, pool_size=5)` | asyncpg + pgvector native vector ops |
| `RedisMemoryStore` | `(redis_url=, prefix=, db=)` | Hashes + sorted sets; optional RediSearch |
| `QdrantMemoryStore` | `(url=, port=, api_key=, vector_size=)` | Native cosine search |
| `ChromaMemoryStore` | `(persist_directory=, collection_prefix=)` | Sync client wrapped in asyncio.to_thread |
| `MongoMemoryStore` | `(connection_string=, database=)` | motor; Python-side vector search |
| `PineconeMemoryStore` | `(api_key=, index_name=, vector_size=)` | Managed service; v5 SDK |
| `Mem0MemoryStore` | `(api_key=, org_id=, project_id=)` | Hosted Mem0 API; no raw embedding search |

## Usage with Agent

```python
agent = Agent(model=model, memory=memory)

# Memory is automatically:
# 1. Recalled before each arun() — injected into RunContext.memory_context
# 2. Stored after each arun() — fire-and-forget via asyncio.create_task()
# All memory failures are non-fatal (logged, never raised).
```

## See Also

- `agents/` — Agent integration via `memory=` parameter
- `knowledge/` — RAG pipeline (complementary to memory)
- `run/` — `RunContext.memory_context` field
