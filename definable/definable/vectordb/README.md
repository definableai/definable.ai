# VectorDB

> Unified vector database interface with pluggable backends for Definable AI.

VectorDB provides a common abstraction over vector storage backends. Documents arrive pre-embedded from the Knowledge layer. The embedder on VectorDB is used only for embedding **search queries**.

## Quick Start

```python
from definable.vectordb import InMemoryVectorDB, Distance
from definable.knowledge.document import Document
from definable.knowledge.embedder.openai import OpenAIEmbedder

# Create an embedder and database
embedder = OpenAIEmbedder()
db = InMemoryVectorDB(name="my_docs", embedder=embedder)
db.create()

# Prepare documents with embeddings
docs = [
  Document(content="Python is a programming language", meta_data={"source": "wiki"}),
  Document(content="JavaScript runs in the browser", meta_data={"source": "wiki"}),
  Document(content="Rust is a systems programming language", meta_data={"source": "blog"}),
]
for doc in docs:
  doc.embedding = embedder.get_embedding(doc.content)

# Insert and search
db.insert(docs)
results = db.search("systems programming", limit=2)
print(results[0].content)  # "Rust is a systems programming language"
```

## Architecture

```
VectorDB (ABC)
  │
  ├── InMemoryVectorDB ─── numpy cosine similarity, no external deps
  ├── PgVector ──────────── PostgreSQL + pgvector extension
  ├── Qdrant ────────────── Qdrant vector search engine
  ├── ChromaDb ──────────── ChromaDB embedded/client
  ├── MongoDb ───────────── MongoDB Atlas vector search
  ├── RedisDB ───────────── Redis + RediSearch module
  └── PineconeDb ────────── Pinecone managed cloud service
```

### Module Structure

```
vectordb/
├── __init__.py           # Public API: VectorDB, Distance, SearchType, all backends
├── base.py               # VectorDB abstract base class
├── distance.py           # Distance enum (cosine, l2, max_inner_product)
├── search.py             # SearchType enum (vector, keyword, hybrid)
├── memory/
│   └── memory.py         # InMemoryVectorDB
├── pgvector/
│   └── pgvector.py       # PgVector
├── qdrant/
│   └── qdrant.py         # Qdrant
├── chroma/
│   └── chromadb.py       # ChromaDb
├── mongodb/
│   └── mongodb.py        # MongoDb
├── redis/
│   └── redisdb.py        # RedisDB
└── pineconedb/
    └── pineconedb.py     # PineconeDb
```

## API Reference

### VectorDB (Base Class)

All backends implement these methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `create` | `db.create()` | Initialize the collection/table |
| `insert` | `db.insert(docs)` or `db.insert(hash, docs)` | Insert pre-embedded documents |
| `search` | `db.search(query, limit=5, filters=None)` | Search by text query |
| `upsert` | `db.upsert(hash, docs)` | Insert or replace by content hash |
| `drop` | `db.drop()` | Delete all stored documents |
| `delete` | `db.delete()` | Delete the collection |
| `delete_by_id` | `db.delete_by_id(doc_id)` | Delete a specific document |
| `delete_by_name` | `db.delete_by_name(name)` | Delete by document name |
| `delete_by_metadata` | `db.delete_by_metadata({"key": "val"})` | Delete by metadata match |
| `get_count` | `db.get_count()` | Number of stored documents |
| `exists` | `db.exists()` | Check if collection exists |

**Async variants:** Every method has an async counterpart — `async_create`, `async_insert`, `async_search`, etc. Plus convenience aliases: `ainsert`, `asearch`.

### Distance

```python
from definable.vectordb import Distance

Distance.cosine  # Cosine similarity (default)
Distance.l2  # Euclidean distance
Distance.max_inner_product  # Dot product
```

### SearchType

```python
from definable.vectordb import SearchType

SearchType.vector  # Pure vector similarity
SearchType.keyword  # BM25 keyword search
SearchType.hybrid  # Combined vector + keyword
```

## Backend Implementations

### InMemoryVectorDB

No external dependencies beyond numpy. Best for testing and small datasets.

```python
from definable.vectordb import InMemoryVectorDB
from definable.knowledge.embedder.openai import OpenAIEmbedder

db = InMemoryVectorDB(
  name="my_collection",  # Collection name
  embedder=OpenAIEmbedder(),  # For embedding search queries
  distance=Distance.cosine,  # Similarity metric
)
db.create()  # No-op for in-memory, but call for API consistency
```

> **Important:** Pass `embedder=` explicitly. If omitted, defaults to `OpenAIEmbedder()` which requires `OPENAI_API_KEY`.

### PgVector

PostgreSQL with the pgvector extension.

```python
from definable.vectordb import PgVector

db = PgVector(
  name="documents",
  db_url="postgresql://user:pass@localhost:5432/mydb",
  embedder=embedder,
)
```

> **Requires:** `pip install pgvector asyncpg`

### Qdrant

Qdrant vector search engine (local or cloud).

```python
from definable.vectordb import Qdrant

db = Qdrant(
  name="documents",
  url="http://localhost:6333",
  embedder=embedder,
)
```

> **Requires:** `pip install qdrant-client`

### ChromaDb

ChromaDB embedded or client mode.

```python
from definable.vectordb import ChromaDb

db = ChromaDb(
  name="documents",
  embedder=embedder,
)
```

> **Requires:** `pip install chromadb`

### MongoDb

MongoDB Atlas vector search.

```python
from definable.vectordb import MongoDb

db = MongoDb(
  name="documents",
  connection_string="mongodb+srv://...",
  embedder=embedder,
)
```

> **Requires:** `pip install pymongo`

### RedisDB

Redis with the RediSearch module.

```python
from definable.vectordb import RedisDB

db = RedisDB(
  name="documents",
  url="redis://localhost:6379",
  embedder=embedder,
)
```

> **Requires:** `pip install redis`

### PineconeDb

Pinecone managed cloud vector service.

```python
from definable.vectordb import PineconeDb

db = PineconeDb(
  name="documents",
  api_key="your-api-key",
  embedder=embedder,
)
```

> **Requires:** `pip install pinecone-client`

## Patterns & Recipes

### Insert with Content Hash (Deduplication)

```python
db.insert("hash123", docs)  # Explicit content hash
db.insert(docs)  # Auto-generates hash from content
```

### Metadata Filtering

```python
results = db.search("query", limit=5, filters={"source": "blog"})
```

### Async Usage

```python
import asyncio
from definable.vectordb import InMemoryVectorDB


async def main():
  db = InMemoryVectorDB(embedder=embedder)
  await db.async_create()
  await db.ainsert(docs)
  results = await db.asearch("search query", limit=3)
  print(len(results))


asyncio.run(main())
```

### With Knowledge Layer

In typical usage, you don't interact with VectorDB directly — the Knowledge layer manages embedding, chunking, and insertion:

```python
from definable.agent import Agent
from definable.knowledge import Knowledge
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  top_k=5,
)
agent = Agent(model="openai/gpt-4o-mini", knowledge=knowledge)
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `InMemoryVectorDB()` without embedder | Defaults to `OpenAIEmbedder()` — needs `OPENAI_API_KEY` |
| `dimensions=N` kwarg | Deprecated and ignored. Pass `embedder=` instead |
| Documents must have embeddings | The Knowledge layer handles this. For direct use, embed manually |
| `db.search()` returns `Document` objects | Access `.content`, `.meta_data`, `.embedding` |
| `Document(metadata={})` | Wrong — use `meta_data` (with underscore) |
| `knowledge=True` on Agent | Raises ValueError. Must provide `vector_db` |

## Related Modules

- **[Knowledge](../../knowledge/README.md)** — High-level RAG pipeline that uses VectorDB
- **[Embedders](../../knowledge/embedder/README.md)** — Text → vector conversion
- **[Document](../../knowledge/document/)** — The Document data class stored in VectorDB
