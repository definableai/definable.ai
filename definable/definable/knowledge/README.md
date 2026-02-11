# knowledge

RAG (Retrieval-Augmented Generation) pipeline — documents, chunkers, embedders, vector databases, and rerankers.

## Quick Start

```python
from definable.agents import Agent
from definable.knowledge import Knowledge, InMemoryVectorDB, TextChunker
from definable.knowledge import OpenAIEmbedder

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  embedder=OpenAIEmbedder(),
  chunker=TextChunker(chunk_size=500),
)

# Add documents from files, URLs, or raw text
knowledge.add("docs/guide.pdf")
knowledge.add("https://example.com/faq")

# Use with an agent
agent = Agent(
  model=model,
  knowledge=knowledge,
  instructions="Answer using the provided context.",
)
response = agent.run("How do I get started?")
```

## Module Structure

```
knowledge/
├── __init__.py          # Public API (implementations lazy-loaded)
├── base.py              # Knowledge orchestrator
├── document/
│   └── base.py          # Document dataclass
├── chunkers/
│   ├── base.py          # Chunker ABC
│   ├── text.py          # TextChunker (separator-based)
│   └── recursive.py     # RecursiveChunker (hierarchical)
├── embedders/
│   ├── base.py          # Embedder ABC
│   ├── openai.py        # OpenAIEmbedder
│   └── voyageai.py      # VoyageAIEmbedder
├── readers/
│   ├── base.py          # Reader ABC, ReaderConfig
│   ├── text.py          # TextReader (.txt, .md, .csv, .log, .rst)
│   ├── pdf.py           # PDFReader (requires pypdf)
│   └── url.py           # URLReader (web pages)
├── rerankers/
│   ├── base.py          # Reranker ABC
│   └── cohere.py        # CohereReranker
└── vector_dbs/
    ├── base.py          # VectorDB ABC
    ├── memory.py        # InMemoryVectorDB (pure Python cosine similarity)
    └── pgvector.py      # PgVectorDB (PostgreSQL + pgvector)
```

## API Reference

### Knowledge

The main orchestrator. Manages the pipeline: Source -> Reader -> Chunker -> Embedder -> VectorDB.

```python
from definable.knowledge import Knowledge

kb = Knowledge(
  vector_db=InMemoryVectorDB(),
  embedder=OpenAIEmbedder(),
  chunker=TextChunker(),
  readers=[TextReader(), PDFReader()],
)
```

| Method | Description |
|--------|-------------|
| `add(source, reader=None, chunk=True)` | Add a file, URL, or Document (sync) |
| `aadd(source, ...)` | Async version |
| `search(query, top_k=10, rerank=True, filter=None)` | Search and optionally rerank (sync) |
| `asearch(query, ...)` | Async version |
| `remove(ids)` / `aremove(ids)` | Remove documents by ID |
| `clear()` | Remove all documents |
| `__len__()` | Total document count |

### Document

```python
from definable.knowledge import Document
```

Multimodal document with embedding and metadata.

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Document text |
| `id` | `Optional[str]` | Unique identifier (auto-generated) |
| `name` | `Optional[str]` | Display name |
| `meta_data` | `Dict` | Arbitrary metadata |
| `embedding` | `Optional[List[float]]` | Vector embedding |
| `source` | `Optional[str]` | Origin (file path, URL) |
| `chunk_index` / `chunk_total` | `Optional[int]` | Position in chunked sequence |
| `parent_id` | `Optional[str]` | ID of parent document |

### Chunkers

```python
from definable.knowledge import Chunker, TextChunker, RecursiveChunker
```

| Class | Description |
|-------|-------------|
| `Chunker` | ABC with `chunk_size` and `chunk_overlap` fields |
| `TextChunker(separator="\n\n")` | Splits on a single separator |
| `RecursiveChunker(separators=[...])` | Hierarchical splitting: tries separators in order |

### Embedders

```python
from definable.knowledge import Embedder, OpenAIEmbedder, VoyageAIEmbedder
```

| Class | Default Model | Env Var |
|-------|--------------|---------|
| `OpenAIEmbedder` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| `VoyageAIEmbedder` | `voyage-2` | `VOYAGE_API_KEY` |

Both provide `get_embedding(text)`, `async_get_embedding(text)`, and batch variants.

### Readers

```python
from definable.knowledge import Reader, TextReader, PDFReader, URLReader
```

| Class | Supported Formats | Dependencies |
|-------|-------------------|--------------|
| `TextReader` | `.txt`, `.md`, `.csv`, `.log`, `.rst` | None |
| `PDFReader` | `.pdf` | `pypdf` |
| `URLReader` | HTTP/HTTPS URLs | `httpx` |

### Rerankers

```python
from definable.knowledge import Reranker, CohereReranker
```

| Class | Default Model | Env Var |
|-------|--------------|---------|
| `CohereReranker` | `rerank-multilingual-v3.0` | `COHERE_API_KEY` |

### Vector Databases

```python
from definable.knowledge import VectorDB, InMemoryVectorDB, PgVectorDB
```

| Class | Description | Dependencies |
|-------|-------------|--------------|
| `InMemoryVectorDB` | Pure Python cosine similarity, no persistence | None |
| `PgVectorDB` | PostgreSQL with pgvector extension | `psycopg` |

Both implement: `add(docs)`, `aadd(docs)`, `search(embedding, top_k, filter)`, `asearch(...)`, `delete(ids)`, `clear()`, `count()`.

## Usage with Agent

Knowledge can be integrated with an agent in two ways:

**Automatic retrieval** (via middleware — recommended):

```python
agent = Agent(model=model, knowledge=knowledge)
# Knowledge context is automatically retrieved and injected into the system prompt.
```

**Explicit toolkit** (agent decides when to search):

```python
from definable.agents import KnowledgeToolkit

agent = Agent(
  model=model,
  toolkits=[KnowledgeToolkit(knowledge=knowledge)],
)
# Agent calls search_knowledge() tool when it decides retrieval is needed.
```

## See Also

- `agents/` — Agent integration with `knowledge=` parameter
- `agents/middleware.py` — `KnowledgeMiddleware` for automatic RAG
- `agents/toolkits/knowledge.py` — `KnowledgeToolkit` for explicit RAG
- `readers/` — Advanced file parsing (separate from knowledge readers)
- `vectordbs/` — Legacy vector DB interface (superseded by `knowledge/vector_dbs/`)
