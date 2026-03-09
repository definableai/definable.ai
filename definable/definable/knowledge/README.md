# Knowledge

> RAG (Retrieval-Augmented Generation) pipeline — documents, chunkers, embedders, vector databases, rerankers, full-text search, and scoring.

## Quick Start

```python
from definable.agent import Agent
from definable.knowledge import Knowledge, Document
from definable.knowledge.embedder import OpenAIEmbedder
from definable.knowledge.chunker import RecursiveChunker
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  embedder=OpenAIEmbedder(),
  chunker=RecursiveChunker(),
)

# Add documents from files, URLs, or raw text
knowledge.add("docs/guide.pdf")
knowledge.add("https://example.com/faq")

# Use with an agent
agent = Agent(
  model="openai/gpt-4o-mini",
  knowledge=knowledge,
  instructions="Answer using the provided context.",
)
result = await agent.arun("How do I get started?")
```

## Architecture

```
Knowledge (orchestrator)
  │
  ├── vector_db: VectorDB ─── stores document embeddings
  ├── embedder: Embedder ──── text → vector conversion
  ├── chunker: Chunker ────── splits documents into chunks
  ├── reranker: Reranker ──── re-scores search results (optional)
  ├── readers: [Reader] ───── parse files (PDF, text, URL, JSON)
  │
  ├── fts_index: FTSIndex ───── SQLite FTS5 full-text search (optional)
  ├── hybrid_config: HybridSearchConfig ─── vector + text merge (optional)
  ├── temporal_decay: TemporalDecay ─────── score decay by age (optional)
  └── mmr: MMRConfig ──────────────────────── diversity reranking (optional)

Search Pipeline (when all features enabled):
  Query → Vector Search → Hybrid Merge (FTS + Vector) → Reranker → Temporal Decay → MMR → Results
```

### Module Structure

```
knowledge/
├── __init__.py              # Public API (lazy-loaded)
├── base.py                  # Knowledge orchestrator
├── document/
│   └── base.py              # Document dataclass
├── chunker/
│   ├── __init__.py          # Chunker, RecursiveChunker, TextChunker, MarkdownChunker, SemanticChunker
│   ├── base.py              # Chunker ABC
│   ├── text.py              # TextChunker (separator-based)
│   ├── recursive.py         # RecursiveChunker (hierarchical)
│   ├── markdown.py          # MarkdownChunker (heading-split)
│   └── semantic.py          # SemanticChunker (embedding-based)
├── embedder/
│   ├── __init__.py          # Embedder, OpenAI, VoyageAI, Google, Mistral, Fallback
│   ├── base.py              # Embedder ABC
│   ├── openai.py            # OpenAIEmbedder
│   ├── voyageai.py          # VoyageAIEmbedder
│   ├── google.py            # GoogleEmbedder
│   ├── mistral.py           # MistralEmbedder
│   └── fallback.py          # FallbackEmbedder (provider chain)
├── reader/
│   ├── text.py              # TextReader
│   ├── pdf.py               # PDFReader
│   ├── url.py               # URLReader
│   └── json_reader.py       # JSONReader
├── reranker/
│   ├── __init__.py          # Reranker, CohereReranker, SentenceTransformerReranker
│   ├── base.py              # Reranker ABC
│   ├── cohere.py            # CohereReranker
│   └── sentence_transformer.py  # SentenceTransformerReranker (local cross-encoder)
├── fts/
│   ├── __init__.py          # FTSIndex, HybridSearchConfig, HybridSearcher
│   ├── index.py             # FTSIndex (SQLite FTS5)
│   ├── hybrid.py            # HybridSearcher (RRF + weighted merge)
│   └── keywords.py          # Keyword extraction
└── scoring/
    ├── __init__.py          # TemporalDecay, MMRConfig, mmr_rerank
    ├── temporal.py          # TemporalDecay (exponential, evergreen exempt)
    └── mmr.py               # MMR (cosine + jaccard fallback)
```

## API Reference

### Knowledge

The main orchestrator. Manages: Source → Reader → Chunker → Embedder → VectorDB.

```python
from definable.knowledge import Knowledge

kb = Knowledge(
  vector_db=InMemoryVectorDB(),  # Required — where embeddings live
  embedder=OpenAIEmbedder(),  # Optional — defaults to OpenAIEmbedder
  chunker=RecursiveChunker(),  # Optional — defaults to RecursiveChunker
  readers=[],  # Optional — auto-detected by file type
  reranker=None,  # Optional — re-score results
  top_k=10,  # Number of results to return
  trigger="auto",  # "always" | "auto" | "never"
  # Advanced scoring (optional)
  fts_index=None,  # FTSIndex for full-text search
  hybrid_config=None,  # HybridSearchConfig for vector + text merge
  temporal_decay=None,  # TemporalDecay for age-based scoring
  mmr=None,  # MMRConfig for diversity reranking
)
```

| Method | Description |
|--------|-------------|
| `add(source, reader=None, chunk=True)` | Add a file, URL, or Document (sync) |
| `aadd(source, ...)` | Async version (also auto-indexes FTS) |
| `search(query, top_k=10, rerank=True)` | Search pipeline (sync) |
| `asearch(query, ...)` | Async search pipeline |

### Document

```python
from definable.knowledge import Document

doc = Document(
  content="Hello world",
  meta_data={"source": "wiki"},  # NOTE: meta_data, NOT metadata
)
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Document text |
| `id` | `str` | Unique identifier (auto-generated) |
| `name` | `str` | Display name |
| `meta_data` | `dict` | Arbitrary metadata (**NOT** `metadata`) |
| `embedding` | `list[float]` | Vector embedding |
| `source` | `str` | Origin (file path, URL) |

## Chunkers

```python
from definable.knowledge.chunker import (
  RecursiveChunker,  # Hierarchical splitting (default)
  TextChunker,  # Single-separator splitting
  MarkdownChunker,  # Heading-based splitting
  SemanticChunker,  # Embedding-based boundary detection
)
```

| Chunker | Best For | Key Config |
|---------|----------|------------|
| `RecursiveChunker` | General text | `chunk_size=500, chunk_overlap=50, separators=[...]` |
| `TextChunker` | Simple splits | `chunk_size=500, separator="\n\n"` |
| `MarkdownChunker` | Markdown docs | `max_heading_depth=3, preserve_code_blocks=True` |
| `SemanticChunker` | Content-aware | `embedder=..., threshold=0.5, window_size=3` |

## Embedders

```python
from definable.knowledge.embedder import (
  OpenAIEmbedder,  # text-embedding-3-small (default)
  VoyageAIEmbedder,  # voyage-2
  GoogleEmbedder,  # text-embedding-004 (requires google-genai)
  MistralEmbedder,  # mistral-embed (requires mistralai)
  FallbackEmbedder,  # Multi-provider failover chain
)
```

| Embedder | Default Model | Env Var | Dims |
|----------|--------------|---------|------|
| `OpenAIEmbedder` | `text-embedding-3-small` | `OPENAI_API_KEY` | 1536 |
| `VoyageAIEmbedder` | `voyage-2` | `VOYAGEAI_API_KEY` | 1024 |
| `GoogleEmbedder` | `text-embedding-004` | `GOOGLE_API_KEY` | 768 |
| `MistralEmbedder` | `mistral-embed` | `MISTRAL_API_KEY` | 1024 |

All provide: `get_embedding(text) -> list[float]`, `get_embedding_and_usage(text)`, and async variants.

### FallbackEmbedder

Chain of embedders — automatically switches to the next on failure:

```python
from definable.knowledge.embedder import FallbackEmbedder, OpenAIEmbedder, VoyageAIEmbedder

embedder = FallbackEmbedder(
  providers=[
    OpenAIEmbedder(),  # Primary
    VoyageAIEmbedder(),  # Fallback
  ]
)
```

> **Gotcha:** `FallbackEmbedder(providers=[])` raises ValueError. At least one provider required.

## Rerankers

```python
from definable.knowledge.reranker import (
  CohereReranker,  # Cloud API reranking
  SentenceTransformerReranker,  # Local cross-encoder
)
```

| Reranker | Model | Env Var |
|----------|-------|---------|
| `CohereReranker` | `rerank-multilingual-v3.0` | `COHERE_API_KEY` |
| `SentenceTransformerReranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | None (local) |

## Readers

```python
from definable.knowledge import TextReader, PDFReader, URLReader
from definable.knowledge.reader.json_reader import JSONReader
```

| Reader | Formats | Dependencies |
|--------|---------|--------------|
| `TextReader` | `.txt`, `.md`, `.csv`, `.log`, `.rst` | None |
| `PDFReader` | `.pdf` | `pypdf` |
| `URLReader` | HTTP/HTTPS URLs | `httpx` |
| `JSONReader` | `.json` | None |

### JSONReader

```python
from definable.knowledge.reader.json_reader import JSONReader

reader = JSONReader(
  content_key="text",  # Extract content from this field
  metadata_keys=["title"],  # Extract these as metadata
  flatten=True,  # Flatten nested structures
)
```

## Full-Text Search (FTS)

SQLite FTS5-based keyword search for hybrid retrieval:

```python
from definable.knowledge import FTSIndex, HybridSearchConfig

# Create FTS index
fts = FTSIndex()
await fts.initialize()  # REQUIRED before use

# Use with Knowledge
knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  fts_index=fts,
  hybrid_config=HybridSearchConfig(
    method="rrf",  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    vector_weight=0.7,  # For weighted method
    text_weight=0.3,  # For weighted method
  ),
)
```

> **Gotcha:** Must call `await fts.initialize()` before any search/add operations.

## Scoring & Diversity

### TemporalDecay

Exponential score decay based on document age:

```python
from definable.knowledge import TemporalDecay

knowledge = Knowledge(
  vector_db=db,
  temporal_decay=TemporalDecay(
    half_life_days=30.0,  # Score halves every 30 days
    # Evergreen documents (meta_data.evergreen=True) are exempt
  ),
)
```

### MMR (Maximal Marginal Relevance)

Diversity reranking to avoid redundant results:

```python
from definable.knowledge import MMRConfig

knowledge = Knowledge(
  vector_db=db,
  mmr=MMRConfig(
    lambda_param=0.7,  # 0.0 = max diversity, 1.0 = pure relevance
  ),
)
```

## Agent Integration

### Automatic Retrieval (Recommended)

```python
agent = Agent(model="openai/gpt-4o-mini", knowledge=knowledge)
# Knowledge context is automatically injected into the system prompt
```

### Path Shorthand

```python
# Auto-configures InMemoryVectorDB + OpenAIEmbedder + RecursiveChunker
agent = Agent(model="openai/gpt-4o-mini", knowledge="./docs/")
```

### Full Pipeline Example

```python
from definable.knowledge import (
  Knowledge,
  FTSIndex,
  HybridSearchConfig,
  TemporalDecay,
  MMRConfig,
)
from definable.knowledge.embedder import OpenAIEmbedder, FallbackEmbedder, VoyageAIEmbedder
from definable.knowledge.chunker import MarkdownChunker
from definable.knowledge.reranker import CohereReranker
from definable.vectordb import InMemoryVectorDB

fts = FTSIndex()

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  embedder=FallbackEmbedder(providers=[OpenAIEmbedder(), VoyageAIEmbedder()]),
  chunker=MarkdownChunker(max_heading_depth=3),
  reranker=CohereReranker(),
  fts_index=fts,
  hybrid_config=HybridSearchConfig(method="rrf"),
  temporal_decay=TemporalDecay(half_life_days=30.0),
  mmr=MMRConfig(lambda_param=0.7),
  top_k=10,
)
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `Document(metadata={})` | Wrong — use `meta_data` (with underscore) |
| `knowledge=True` on Agent | Raises ValueError — must provide `vector_db` |
| `FTSIndex` without `initialize()` | Must call `await fts.initialize()` first |
| `FallbackEmbedder(providers=[])` | Raises ValueError — at least one provider required |
| InMemoryVectorDB without embedder | Defaults to OpenAIEmbedder (needs `OPENAI_API_KEY`) |

## Related Modules

- **[VectorDB](../../vectordb/README.md)** — Vector storage backends
- **[Agent](../../agent/README.md)** — Knowledge integrates via `knowledge=` parameter
- **[Memory](../../memory/README.md)** — Session memory (different from knowledge)
