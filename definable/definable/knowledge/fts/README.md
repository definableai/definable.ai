# knowledge/fts

Full-text keyword search backed by SQLite FTS5. Complements vector similarity search by catching exact keyword matches that embeddings miss — proper nouns, version strings, error codes, identifiers. Used standalone or merged with vector results through `HybridSearcher`.

## Module structure

```
fts/
├── __init__.py    # Public API
├── index.py       # FTSIndex (SQLite FTS5)
├── hybrid.py      # HybridSearcher (RRF + weighted merge)
└── keywords.py    # extract_keywords utility
```

## Quick start

```python
from definable.knowledge.fts import FTSIndex, HybridSearcher, HybridSearchConfig
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB

# Standalone FTS
fts = FTSIndex()
await fts.initialize()  # REQUIRED — creates the FTS5 table

docs = [Document(content="Machine learning fundamentals")]
await fts.add("batch-001", docs)

results = await fts.search("machine learning", limit=10)
# [(doc_id, bm25_score, content), ...]

await fts.close()
```

```python
# Hybrid search attached to Knowledge
from definable.knowledge.fts import FTSIndex, HybridSearchConfig

fts = FTSIndex()
await fts.initialize()

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  fts_index=fts,
  hybrid_config=HybridSearchConfig(
    vector_weight=0.6,
    text_weight=0.4,
    merge_strategy="rrf",
  ),
)
# Knowledge.aadd() indexes into FTS automatically
await knowledge.aadd(docs)
```

## API reference

### FTSIndex

```python
from definable.knowledge.fts import FTSIndex

fts = FTSIndex(
  db_path=None,              # str | None — path to .db file, None = in-memory
  table_name="fts_documents" # str — FTS5 virtual table name
)
```

`db_path=None` stores the index in memory (lost on process exit). Pass a file path for persistence across restarts.

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize` | `async () -> None` | Create FTS5 table. Must be called before any other method. |
| `add` | `async (content_hash: str, documents: list[Document]) -> int` | Index documents. Returns count added. |
| `search` | `async (query: str, limit: int = 20) -> list[tuple[str, float, str]]` | Returns `(doc_id, bm25_score, content)` tuples. |
| `search_documents` | `async (query: str, limit: int = 20) -> list[Document]` | Like `search()` but returns `Document` objects with `reranking_score` set. |
| `delete` | `async (content_hash: str) -> int` | Remove documents by batch hash. Returns count deleted. |
| `clear` | `async () -> None` | Remove all indexed documents. |
| `count` | `async () -> int` | Total number of indexed documents. |
| `close` | `async () -> None` | Close the SQLite connection. |

**BM25 score normalization**

SQLite FTS5 returns negative BM25 ranks (lower = better). `FTSIndex` converts them to a positive score with `1.0 / (1.0 + abs(rank))`. This maps into the range `(0, 1]`.

### HybridSearchConfig

```python
from definable.knowledge.fts import HybridSearchConfig

config = HybridSearchConfig(
  vector_weight=0.6,        # float — weight for vector scores
  text_weight=0.4,          # float — weight for BM25 scores
  merge_strategy="rrf",     # "rrf" | "weighted"
  rrf_k=60,                 # int — RRF smoothing constant
  fts_fetch_multiplier=3,   # int — fetch limit * multiplier from FTS for coverage
)
```

**Merge strategies**

- `"rrf"` (default) — Reciprocal Rank Fusion. Score = `sum(weight / (k + rank))`. Robust to score scale differences between vector and BM25 results. Preferred for most use cases.
- `"weighted"` — Normalizes both score sets to `[0, 1]` then combines as `vector_weight * v_norm + text_weight * f_norm`. More sensitive to score distribution but preserves relative magnitude.

Weights do not need to sum to 1.0, though keeping them near 1.0 makes tuning intuitive.

### HybridSearcher

```python
from definable.knowledge.fts import HybridSearcher, HybridSearchConfig, FTSIndex

fts = FTSIndex()
await fts.initialize()

searcher = HybridSearcher(
  fts_index=fts,
  config=HybridSearchConfig(),  # optional, defaults shown above
)

merged = await searcher.merge(
  vector_results=vector_docs,  # list[Document] from vector DB
  query="machine learning",    # str — passed to FTS search
  limit=10,                    # int — max results returned
)
```

`merge()` runs an FTS search internally (`fts_fetch_multiplier * limit` candidates), then merges with the provided vector results. Documents appearing in both result sets receive contributions from both retrieval paths.

### extract_keywords

```python
from definable.knowledge.fts import extract_keywords

keywords = extract_keywords("hello world machine learning")
# ["hello", "world", "machine", "learning"]

keywords = extract_keywords("What is the best approach?")
# ["best", "approach"]  — stop words removed

keywords = extract_keywords("gpt-4o release notes", max_keywords=5)
# ["gpt-4o", "release", "notes"]  — hyphens preserved
```

Removes English stop words, lowercases tokens, and preserves hyphenated terms (e.g. `gpt-4o`, `step-by-step`). Useful for debugging what terms the FTS index will match against. The `max_keywords` parameter caps extraction at 10 by default.

## Integration with Knowledge

When `fts_index` and `hybrid_config` are both set on `Knowledge`, the full retrieval pipeline becomes:

```
query
  → vector search (top_k * fts_fetch_multiplier candidates)
  → FTS keyword search
  → HybridSearcher.merge()
  → optional Reranker
  → optional TemporalDecay
  → optional MMR diversity
  → top_k results returned
```

`Knowledge.aadd()` automatically calls `FTSIndex.add()` in parallel with vector insertion, so documents are always kept in sync between the two indexes.

```python
from definable.knowledge import Knowledge, Document
from definable.knowledge.fts import FTSIndex, HybridSearchConfig
from definable.vectordb import InMemoryVectorDB

fts = FTSIndex(db_path="./search.db")  # persisted
await fts.initialize()

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  fts_index=fts,
  hybrid_config=HybridSearchConfig(merge_strategy="rrf"),
  top_k=5,
)

await knowledge.aadd([
  Document(content="Python asyncio tutorial", meta_data={"source": "docs.py"}),
  Document(content="FastAPI routing guide",   meta_data={"source": "fastapi.md"}),
])

# Both vector and keyword paths are searched automatically
results = await knowledge.asearch("asyncio event loop")
```

## Persistent vs in-memory index

| `db_path` | Behavior |
|-----------|----------|
| `None` (default) | In-memory SQLite. Fast, no I/O. Lost on process exit. |
| `"./fts.db"` | File-backed SQLite with WAL mode. Survives restarts. |

Use a persistent path when your knowledge base is large or expensive to reindex. Use in-memory for unit tests or ephemeral pipelines.

## Gotchas

**`initialize()` is not optional.** Any method called before `initialize()` raises `RuntimeError: FTSIndex not initialized`. This includes `add`, `search`, `count`, `delete`, and `clear`.

```python
# WRONG
fts = FTSIndex()
await fts.add("h", docs)  # RuntimeError

# RIGHT
fts = FTSIndex()
await fts.initialize()
await fts.add("h", docs)
```

**FTS5 must be available.** Standard CPython builds on macOS and Linux include FTS5. If you see `RuntimeError: SQLite FTS5 extension is not available`, your Python was compiled without it (rare with official installers).

**`content_hash` is a batch identifier, not a document ID.** `FTSIndex.add()` takes a `content_hash` string that tags the entire batch. `FTSIndex.delete(content_hash)` removes all documents added under that hash. Individual document deletion is not supported — clear and re-add the batch instead.

**Stop words are always filtered.** The query `"the machine"` searches for `machine` only. Short or purely stop-word queries return no results. Use `extract_keywords()` to preview what terms will be submitted.

**Deduplication uses content prefix.** `HybridSearcher` deduplicates by the first 200 characters of `Document.content`. Documents with identical openings are treated as the same document during merging.
