# knowledge/chunker

Text chunking sub-module. Splits `Document` objects into smaller `Document` objects suitable for embedding and vector search. Chunking is the digestion step of the RAG pipeline — it turns large source documents into retrieval-sized units.

## Quick start

```python
from definable.knowledge.chunker import RecursiveChunker
from definable.knowledge import Document

chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(Document(content="Long document text..."))
# Returns List[Document], each with chunk_index and chunk_total set
```

Chunk multiple documents at once:

```python
chunks = chunker.chunk_many([doc1, doc2, doc3])
```

`Knowledge` uses `RecursiveChunker` by default when you pass a `chunker=` argument or rely on auto-configuration:

```python
from definable.knowledge import Knowledge
from definable.knowledge.chunker import MarkdownChunker
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  chunker=MarkdownChunker(chunk_size=800, max_heading_depth=2),
)
```

## Module structure

```
chunker/
├── __init__.py     # Public API (all implementations lazy-loaded)
├── base.py         # Chunker ABC (dataclass)
├── text.py         # TextChunker (separator-based)
├── recursive.py    # RecursiveChunker (hierarchical multi-separator)
├── markdown.py     # MarkdownChunker (heading-aware)
└── semantic.py     # SemanticChunker (embedding-based boundaries)
```

All implementations are lazy-loaded — importing from `__init__` does not import implementation modules until the class is first accessed.

## API reference

### `Chunker` (ABC, `base.py`)

Base dataclass. All implementations inherit from this.

```python
@dataclass
class Chunker(ABC):
  chunk_size: int = 1000
  chunk_overlap: int = 200

  def chunk(self, document: Document) -> List[Document]: ...
  def chunk_many(self, documents: List[Document]) -> List[Document]: ...
```

`chunk_many` is a concrete helper that calls `chunk` on each document and concatenates results. Override it only if batching provides a performance advantage (e.g., `SemanticChunker` with batched embeddings).

Every output `Document` carries:

| Field | Value |
|---|---|
| `content` | The chunk text |
| `parent_id` | `id` of the source document |
| `chunk_index` | 0-based position in the chunk sequence |
| `chunk_total` | Total number of chunks from this source |
| `source` | Copied from the source document |
| `source_type` | Copied from the source document |
| `meta_data` | Source `meta_data` merged with `{"chunk_index": int, "chunk_total": int}` |

### `TextChunker`

Simple separator-based chunker. Splits on a single separator string and merges parts to fill `chunk_size`. Fastest option for plain prose when structure does not matter.

```python
from definable.knowledge.chunker import TextChunker

chunker = TextChunker(
  chunk_size=1000,
  separator="\n\n",    # split on double newline (paragraph breaks)
  keep_separator=False,
)
chunks = chunker.chunk(document)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Maximum characters per chunk |
| `chunk_overlap` | `200` | Characters of overlap between consecutive chunks |
| `separator` | `"\n\n"` | String to split on |
| `keep_separator` | `False` | Whether to include the separator in output chunks |

When `chunk_overlap > 0`, the trailing `chunk_overlap` characters of each chunk are prepended to the next chunk.

### `RecursiveChunker`

Hierarchical chunker that tries separators in order: `["\n\n", "\n", ". ", " ", ""]`. If the primary separator does not produce chunks small enough, it recurses with the next separator. Falls back to hard character splitting if no separator works.

**Default and recommended choice** for unstructured text.

```python
from definable.knowledge.chunker import RecursiveChunker

chunker = RecursiveChunker(
  chunk_size=1000,
  chunk_overlap=200,
  separators=["\n\n", "\n", ". ", " ", ""],  # tried in order
)
chunks = chunker.chunk(document)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Maximum characters per chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Ordered list of separator strings |

An empty string `""` as the last separator triggers hard character splitting, ensuring chunks never exceed `chunk_size` regardless of content.

### `MarkdownChunker`

Structure-aware chunker for Markdown documents. Splits on headings first (configurable maximum depth), then falls back to paragraph splitting for large sections. Fenced code blocks are extracted before splitting and restored after, so they are never broken mid-block.

```python
from definable.knowledge.chunker import MarkdownChunker

chunker = MarkdownChunker(
  chunk_size=1000,
  chunk_overlap=200,
  max_heading_depth=3,      # split on H1, H2, H3 (not H4-H6)
  preserve_code_blocks=True,
)
chunks = chunker.chunk(document)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Maximum characters per chunk |
| `chunk_overlap` | `200` | Overlap for paragraph-level splits within large sections |
| `max_heading_depth` | `3` | Maximum heading level (`#` count) to split on (1–6) |
| `preserve_code_blocks` | `True` | Extract and protect fenced code blocks from being split |

Output `meta_data` includes `{"chunker": "markdown"}` in addition to the standard `chunk_index` / `chunk_total` fields.

### `SemanticChunker`

Embedding-based chunker that finds chunk boundaries where cosine similarity between consecutive sentence groups drops below a threshold. Produces semantically coherent chunks rather than fixed-size ones.

```python
from definable.knowledge.chunker import SemanticChunker
from definable.knowledge.embedder import OpenAIEmbedder

chunker = SemanticChunker(
  chunk_size=1000,            # soft size limit (not strictly enforced)
  chunk_overlap=200,
  embedder=OpenAIEmbedder(),  # None = fall back to size-based splitting
  similarity_threshold=0.5,   # 0.0-1.0; lower = fewer, larger chunks
  sentence_window=1,          # sentences to average on each side of a boundary candidate
  min_sentences=2,            # minimum sentences per chunk (merges tiny chunks)
)
chunks = chunker.chunk(document)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Soft size limit; used for fallback size-based splitting |
| `chunk_overlap` | `200` | Overlap for fallback size-based splitting |
| `embedder` | `None` | Embedder instance; `None` falls back to size-based splitting |
| `similarity_threshold` | `0.5` | Cosine similarity below which a boundary is inserted |
| `sentence_window` | `1` | Sentences averaged per side when computing similarity |
| `min_sentences` | `2` | Minimum sentences per chunk; tiny trailing chunks are merged |

When `embedder` is `None` or embedding fails, `SemanticChunker` falls back to size-based splitting silently — it never raises. Output `meta_data` includes `{"chunker": "semantic"}`.

## Chooser table

| Chunker | Best for | Respects structure | Needs embedder |
|---|---|---|---|
| `TextChunker` | Homogeneous plain text, CSV rows, logs | No | No |
| `RecursiveChunker` | General prose, mixed content | Partially (via separators) | No |
| `MarkdownChunker` | Docs, README files, wikis, reports | Yes (headings + code blocks) | No |
| `SemanticChunker` | High-quality RAG where topic coherence matters most | By meaning | Yes (soft) |

## Gotchas

- `chunk_size` is measured in **characters**, not tokens. A 1000-character chunk is roughly 200–300 tokens for English text. Adjust for your model's context window.
- `chunk_overlap` prepends trailing characters from the previous chunk to the current one, increasing the stored chunk length beyond `chunk_size`. Total stored size per chunk can be up to `chunk_size + chunk_overlap`.
- `MarkdownChunker` splits code blocks by placeholder substitution. A code block that appears at the very start of a section is included in that section's chunk. If a section with a code block exceeds `chunk_size`, the code block is kept intact and the surrounding prose is split.
- `SemanticChunker` calls `embedder.get_embedding()` once per sentence. For long documents with many sentences this generates many API calls. Consider using `RecursiveChunker` for bulk ingestion and reserving `SemanticChunker` for smaller, high-value documents.
- If `SemanticChunker` receives a document with fewer than `min_sentences` sentences it returns the document unchanged (a single-item list containing the original).
- All chunkers preserve `source`, `source_type`, and `meta_data` from the parent document. The `id` of the output chunks is auto-generated (`uuid4`), not derived from the parent.

## Related modules

- `definable.knowledge` — `Knowledge`, `Document`, full RAG pipeline
- `definable.knowledge.embedder` — embedder implementations (used by `SemanticChunker`)
- `definable.knowledge.reader` — produces `Document` objects from files and URLs
- `definable.vectordb` — stores chunked documents after embedding
