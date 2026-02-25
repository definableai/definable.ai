# knowledge/embedder

Text embedding sub-module. Converts raw text into float vectors that live in a vector database and power semantic search. Every component in the RAG pipeline (Knowledge, VectorDB, SemanticChunker) depends on an `Embedder` instance.

## Quick start

```python
from definable.knowledge.embedder import OpenAIEmbedder

embedder = OpenAIEmbedder()                        # text-embedding-3-small, 1536 dims
vector = embedder.get_embedding("Hello world")     # List[float]
vector = await embedder.async_get_embedding("Hi")  # async variant
```

Plug directly into `Knowledge`:

```python
from definable.knowledge import Knowledge
from definable.knowledge.embedder import VoyageAIEmbedder
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(embedder=VoyageAIEmbedder()),
)
```

## Module structure

```
embedder/
├── __init__.py     # Public API (all implementations lazy-loaded)
├── base.py         # Embedder ABC (dataclass)
├── openai.py       # OpenAIEmbedder
├── voyageai.py     # VoyageAIEmbedder
├── google.py       # GoogleEmbedder
├── mistral.py      # MistralEmbedder
└── fallback.py     # FallbackEmbedder (provider chain with auto-failover)
```

All implementations are lazy-loaded — importing from `__init__` does not import any provider SDK until the class is first accessed.

## API reference

### `Embedder` (ABC, `base.py`)

Base dataclass. All implementations inherit from this.

```python
@dataclass
class Embedder(ABC):
  dimensions: int = 1536
  batch_size: int = 100

  def get_embedding(self, text: str) -> List[float]: ...
  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]: ...
  async def async_get_embedding(self, text: str) -> List[float]: ...
  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]: ...
```

The `usage` dict is provider-specific. OpenAI returns `{"prompt_tokens": int, "total_tokens": int}`. VoyageAI returns `{"total_tokens": int}`. Google and Mistral may return `None` for usage.

### `OpenAIEmbedder`

```python
from definable.knowledge.embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(
  id="text-embedding-3-small",   # or "text-embedding-3-large" / "text-embedding-ada-002"
  dimensions=1536,               # None = provider default; auto-set from model
  encoding_format="float",       # "float" | "base64"
  user=None,                     # optional end-user ID for OpenAI abuse detection
  api_key=None,                  # falls back to OPENAI_API_KEY env var
  organization=None,
  base_url=None,                 # override for Azure OpenAI or compatible APIs
  request_params=None,           # merged into every embeddings.create() call
  client_params=None,            # merged into OpenAI() constructor
  openai_client=None,            # inject a pre-built OpenAI client (sync)
  async_client=None,             # inject a pre-built AsyncOpenAI client
)
```

**Dimension auto-detection:** when `dimensions=None`, `text-embedding-3-large` resolves to 3072; all other models resolve to 1536. The `dimensions` parameter is only sent for `text-embedding-3-*` models.

**Batch method** (async only):

```python
embeddings, usages = await embedder.async_get_embeddings_batch_and_usage(["text1", "text2"])
# Returns Tuple[List[List[float]], List[Optional[Dict]]]
```

### `VoyageAIEmbedder`

```python
from definable.knowledge.embedder import VoyageAIEmbedder

embedder = VoyageAIEmbedder(
  id="voyage-2",
  dimensions=1024,
  api_key=None,       # falls back to VOYAGEAI_API_KEY env var
  max_retries=None,
  timeout=None,
  request_params=None,
  client_params=None,
)
```

Requires `pip install voyageai`. Has async batch support via `async_get_embeddings_batch_and_usage`.

### `GoogleEmbedder`

```python
from definable.knowledge.embedder import GoogleEmbedder

embedder = GoogleEmbedder(
  id="text-embedding-004",
  dimensions=768,
  api_key=None,       # falls back to GOOGLE_API_KEY env var
  task_type=None,     # "retrieval_document" | "retrieval_query" |
                      # "semantic_similarity" | "classification" | "clustering"
)
```

Requires `pip install google-genai`. Usage info is not returned by the Google embedding API — `get_embedding_and_usage` always returns `None` for usage.

### `MistralEmbedder`

```python
from definable.knowledge.embedder import MistralEmbedder

embedder = MistralEmbedder(
  id="mistral-embed",
  dimensions=1024,
  api_key=None,  # falls back to MISTRAL_API_KEY env var
)
```

Requires `pip install mistralai`. Usage dict contains `{"prompt_tokens": int, "total_tokens": int}` when available.

### `FallbackEmbedder`

Chains multiple providers with automatic failover. On any error from the active provider, it classifies the failure (auth, rate_limit, timeout, network) and tries the next provider in order. Inherits `dimensions` from the primary provider and updates it dynamically when a fallback activates.

```python
from definable.knowledge.embedder import FallbackEmbedder, OpenAIEmbedder, VoyageAIEmbedder

embedder = FallbackEmbedder(
  providers=[
    OpenAIEmbedder(),    # primary
    VoyageAIEmbedder(),  # fallback
  ]
)

vector = embedder.get_embedding("text")
vector = await embedder.async_get_embedding("text")

# Reset to primary provider manually
embedder.reset()
```

When all providers are exhausted, raises `EmbeddingError` with `.error_type` (an `EmbeddingErrorType` enum), `.provider` (class name of last attempted provider), and `.original` (underlying exception).

## Provider table

| Class | Default model | Dimensions | Required env var | Install |
|---|---|---|---|---|
| `OpenAIEmbedder` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | `openai` |
| `VoyageAIEmbedder` | `voyage-2` | 1024 | `VOYAGEAI_API_KEY` | `voyageai` |
| `GoogleEmbedder` | `text-embedding-004` | 768 | `GOOGLE_API_KEY` | `google-genai` |
| `MistralEmbedder` | `mistral-embed` | 1024 | `MISTRAL_API_KEY` | `mistralai` |
| `FallbackEmbedder` | — (delegates) | from primary | — | — |

## Gotchas

- `FallbackEmbedder(providers=[])` raises `ValueError` immediately. At least one provider is required.
- `dimensions` on `OpenAIEmbedder` is only sent to the API for `text-embedding-3-*` models. Passing it for `ada-002` has no effect.
- When a `FallbackEmbedder` switches to a secondary provider, `self.dimensions` updates to that provider's dimensions. Vector databases that were initialized with the primary's dimension count will reject mismatched embeddings — ensure all providers in the chain output the same dimension, or call `reset()` before switching back.
- Embedder clients are created lazily on first use and cached on the instance. They are not explicitly closed — they rely on garbage collection. Avoid creating many short-lived embedder instances in tight loops.
- On failure, `get_embedding` and `async_get_embedding` return `[]` (empty list) rather than raising. Check for an empty return value before inserting into a vector DB.

## Related modules

- `definable.knowledge` — `Knowledge` class, `Document`, search pipeline
- `definable.vectordb` — `InMemoryVectorDB`, `PgVector`, `Qdrant`, `ChromaDb` (all accept an `Embedder`)
- `definable.knowledge.chunker` — chunking before embedding
- `definable.knowledge.scoring` — `FallbackEmbedder` is used by `HybridSearcher` internally
- `definable.embedder` — top-level re-export alias (same classes)
