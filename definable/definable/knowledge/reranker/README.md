# knowledge/reranker

Document reranking sub-module. Takes the candidate documents returned by a vector (or hybrid) search and reorders them by true relevance to the query. Reranking is a precision layer — the vector search provides recall, the reranker provides the final ranking quality.

## Quick start

```python
from definable.knowledge.reranker import CohereReranker
from definable.knowledge import Document

reranker = CohereReranker()
documents = [Document(content="..."), Document(content="...")]

reranked = reranker.rerank("what is quantum entanglement?", documents)
reranked = await reranker.arerank("what is quantum entanglement?", documents)
```

Plug into `Knowledge` to apply reranking after every search:

```python
from definable.knowledge import Knowledge
from definable.knowledge.reranker import CohereReranker
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  reranker=CohereReranker(top_n=5),
)
```

## Module structure

```
reranker/
├── __init__.py               # Public API (implementations lazy-loaded)
├── base.py                   # Reranker ABC (Pydantic BaseModel)
├── cohere.py                 # CohereReranker (cloud API)
└── sentence_transformer.py   # SentenceTransformerReranker (local, no API)
```

Both implementations are lazy-loaded — importing from `__init__` does not import the underlying SDK until the class is first accessed.

## API reference

### `Reranker` (ABC, `base.py`)

Base class. Extends `pydantic.BaseModel` with `arbitrary_types_allowed=True`.

```python
class Reranker(BaseModel, ABC):
  def rerank(self, query: str, documents: List[Document]) -> List[Document]: ...
  async def arerank(self, query: str, documents: List[Document]) -> List[Document]: ...
```

Both methods return a new list of `Document` objects sorted by descending relevance score. Each returned document has its `reranking_score: Optional[float]` field populated.

Input documents are mutated in-place to set `reranking_score` before being sorted — do not rely on input list stability.

### `CohereReranker`

Cloud-based reranker using Cohere's `/rerank` endpoint. Supports multilingual content. The `arerank` method runs the synchronous Cohere client in a thread pool executor.

```python
from definable.knowledge.reranker import CohereReranker

reranker = CohereReranker(
  model="rerank-multilingual-v3.0",  # or "rerank-english-v3.0"
  api_key=None,  # falls back to COHERE_API_KEY env var
  top_n=None,  # None = return all documents; int = return top N
  cohere_client=None,  # inject a pre-built CohereClient
)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | `"rerank-multilingual-v3.0"` | Cohere rerank model ID |
| `api_key` | `None` | API key; falls back to `COHERE_API_KEY` env var |
| `top_n` | `None` | Maximum documents to return; `None` returns all |
| `cohere_client` | `None` | Inject a pre-built `cohere.Client` instance |

Requires `pip install cohere`.

On error, `rerank` and `arerank` log the exception and return the original, unreranked documents rather than raising — the pipeline degrades gracefully.

### `SentenceTransformerReranker`

Local open-source reranker using a sentence-transformers cross-encoder model. No API key or network access required after the model is downloaded. The cross-encoder scores each query-document pair directly, making it more accurate than bi-encoder retrieval for short-list reranking.

```python
from definable.knowledge.reranker import SentenceTransformerReranker

reranker = SentenceTransformerReranker(
  model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # HuggingFace model ID
  top_n=None,  # None = return all
  device=None,  # None = auto; "cpu" | "cuda" | "mps"
  batch_size=32,
)
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace cross-encoder model |
| `top_n` | `None` | Maximum documents to return |
| `device` | `None` | Inference device; `None` lets sentence-transformers auto-detect |
| `batch_size` | `32` | Number of query-document pairs scored per forward pass |

Requires `pip install sentence-transformers`. The model is downloaded from HuggingFace on first use and cached locally. The `arerank` method runs in a thread pool executor because cross-encoder inference is CPU/GPU-bound.

On error, `rerank` and `arerank` log the exception and return original documents unchanged.

## Implementation table

| Class | Type | API key | Install | Notes |
|---|---|---|---|---|
| `CohereReranker` | Cloud API | `COHERE_API_KEY` | `cohere` | Multilingual; async via thread pool |
| `SentenceTransformerReranker` | Local model | None | `sentence-transformers` | No network after download; GPU-friendly |

## Gotchas

- **`Reranker` is a Pydantic `BaseModel`, not a dataclass.** Unlike `Embedder` and `Chunker`, you configure it with keyword arguments that Pydantic validates, not with `@dataclass` field defaults. Subclasses must declare fields as Pydantic fields.
- **Input documents are mutated.** `rerank` and `arerank` set `document.reranking_score` on each input document before sorting. If you need to preserve the original document objects, deepcopy the list before calling.
- **`top_n=0` on `CohereReranker`** is treated as invalid and silently reset to `None` (return all). Use `None` explicitly to return all documents.
- **`SentenceTransformerReranker` downloads on first use.** The model weight download happens the first time `cross_encoder` is accessed (lazy-load). In production, pre-warm the reranker before serving traffic.
- **`arerank` is not truly async for either implementation.** Both run the synchronous scoring in `asyncio.get_event_loop().run_in_executor(None, ...)`. This avoids blocking the event loop but does not parallelize across multiple rerank calls. For high-throughput scenarios, use a process pool.
- **Empty input returns empty output.** Passing an empty `documents` list returns `[]` immediately, no API call is made.
- **Reranking scores are not probabilities.** `CohereReranker` returns relevance scores in the range `[0, 1]`. `SentenceTransformerReranker` returns raw cross-encoder logits which can be negative or exceed 1. Do not compare scores across implementations.

## Related modules

- `definable.knowledge` — `Knowledge` class; set `reranker=` to activate post-search reranking
- `definable.knowledge.embedder` — embedding for vector retrieval (runs before reranking)
- `definable.knowledge.scoring` — `MMRConfig` and `TemporalDecay` (applied after reranking in the full pipeline)
- `definable.vectordb` — returns candidate documents that the reranker then sorts
