# knowledge/scoring

Post-retrieval scoring utilities that improve result quality beyond raw vector similarity. Two independent passes run in sequence after the reranker: `TemporalDecay` penalizes stale documents, and `mmr_rerank` diversifies results so the top-k set covers more ground.

## Module structure

```
scoring/
├── __init__.py    # Public API
├── temporal.py    # TemporalDecay
└── mmr.py         # MMRConfig, mmr_rerank
```

## Quick start

```python
from definable.knowledge.scoring import TemporalDecay, MMRConfig
from definable.knowledge import Knowledge
from definable.vectordb import InMemoryVectorDB

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  temporal_decay=TemporalDecay(half_life_days=14.0),  # aggressively prefer recent
  mmr=MMRConfig(lambda_param=0.7),  # mostly relevance, some diversity
)
```

Both fields are optional. Omit either to skip that scoring pass.

## API reference

### TemporalDecay

```python
from definable.knowledge.scoring import TemporalDecay

decay = TemporalDecay(
  half_life_days=30.0,  # float — days until score is halved
  enabled=True,  # bool — set False to disable without removing from config
)
```

**Decay formula**

```
score *= exp(-ln(2) / half_life_days * age_days)
```

After one `half_life_days`, the score is multiplied by 0.5. After two half-lives, by 0.25. A document with `half_life_days=30` that is 60 days old retains 25% of its original score.

| `half_life_days` | Score after 30 days | Score after 90 days |
|------------------|---------------------|---------------------|
| 7                | 0.05                | ~0.0001             |
| 30               | 0.50                | 0.125               |
| 90               | 0.79                | 0.50                |
| 365              | 0.94                | 0.84                |

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `decay_factor` | `(age_days: float) -> float` | Returns the multiplier for a given age. Always in `(0, 1]`. |
| `apply` | `(documents: list[Document], now: float \| None = None) -> list[Document]` | Applies decay in place and returns documents sorted by decayed score. |

**Metadata keys**

`apply()` reads two metadata keys on each document:

| Key | Type | Meaning |
|-----|------|---------|
| `inserted_at` | `float` (Unix timestamp) | When the document was added to the index. |
| `created_at` | `float` (Unix timestamp) | Fallback if `inserted_at` is absent. |
| `evergreen` | `bool` | If `True`, the document is exempt from decay. |

Documents without either timestamp are left unchanged. The `now` parameter overrides the current time (useful in tests).

```python
import time
from definable.knowledge import Document
from definable.knowledge.scoring import TemporalDecay

now = time.time()
docs = [
  Document(content="Recent news", reranking_score=0.9, meta_data={"inserted_at": now - 86400}),  # 1 day old
  Document(content="Old tutorial", reranking_score=0.9, meta_data={"inserted_at": now - 86400 * 60}),  # 60 days old
  Document(content="Core concept", reranking_score=0.9, meta_data={"evergreen": True}),  # exempt
]

decay = TemporalDecay(half_life_days=30.0)
scored = decay.apply(docs)
# "Recent news" ~0.977, "Core concept" 0.9, "Old tutorial" ~0.25
```

### MMRConfig

```python
from definable.knowledge.scoring import MMRConfig

config = MMRConfig(
  lambda_param=0.5,  # float in [0.0, 1.0] — relevance vs diversity balance
  enabled=True,  # bool
)
```

| `lambda_param` | Behavior |
|----------------|----------|
| `1.0` | Pure relevance — identical to no MMR (top documents by score) |
| `0.7` | Mostly relevance, penalizes near-duplicates |
| `0.5` (default) | Equal weight between relevance and diversity |
| `0.0` | Maximum diversity — ignores relevance entirely |

### mmr_rerank

```python
from definable.knowledge.scoring import MMRConfig, mmr_rerank

diverse = mmr_rerank(
  query_embedding=query_emb,  # list[float] | None — query vector
  documents=docs,  # list[Document]
  config=MMRConfig(lambda_param=0.7),
  top_k=5,  # int | None — defaults to len(documents)
)
```

**MMR selection loop**

At each step, the algorithm selects the document that maximizes:

```
MMR = lambda * relevance(doc, query) - (1 - lambda) * max_sim(doc, selected)
```

`relevance` is cosine similarity to the query embedding when embeddings are available, or `reranking_score` otherwise. `max_sim` is cosine similarity to the already-selected set (Jaccard token overlap as fallback when embeddings are absent).

The greedy loop runs `top_k` times and is O(n * k) where n is the candidate count and k is `top_k`. For typical `top_k` values (5–20) over typical candidate sets (20–100), this is fast.

**Embedding availability**

| Condition | Similarity method |
|-----------|------------------|
| `query_embedding` provided and all docs have `.embedding` | Cosine similarity |
| Either is missing | Jaccard overlap on `doc.content` tokens |

The Jaccard fallback is coarser but avoids errors when embeddings are not stored.

## Integration with Knowledge

When both `temporal_decay` and `mmr` are set, the `Knowledge` retrieval pipeline applies them in this order:

```
vector search
  → (optional) HybridSearcher merge
  → (optional) Reranker
  → TemporalDecay.apply()       ← score decay by age
  → mmr_rerank()                ← diversity selection
  → top_k results returned
```

```python
from definable.knowledge import Knowledge
from definable.knowledge.scoring import TemporalDecay, MMRConfig
from definable.knowledge.fts import FTSIndex, HybridSearchConfig
from definable.vectordb import InMemoryVectorDB

fts = FTSIndex()
await fts.initialize()

knowledge = Knowledge(
  vector_db=InMemoryVectorDB(),
  fts_index=fts,
  hybrid_config=HybridSearchConfig(),
  temporal_decay=TemporalDecay(half_life_days=30.0),
  mmr=MMRConfig(lambda_param=0.7),
  top_k=5,
)
```

## Standalone usage

Both utilities work independently of `Knowledge` and can be applied to any list of `Document` objects:

```python
from definable.knowledge.scoring import TemporalDecay, MMRConfig, mmr_rerank

# After your own retrieval step
docs = my_retriever.search("query")

# Apply decay
decay = TemporalDecay(half_life_days=60.0)
docs = decay.apply(docs)

# Diversify
docs = mmr_rerank(query_embedding=None, documents=docs, config=MMRConfig(lambda_param=0.6), top_k=5)
```

## Gotchas

**`TemporalDecay` mutates `reranking_score` in place.** It does not copy documents. If you need the original scores, copy the list or store scores before calling `apply()`.

**Documents without timestamps are unaffected.** Decay only fires when `meta_data["inserted_at"]` or `meta_data["created_at"]` exists and is a valid float. A missing or non-numeric timestamp is silently skipped — the document retains its original score.

**Evergreen documents skip decay but are still ranked against decayed scores.** An evergreen document with `reranking_score=0.5` will rank below a fresh document with `reranking_score=0.6` even after the non-evergreen document is decayed, unless the decayed score falls below 0.5.

**`mmr_rerank` with `lambda_param=1.0` is not a no-op in terms of ordering.** It still runs the greedy loop but with the diversity term zeroed out, so it produces the same order as sorting by score. Prefer `MMRConfig(enabled=False)` to skip entirely.

**MMR is O(n * k).** For large candidate pools (n > 500) with large `top_k` (> 50), the quadratic inner loop may add latency. In practice, retrieval pipelines use small candidate counts and this is not a concern.

**`query_embedding=None` falls back to Jaccard.** Jaccard similarity is token-level (whitespace split) and much weaker than cosine similarity. Always pass the query embedding when you have it.
