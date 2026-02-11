# vectordbs

Legacy vector database interface — superseded by `knowledge/vector_dbs/`.

## Overview

This module contains the original vector database abstraction with a Qdrant implementation. It has been superseded by the `knowledge/vector_dbs/` module which provides a simpler, more consistent interface. New code should use `definable.knowledge.vector_dbs` instead.

## API Reference

### VectorDb (ABC)

```python
from definable.vectordbs.base import VectorDb
```

Abstract base class with CRUD, search, and lifecycle methods: `create()`, `insert()`, `upsert()`, `search()`, `delete()`, `drop()`, `optimize()`.

### Qdrant

```python
from definable.vectordbs.qdrant.ops import Qdrant
```

Qdrant implementation supporting in-memory, local, and cloud deployments. Supports dense, sparse, and hybrid search with optional embedder and reranker.

## Migration

Use `definable.knowledge.vector_dbs` instead:

```python
# Old
from definable.vectordbs.qdrant.ops import Qdrant

# New
from definable.knowledge import InMemoryVectorDB, PgVectorDB
```

## See Also

- `knowledge/vector_dbs/` — Current vector database implementations
- `knowledge/` — Full RAG pipeline
