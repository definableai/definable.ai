"""
Smoke-test all MemoryStore backends.

Runs a minimal round-trip (store, retrieve, search, delete) against each
backend.  Backends whose dependencies are not installed or whose services
are not reachable are skipped gracefully.

Environment variables for optional backends:
    MEMORY_REDIS_URL      — e.g. redis://localhost:6379
    MEMORY_POSTGRES_URL   — e.g. postgresql://user:pass@localhost/dbname
    MEMORY_MONGODB_URL    — e.g. mongodb://localhost:27017
    MEMORY_QDRANT_URL     — e.g. http://localhost:6333
    PINECONE_API_KEY      — Pinecone API key

Usage:
    python definable/examples/memory/03_store_backends.py
"""

import asyncio
import os
import time
from typing import Any, List, Tuple

from definable.memory import InMemoryStore
from definable.memory.types import Episode, KnowledgeAtom


async def test_store(name: str, store: Any) -> str:
  """Run a minimal round-trip against *store*. Returns a status string."""

  await store.initialize()

  # --- Episode round-trip ---
  ep = Episode(
    id="test-ep-1",
    user_id="example-user",
    session_id="example-session",
    role="user",
    content="Hello from the smoke test!",
    embedding=[1.0, 0.0, 0.0],
    topics=["test"],
    created_at=time.time(),
  )
  await store.store_episode(ep)

  episodes = await store.get_episodes(user_id="example-user", limit=10)
  assert len(episodes) >= 1, f"{name}: expected at least 1 episode"
  assert episodes[0].content == "Hello from the smoke test!"

  # --- Atom round-trip ---
  atom = KnowledgeAtom(
    id="test-atom-1",
    user_id="example-user",
    subject="test",
    predicate="is",
    object="passing",
    content="test is passing",
    embedding=[0.0, 1.0, 0.0],
    confidence=0.9,
    created_at=time.time(),
  )
  await store.store_atom(atom)

  atoms = await store.get_atoms(user_id="example-user")
  assert len(atoms) >= 1, f"{name}: expected at least 1 atom"

  # --- Vector search ---
  results = await store.search_episodes_by_embedding(
    [0.9, 0.1, 0.0],
    user_id="example-user",
    top_k=5,
  )
  assert len(results) >= 1, f"{name}: vector search returned no results"

  # --- Cleanup ---
  await store.delete_session_data("example-session")
  await store.close()

  return "PASS"


def _build_backends() -> List[Tuple[str, Any]]:
  """Instantiate each backend, returning (name, store) pairs.

  Backends whose optional dependencies are missing are silently skipped.
  """
  backends: List[Tuple[str, Any]] = []

  # 1. InMemoryStore — always available
  backends.append(("InMemoryStore", InMemoryStore()))

  # 2. SQLiteMemoryStore
  try:
    from definable.memory import SQLiteMemoryStore

    backends.append(("SQLiteMemoryStore", SQLiteMemoryStore("./test_example.db")))
  except ImportError:
    pass

  # 3. ChromaMemoryStore
  try:
    from definable.memory import ChromaMemoryStore

    backends.append(("ChromaMemoryStore", ChromaMemoryStore(collection_prefix="example_")))
  except ImportError:
    pass

  # 4. RedisMemoryStore
  redis_url = os.environ.get("MEMORY_REDIS_URL")
  if redis_url:
    try:
      from definable.memory import RedisMemoryStore

      backends.append(("RedisMemoryStore", RedisMemoryStore(redis_url=redis_url, prefix="example")))
    except ImportError:
      pass

  # 5. PostgresMemoryStore
  pg_url = os.environ.get("MEMORY_POSTGRES_URL")
  if pg_url:
    try:
      from definable.memory import PostgresMemoryStore

      backends.append(("PostgresMemoryStore", PostgresMemoryStore(db_url=pg_url, table_prefix="example_")))
    except ImportError:
      pass

  # 6. MongoMemoryStore
  mongo_url = os.environ.get("MEMORY_MONGODB_URL")
  if mongo_url:
    try:
      from definable.memory import MongoMemoryStore

      backends.append((
        "MongoMemoryStore",
        MongoMemoryStore(connection_string=mongo_url, database="example", collection_prefix="example_"),
      ))
    except ImportError:
      pass

  # 7. QdrantMemoryStore
  qdrant_url = os.environ.get("MEMORY_QDRANT_URL")
  if qdrant_url:
    try:
      from definable.memory import QdrantMemoryStore

      # Parse host and port from URL like http://localhost:6333
      url_clean = qdrant_url.replace("http://", "").replace("https://", "")
      parts = url_clean.split(":")
      host = parts[0]
      port = int(parts[1]) if len(parts) > 1 else 6333
      backends.append(("QdrantMemoryStore", QdrantMemoryStore(url=host, port=port, prefix="example", vector_size=3)))
    except (ImportError, ValueError):
      pass

  # 8. PineconeMemoryStore
  pinecone_key = os.environ.get("PINECONE_API_KEY")
  if pinecone_key:
    try:
      from definable.memory import PineconeMemoryStore

      backends.append((
        "PineconeMemoryStore",
        PineconeMemoryStore(api_key=pinecone_key, index_name="example-memory", vector_size=3),
      ))
    except ImportError:
      pass

  return backends


async def main():
  backends = _build_backends()
  results: List[Tuple[str, str]] = []

  print(f"Testing {len(backends)} backend(s)...\n")

  for name, store in backends:
    try:
      status = await test_store(name, store)
    except Exception as exc:
      status = f"FAIL ({type(exc).__name__}: {exc})"
    results.append((name, status))
    symbol = "+" if status == "PASS" else "x"
    print(f"  [{symbol}] {name}: {status}")

  # Cleanup SQLite test file
  if os.path.exists("./test_example.db"):
    os.remove("./test_example.db")

  # Summary table
  passed = sum(1 for _, s in results if s == "PASS")
  print(f"\n{'=' * 50}")
  print(f"Results: {passed}/{len(results)} backends passed")
  print(f"{'=' * 50}")
  for name, status in results:
    print(f"  {name:<25s} {status}")


if __name__ == "__main__":
  asyncio.run(main())
