"""
Vector databases: InMemory and PgVector.

This example shows how to:
- Use InMemoryVectorDB for development
- Configure PgVectorDB for production
- Understand vector database operations

Requirements:
    No API keys required for InMemory example.
    PostgreSQL with pgvector extension for PgVector example.
"""

import os
from typing import List

from definable.knowledge import (
  Document,
  Embedder,
  InMemoryVectorDB,
  Knowledge,
)


class MockEmbedder(Embedder):
  """Simple mock embedder for demonstration."""

  dimensions: int = 64

  def get_embedding(self, text: str) -> List[float]:
    import hashlib

    hash_bytes = hashlib.sha256(text.lower().encode()).digest()
    return [(b / 127.5) - 1.0 for b in hash_bytes[: self.dimensions]]

  def get_embedding_and_usage(self, text: str):
    return self.get_embedding(text), {"tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    return self.get_embedding_and_usage(text)


def in_memory_vector_db():
  """InMemoryVectorDB for development and testing."""
  print("InMemoryVectorDB Example")
  print("=" * 50)

  # Create in-memory vector database
  vector_db = InMemoryVectorDB(
    dimensions=64,  # Must match embedder dimensions
    collection_name="my_collection",  # Optional: name for the collection
  )

  print("Created InMemoryVectorDB")
  print(f"  Collection: {vector_db.collection_name}")
  print(f"  Dimensions: {vector_db.dimensions}")

  # Create embedder
  embedder = MockEmbedder()

  # Create documents with embeddings
  documents = [
    Document(content="Python is great for data science."),
    Document(content="JavaScript powers modern web applications."),
    Document(content="Rust offers memory safety guarantees."),
  ]

  # Generate embeddings and add to vector DB
  for doc in documents:
    doc.embedding = embedder.get_embedding(doc.content)

  # Add documents
  ids = vector_db.add(documents)
  print(f"\nAdded {len(ids)} documents")
  print(f"  IDs: {ids}")

  # Search
  query = "What language is good for data analysis?"
  query_embedding = embedder.get_embedding(query)

  results = vector_db.search(query_embedding, top_k=2)
  print(f"\nSearch results for: {query}")
  for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.content}")

  # Count documents
  count = vector_db.count()
  print(f"\nTotal documents: {count}")

  # Clear
  vector_db.clear()
  print(f"After clear: {vector_db.count()} documents")


def pgvector_db_example():
  """PgVectorDB for production use."""
  print("\n" + "=" * 50)
  print("PgVectorDB Example (PostgreSQL)")
  print("=" * 50)

  # Check for PostgreSQL connection string
  connection_string = os.getenv("DATABASE_URL") or os.getenv("PGVECTOR_URL")

  if not connection_string:
    print("No PostgreSQL connection string found.")
    print("Set DATABASE_URL or PGVECTOR_URL environment variable.")
    print("\nExample connection string:")
    print("  postgresql://user:password@localhost:5432/mydb")
    print("\nSkipping PgVector example...")
    return

  try:
    from definable.vectordb import PgVector as PgVectorDB

    # Create PgVector database
    vector_db = PgVectorDB(  # type: ignore[call-arg]
      connection_string=connection_string,
      collection_name="documents",
      dimensions=64,
    )

    print("Connected to PostgreSQL")
    print(f"  Collection: {vector_db.collection_name}")  # type: ignore[attr-defined]
    print(f"  Dimensions: {vector_db.dimensions}")

    # Use with knowledge base
    embedder = MockEmbedder()
    kb = Knowledge(vector_db=vector_db, embedder=embedder)

    # Add test document
    kb.add(Document(content="Test document for PgVector."))
    print("Added test document")

    # Search
    results = kb.search("test document", top_k=1)
    print(f"Search found: {len(results)} results")

    # Clean up
    vector_db.clear()  # type: ignore[attr-defined]
    print("Cleared test data")

  except ImportError:
    print("PgVectorDB not available. Install with: pip install pgvector")
  except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")


def vector_db_operations():
  """Common vector database operations."""
  print("\n" + "=" * 50)
  print("Vector Database Operations")
  print("=" * 50)

  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  # 1. Add documents
  print("\n1. Adding documents:")
  docs = [
    Document(id="doc-1", content="First document about AI"),
    Document(id="doc-2", content="Second document about ML"),
    Document(id="doc-3", content="Third document about data"),
  ]

  for doc in docs:
    doc.embedding = embedder.get_embedding(doc.content)

  ids = vector_db.add(docs)
  print(f"   Added documents: {ids}")

  # 2. Search with different top_k
  print("\n2. Search with different top_k:")
  query_embedding = embedder.get_embedding("artificial intelligence")

  for k in [1, 2, 3]:
    results = vector_db.search(query_embedding, top_k=k)
    print(f"   top_k={k}: Found {len(results)} results")

  # 3. Delete specific documents
  print("\n3. Delete operations:")
  print(f"   Before delete: {vector_db.count()} documents")

  vector_db.delete(["doc-2"])  # Delete by ID
  print(f"   After deleting doc-2: {vector_db.count()} documents")

  # 4. Clear all
  print("\n4. Clear all:")
  vector_db.clear()
  print(f"   After clear: {vector_db.count()} documents")


def vector_db_with_filters():
  """Using metadata filters with vector search."""
  print("\n" + "=" * 50)
  print("Vector Search with Metadata Filters")
  print("=" * 50)

  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  # Add documents with metadata
  docs = [
    Document(
      content="Python tutorial for beginners",
      meta_data={"category": "tutorial", "level": "beginner"},
    ),
    Document(
      content="Advanced Python techniques",
      meta_data={"category": "tutorial", "level": "advanced"},
    ),
    Document(
      content="Python API documentation",
      meta_data={"category": "docs", "level": "reference"},
    ),
    Document(
      content="JavaScript basics",
      meta_data={"category": "tutorial", "level": "beginner"},
    ),
  ]

  for doc in docs:
    doc.embedding = embedder.get_embedding(doc.content)
  vector_db.add(docs)

  print(f"Added {len(docs)} documents with metadata")

  # Search with filter
  query_embedding = embedder.get_embedding("programming tutorial")

  # Note: Filter support depends on vector DB implementation
  # InMemoryVectorDB may have limited filter support
  print("\nSearch for tutorials:")
  results = vector_db.search(query_embedding, top_k=3)
  for doc in results:
    print(f"  - {doc.content}")
    print(f"    Meta: {doc.meta_data}")


def choosing_vector_db():
  """Guidance on choosing a vector database."""
  print("\n" + "=" * 50)
  print("Choosing a Vector Database")
  print("=" * 50)

  print("""
InMemoryVectorDB:
  ✓ Zero configuration required
  ✓ Perfect for development and testing
  ✓ Fast for small datasets
  ✗ Data lost on restart
  ✗ Limited by available memory
  ✗ Single process only

PgVectorDB:
  ✓ Persistent storage
  ✓ Scales to millions of vectors
  ✓ ACID transactions
  ✓ Combines with relational queries
  ✓ Production-ready
  ✗ Requires PostgreSQL setup
  ✗ Needs pgvector extension

Recommendation:
  - Development/Testing: InMemoryVectorDB
  - Production (small-medium): PgVectorDB
  - Production (large scale): Consider Pinecone, Weaviate, Qdrant
""")


def main():
  in_memory_vector_db()
  pgvector_db_example()
  vector_db_operations()
  vector_db_with_filters()
  choosing_vector_db()


if __name__ == "__main__":
  main()
