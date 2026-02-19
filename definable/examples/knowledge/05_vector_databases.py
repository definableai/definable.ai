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

from definable.embedder import Embedder
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB


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

  # Create embedder
  embedder = MockEmbedder()

  # Create in-memory vector database with embedder
  # The embedder is used internally for search query embedding.
  vector_db = InMemoryVectorDB(
    name="my_collection",
    embedder=embedder,
  )

  print("Created InMemoryVectorDB")
  print(f"  Name: {vector_db.name}")

  # Create documents with embeddings
  documents = [
    Document(content="Python is great for data science."),
    Document(content="JavaScript powers modern web applications."),
    Document(content="Rust offers memory safety guarantees."),
  ]

  # Generate embeddings for documents
  for doc in documents:
    doc.embedding = embedder.get_embedding(doc.content)

  # Insert documents (embeddings must be pre-computed)
  vector_db.insert(documents)
  print(f"\nInserted {len(documents)} documents")

  # Search — query is a string; the embedder embeds it internally
  query = "What language is good for data analysis?"
  results = vector_db.search(query, limit=2)
  print(f"\nSearch results for: {query}")
  for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.content}")

  # Count documents
  count = vector_db.count()
  print(f"\nTotal documents: {count}")

  # Drop all data
  vector_db.drop()
  print(f"After drop: {vector_db.count()} documents")


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
    vector_db.drop()
    print("Dropped test data")

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
  vector_db = InMemoryVectorDB(embedder=embedder)

  # 1. Add documents
  print("\n1. Adding documents:")
  docs = [
    Document(id="doc-1", content="First document about AI"),
    Document(id="doc-2", content="Second document about ML"),
    Document(id="doc-3", content="Third document about data"),
  ]

  for doc in docs:
    doc.embedding = embedder.get_embedding(doc.content)

  vector_db.insert(docs)
  print(f"   Inserted {len(docs)} documents")

  # 2. Search with different limits
  print("\n2. Search with different limits:")
  for k in [1, 2, 3]:
    results = vector_db.search("artificial intelligence", limit=k)
    print(f"   limit={k}: Found {len(results)} results")

  # 3. Delete specific documents
  print("\n3. Delete operations:")
  print(f"   Before delete: {vector_db.count()} documents")

  vector_db.delete_by_id("doc-2")
  print(f"   After deleting doc-2: {vector_db.count()} documents")

  # 4. Drop all
  print("\n4. Drop all:")
  vector_db.drop()
  print(f"   After drop: {vector_db.count()} documents")


def vector_db_with_filters():
  """Using metadata filters with vector search."""
  print("\n" + "=" * 50)
  print("Vector Search with Metadata Filters")
  print("=" * 50)

  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(embedder=embedder)

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
  vector_db.insert(docs)

  print(f"Added {len(docs)} documents with metadata")

  # Search with metadata filter
  print("\nSearch for tutorials (with category filter):")
  results = vector_db.search("programming tutorial", limit=3, filters={"category": "tutorial"})
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
  + Zero configuration required
  + Perfect for development and testing
  + Fast for small datasets
  - Data lost on restart
  - Limited by available memory
  - Single process only

PgVectorDB:
  + Persistent storage
  + Scales to millions of vectors
  + ACID transactions
  + Combines with relational queries
  + Production-ready
  - Requires PostgreSQL setup
  - Needs pgvector extension

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
