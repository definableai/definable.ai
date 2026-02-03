"""
Document management in knowledge bases.

This example shows how to:
- Add documents from various sources
- Update and delete documents
- Query document metadata
- Clear the knowledge base

Requirements:
    No API keys required for this example (uses mock embedder).
"""

from typing import List

from definable.knowledge import Document, Embedder, InMemoryVectorDB, Knowledge


class MockEmbedder(Embedder):
  """Simple mock embedder for demonstration."""

  dimensions: int = 64

  def get_embedding(self, text: str) -> List[float]:
    import hashlib

    hash_bytes = hashlib.sha256(text.lower().encode()).digest()
    embedding = [(b / 127.5) - 1.0 for b in hash_bytes[: self.dimensions]]
    return embedding

  def get_embedding_and_usage(self, text: str):
    return self.get_embedding(text), {"tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    return self.get_embedding_and_usage(text)


def setup_knowledge_base():
  """Set up a knowledge base."""
  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)
  return Knowledge(vector_db=vector_db, embedder=embedder)


def add_documents_example():
  """Different ways to add documents."""
  print("Adding Documents")
  print("=" * 50)

  kb = setup_knowledge_base()

  # Method 1: Add a simple Document object
  doc1 = Document(
    content="Python is a versatile programming language.",
    meta_data={"source": "manual", "topic": "programming"},
  )
  ids = kb.add(doc1)
  print(f"1. Added Document object: {ids}")

  # Method 2: Add with custom ID
  doc2 = Document(
    id="custom-id-001",
    content="JavaScript runs in web browsers.",
    meta_data={"source": "manual", "topic": "web"},
  )
  ids = kb.add(doc2)
  print(f"2. Added with custom ID: {ids}")

  # Method 3: Add Document with name
  doc3 = Document(
    name="rust-intro",
    content="Rust is a systems programming language focused on safety.",
    meta_data={"source": "docs", "topic": "programming"},
  )
  ids = kb.add(doc3)
  print(f"3. Added with name: {ids}")

  # Verify documents were added
  results = kb.search("programming language", top_k=5)
  print(f"\nSearch found {len(results)} documents")

  return kb


def document_with_metadata():
  """Working with document metadata."""
  print("\n" + "=" * 50)
  print("Document Metadata")
  print("=" * 50)

  kb = setup_knowledge_base()

  # Add documents with rich metadata
  documents = [
    Document(
      content="Q1 revenue increased by 15% compared to last year.",
      meta_data={
        "type": "financial",
        "quarter": "Q1",
        "year": 2024,
        "department": "finance",
      },
    ),
    Document(
      content="Q2 showed steady growth with 12% increase in sales.",
      meta_data={
        "type": "financial",
        "quarter": "Q2",
        "year": 2024,
        "department": "sales",
      },
    ),
    Document(
      content="New product launch planned for Q3 2024.",
      meta_data={
        "type": "announcement",
        "quarter": "Q3",
        "year": 2024,
        "department": "product",
      },
    ),
  ]

  for doc in documents:
    kb.add(doc)
    print(f"Added: {doc.meta_data}")

  # Search and see metadata
  print("\nSearch results with metadata:")
  results = kb.search("revenue growth", top_k=3)
  for doc in results:
    print(f"  - {doc.content[:40]}...")
    print(f"    Metadata: {doc.meta_data}")


def clear_knowledge_base():
  """Clear all documents from knowledge base."""
  print("\n" + "=" * 50)
  print("Clearing Knowledge Base")
  print("=" * 50)

  kb = setup_knowledge_base()

  # Add some documents
  for i in range(5):
    kb.add(Document(content=f"Document number {i}"))

  # Check count
  results = kb.search("document", top_k=10)
  print(f"Documents before clear: {len(results)}")

  # Clear all documents
  kb.clear()
  print("Knowledge base cleared")

  # Verify
  results = kb.search("document", top_k=10)
  print(f"Documents after clear: {len(results)}")


def document_sources():
  """Track document sources."""
  print("\n" + "=" * 50)
  print("Document Sources")
  print("=" * 50)

  kb = setup_knowledge_base()

  # Documents from different sources
  documents = [
    Document(
      content="API endpoint documentation for user management.",
      source="api_docs.md",
      source_type="file",
      meta_data={"category": "api"},
    ),
    Document(
      content="Tutorial on setting up authentication.",
      source="https://docs.example.com/auth",
      source_type="url",
      meta_data={"category": "tutorial"},
    ),
    Document(
      content="Internal notes about system architecture.",
      source="user_input",
      source_type="text",
      meta_data={"category": "internal"},
    ),
  ]

  for doc in documents:
    kb.add(doc)
    print(f"Added from {doc.source_type}: {doc.source}")

  # Search and see sources
  print("\nSearch results with source info:")
  results = kb.search("documentation", top_k=3)
  for doc in results:
    print(f"  - Content: {doc.content[:40]}...")
    print(f"    Source: {doc.source} ({doc.source_type})")


def batch_operations():
  """Batch document operations."""
  print("\n" + "=" * 50)
  print("Batch Operations")
  print("=" * 50)

  kb = setup_knowledge_base()

  # Create batch of documents
  batch = [Document(content=f"Batch document {i}: Information about topic {i}") for i in range(10)]

  print(f"Adding {len(batch)} documents in batch...")

  # Add all at once
  for doc in batch:
    kb.add(doc)

  print("Batch add complete")

  # Verify
  results = kb.search("batch document", top_k=10)
  print(f"Found {len(results)} documents")


def main():
  add_documents_example()
  document_with_metadata()
  clear_knowledge_base()
  document_sources()
  batch_operations()


if __name__ == "__main__":
  main()
