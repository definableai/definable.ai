"""
Basic RAG (Retrieval-Augmented Generation) setup.

This example shows how to:
- Create a Knowledge base
- Add documents with embeddings
- Search for relevant documents

Requirements:
    export VOYAGE_API_KEY=pa-...  # Or use mock embedder

Note: This example can work without API keys using a mock embedder.
"""

from typing import List

from definable.embedder import Embedder
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB


class MockEmbedder(Embedder):
  """A simple mock embedder for demonstration.

  Creates embeddings based on text similarity using character-level hashing.
  For production, use VoyageAIEmbedder or OpenAIEmbedder.
  """

  dimensions: int = 128

  def get_embedding(self, text: str) -> List[float]:
    """Generate a mock embedding."""
    import hashlib

    # Normalize text
    text = text.lower().strip()

    # Create deterministic embedding from text content
    embedding = [0.0] * self.dimensions

    # Use multiple hash functions for better distribution
    for i, word in enumerate(text.split()):
      word_hash = hashlib.md5(word.encode()).digest()
      for j, byte in enumerate(word_hash):
        idx = (i + j) % self.dimensions
        embedding[idx] += (byte / 255.0 - 0.5) * (1 / (i + 1))

    # Normalize
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
      embedding = [x / magnitude for x in embedding]

    return embedding

  def get_embedding_and_usage(self, text: str):
    """Get embedding with usage info."""
    embedding = self.get_embedding(text)
    usage = {"tokens": len(text.split())}
    return embedding, usage

  async def async_get_embedding(self, text: str) -> List[float]:
    """Async get embedding (same as sync for mock)."""
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    """Async get embedding with usage info."""
    return self.get_embedding_and_usage(text)


def basic_rag_setup():
  """Basic RAG setup with in-memory storage."""
  print("Basic RAG Setup")
  print("=" * 50)

  # Create embedder (use mock for demo, VoyageAIEmbedder for production)
  embedder = MockEmbedder()

  # Create vector database
  vector_db = InMemoryVectorDB(
    dimensions=embedder.dimensions,
    collection_name="my_documents",
  )

  # Create knowledge base
  kb = Knowledge(
    vector_db=vector_db,
    embedder=embedder,
  )

  print(f"Created knowledge base with {embedder.dimensions}-dimensional embeddings")
  return kb


def add_documents(kb: Knowledge):
  """Add documents to the knowledge base."""
  print("\n" + "=" * 50)
  print("Adding Documents")
  print("=" * 50)

  # Create documents
  documents = [
    Document(
      content="Python is a high-level programming language known for its readability and versatility.",
      meta_data={"topic": "programming", "language": "python"},
    ),
    Document(
      content="Machine learning is a subset of AI that enables systems to learn from data.",
      meta_data={"topic": "ai", "subtopic": "machine_learning"},
    ),
    Document(
      content="Neural networks are computing systems inspired by biological neural networks.",
      meta_data={"topic": "ai", "subtopic": "neural_networks"},
    ),
    Document(
      content="REST APIs use HTTP methods like GET, POST, PUT, and DELETE for web services.",
      meta_data={"topic": "web", "subtopic": "apis"},
    ),
    Document(
      content="Docker containers package applications with their dependencies for consistent deployment.",
      meta_data={"topic": "devops", "subtopic": "containers"},
    ),
  ]

  # Add documents to knowledge base
  for doc in documents:
    kb.add(doc)
    print(f"Added: {doc.content[:50]}...")

  print(f"\nTotal documents added: {len(documents)}")


def search_documents(kb: Knowledge):
  """Search for relevant documents."""
  print("\n" + "=" * 50)
  print("Searching Documents")
  print("=" * 50)

  # Search queries
  queries = [
    "What is Python?",
    "Tell me about machine learning",
    "How do containers work?",
    "What are neural networks?",
  ]

  for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 40)

    # Search with top_k=2 to get 2 most relevant documents
    results = kb.search(query, top_k=2)

    for i, doc in enumerate(results, 1):
      score = doc.reranking_score or "N/A"
      print(f"  {i}. {doc.content[:60]}...")
      print(f"     Meta: {doc.meta_data}")
      print(f"     Score: {score}")


def complete_example():
  """Complete RAG example from setup to search."""
  print("\n" + "=" * 50)
  print("Complete RAG Example")
  print("=" * 50)

  # Setup
  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)
  kb = Knowledge(vector_db=vector_db, embedder=embedder)

  # Add sample documents
  sample_docs = [
    "The Earth orbits the Sun at an average distance of 93 million miles.",
    "Water freezes at 32 degrees Fahrenheit (0 degrees Celsius).",
    "The human brain contains approximately 86 billion neurons.",
    "Light travels at about 186,000 miles per second in a vacuum.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
  ]

  for content in sample_docs:
    kb.add(Document(content=content))

  # Search
  print("\nSearching for: 'How fast does light travel?'")
  results = kb.search("How fast does light travel?", top_k=3)

  print("\nResults:")
  for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.content}")


def main():
  # Basic setup
  kb = basic_rag_setup()

  # Add documents
  add_documents(kb)

  # Search
  search_documents(kb)

  # Complete example
  complete_example()


if __name__ == "__main__":
  main()
