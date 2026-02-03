"""
Custom embedders: OpenAI and VoyageAI.

This example shows how to:
- Use VoyageAI embeddings
- Use OpenAI embeddings
- Create a custom embedder
- Compare embedding dimensions

Requirements:
    export VOYAGE_API_KEY=pa-...   # For VoyageAI
    export OPENAI_API_KEY=sk-...   # For OpenAI
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from definable.knowledge import Document, Embedder, InMemoryVectorDB, Knowledge


def voyageai_embedder_example():
  """Use VoyageAI embeddings."""
  print("VoyageAI Embedder")
  print("=" * 50)

  # Check if API key is available
  if not os.getenv("VOYAGE_API_KEY"):
    print("VOYAGE_API_KEY not set, skipping VoyageAI example...")
    print("Set the environment variable to use VoyageAI embeddings.")
    return None

  from definable.knowledge import VoyageAIEmbedder

  # Create VoyageAI embedder
  embedder = VoyageAIEmbedder(
    # api_key=os.getenv("VOYAGE_API_KEY"),  # Auto-detected from env
    id="voyage-3",  # Model ID
    dimensions=1024,  # voyage-3 outputs 1024 dimensions
  )

  print(f"Model: {embedder.id}")
  print(f"Dimensions: {embedder.dimensions}")

  # Test embedding
  text = "Machine learning is a subset of artificial intelligence."
  embedding = embedder.get_embedding(text)

  print(f"\nTest text: {text}")
  print(f"Embedding length: {len(embedding)}")
  print(f"First 5 values: {embedding[:5]}")

  return embedder


def openai_embedder_example():
  """Use OpenAI embeddings."""
  print("\n" + "=" * 50)
  print("OpenAI Embedder")
  print("=" * 50)

  # Check if API key is available
  if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY not set, skipping OpenAI example...")
    print("Set the environment variable to use OpenAI embeddings.")
    return None

  # Note: OpenAIEmbedder may need to be imported differently
  # depending on your definable version
  try:
    from definable.knowledge.embedders.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder(
      id="text-embedding-3-large",
      # dimensions=3072,  # Optional: specify output dimensions
    )

    print(f"Model: {embedder.id}")
    print(f"Dimensions: {embedder.dimensions}")

    # Test embedding
    text = "Natural language processing enables computers to understand text."
    embedding = embedder.get_embedding(text)

    print(f"\nTest text: {text}")
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    return embedder
  except ImportError:
    print("OpenAIEmbedder not available in this version.")
    return None


class CustomEmbedder(Embedder):
  """A custom embedder implementation.

  This example shows how to create your own embedder,
  perhaps wrapping a different embedding service.
  """

  dimensions: int = 256

  def __init__(self, normalize: bool = True):
    self.normalize = normalize

  def get_embedding(self, text: str) -> List[float]:
    """Generate embedding for text."""
    import hashlib
    import math

    # This is a simple hash-based embedding for demonstration
    # In practice, you would call an embedding API here

    # Create base embedding from text hash
    text_normalized = text.lower().strip()
    embedding = [0.0] * self.dimensions

    # Use word-level features
    words = text_normalized.split()
    for i, word in enumerate(words):
      word_hash = hashlib.sha256(word.encode()).digest()
      for j, byte in enumerate(word_hash[:32]):
        idx = (i * 32 + j) % self.dimensions
        embedding[idx] += (byte / 127.5 - 1.0) / (i + 1)

    # Normalize if requested
    if self.normalize:
      magnitude = math.sqrt(sum(x**2 for x in embedding))
      if magnitude > 0:
        embedding = [x / magnitude for x in embedding]

    return embedding

  def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Get embedding with usage statistics."""
    embedding = self.get_embedding(text)
    usage = {
      "tokens": len(text.split()),
      "characters": len(text),
    }
    return embedding, usage

  async def async_get_embedding(self, text: str) -> List[float]:
    """Async version of get_embedding."""
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Async version of get_embedding_and_usage."""
    return self.get_embedding_and_usage(text)


def custom_embedder_example():
  """Use a custom embedder."""
  print("\n" + "=" * 50)
  print("Custom Embedder")
  print("=" * 50)

  embedder = CustomEmbedder(normalize=True)

  print(f"Dimensions: {embedder.dimensions}")

  # Test embedding
  text = "Custom embedders allow flexibility in embedding generation."
  embedding, usage = embedder.get_embedding_and_usage(text)

  print(f"\nTest text: {text}")
  print(f"Embedding length: {len(embedding)}")
  print(f"Usage: {usage}")
  print(f"First 5 values: {[f'{v:.4f}' for v in embedding[:5]]}")

  return embedder


def embedder_with_knowledge_base():
  """Use different embedders with knowledge bases."""
  print("\n" + "=" * 50)
  print("Embedder with Knowledge Base")
  print("=" * 50)

  # Use custom embedder (always available)
  embedder = CustomEmbedder()

  # Create knowledge base
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)
  kb = Knowledge(vector_db=vector_db, embedder=embedder)

  # Add documents
  documents = [
    "Python is a versatile programming language.",
    "JavaScript is the language of the web.",
    "Rust provides memory safety without garbage collection.",
    "Go is designed for simplicity and efficiency.",
  ]

  for content in documents:
    kb.add(Document(content=content))

  print(f"Added {len(documents)} documents")

  # Search
  query = "What language is good for web development?"
  results = kb.search(query, top_k=2)

  print(f"\nQuery: {query}")
  for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.content}")


def compare_embedders():
  """Compare different embedders."""
  print("\n" + "=" * 50)
  print("Comparing Embedders")
  print("=" * 50)

  # Test text
  text = "Deep learning uses neural networks with many layers."

  embedders = []

  # Custom embedder (always available)
  custom = CustomEmbedder()
  embedders.append(("Custom", custom))

  # Try VoyageAI
  if os.getenv("VOYAGE_API_KEY"):
    try:
      from definable.knowledge import VoyageAIEmbedder

      voyage = VoyageAIEmbedder()
      embedders.append(("VoyageAI", voyage))
    except Exception as e:
      print(f"VoyageAI not available: {e}")

  # Compare
  print(f"\nText: {text}")
  print("\nEmbedder comparison:")
  for name, embedder in embedders:
    embedding = embedder.get_embedding(text)
    magnitude = sum(x**2 for x in embedding) ** 0.5
    print(f"  {name}:")
    print(f"    Dimensions: {len(embedding)}")
    print(f"    Magnitude: {magnitude:.4f}")
    print(f"    First 3: {[f'{v:.4f}' for v in embedding[:3]]}")


def main():
  voyageai_embedder_example()
  openai_embedder_example()
  custom_embedder_example()
  embedder_with_knowledge_base()
  compare_embedders()


if __name__ == "__main__":
  main()
