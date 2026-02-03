"""
Reranking search results with CohereReranker.

This example shows how to:
- Use CohereReranker to improve search results
- Understand how reranking works
- Compare results with and without reranking

Requirements:
    export COHERE_API_KEY=...  # For Cohere reranking

Note: This example includes a mock reranker to work without API keys.
"""

import os
from typing import List

from definable.knowledge import Document, Embedder, InMemoryVectorDB, Knowledge, Reranker


class MockEmbedder(Embedder):
  """Simple mock embedder."""

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


class MockReranker(Reranker):
  """A mock reranker for demonstration.

  Uses simple keyword matching to simulate reranking.
  In production, use CohereReranker for much better results.
  """

  def rerank(
    self,
    query: str,
    documents: List[Document],
    top_k: int = 5,
  ) -> List[Document]:
    """Rerank documents based on keyword overlap."""
    query_words = set(query.lower().split())

    scored_docs = []
    for doc in documents:
      doc_words = set(doc.content.lower().split())
      # Calculate simple overlap score
      overlap = len(query_words & doc_words)
      score = overlap / max(len(query_words), 1)

      # Create new document with score
      reranked_doc = Document(
        id=doc.id,
        content=doc.content,
        meta_data=doc.meta_data,
        embedding=doc.embedding,
        reranking_score=score,
      )
      scored_docs.append((score, reranked_doc))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]

  async def arerank(
    self,
    query: str,
    documents: List[Document],
    top_k: int = 5,
  ) -> List[Document]:
    """Async version of rerank."""
    return self.rerank(query, documents, top_k)


def cohere_reranker_example():
  """Use Cohere reranker for improved results."""
  print("Cohere Reranker Example")
  print("=" * 50)

  # Check for API key
  if not os.getenv("COHERE_API_KEY"):
    print("COHERE_API_KEY not set, using mock reranker...")
    print("Set the environment variable to use Cohere reranking.\n")
    return None

  try:
    from definable.knowledge import CohereReranker

    reranker = CohereReranker(
      # api_key=os.getenv("COHERE_API_KEY"),  # Auto-detected
      model="rerank-english-v3.0",  # or rerank-multilingual-v3.0
    )

    print("Using Cohere model: rerank-english-v3.0")
    return reranker

  except ImportError:
    print("CohereReranker not available.")
    return None


def compare_with_without_reranking():
  """Compare search results with and without reranking."""
  print("\n" + "=" * 50)
  print("Comparing With/Without Reranking")
  print("=" * 50)

  embedder = MockEmbedder()
  reranker = MockReranker()

  # Create documents
  documents = [
    Document(content="Python is a popular programming language used in data science."),
    Document(content="The python snake is one of the largest reptiles in the world."),
    Document(content="Machine learning algorithms can be implemented in Python."),
    Document(content="Data analysis with Python uses libraries like pandas and numpy."),
    Document(content="Python programming is known for its simple and readable syntax."),
    Document(content="Pythons are non-venomous snakes that kill prey by constriction."),
  ]

  # Create knowledge base
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)
  kb = Knowledge(vector_db=vector_db, embedder=embedder)

  for doc in documents:
    kb.add(doc)

  # Query
  query = "Python programming for data science"
  print(f"\nQuery: {query}")

  # Without reranking
  print("\nWithout Reranking:")
  results_no_rerank = kb.search(query, top_k=4)
  for i, doc in enumerate(results_no_rerank, 1):
    print(f"  {i}. {doc.content[:60]}...")

  # With reranking
  print("\nWith Reranking:")
  results_no_rerank = kb.search(query, top_k=6)  # Get more, then rerank
  results_reranked = reranker.rerank(query, results_no_rerank, top_k=4)
  for i, doc in enumerate(results_reranked, 1):
    score = f"{doc.reranking_score:.3f}" if doc.reranking_score else "N/A"
    print(f"  {i}. (score: {score}) {doc.content[:50]}...")


def reranking_workflow():
  """Show the complete reranking workflow."""
  print("\n" + "=" * 50)
  print("Reranking Workflow")
  print("=" * 50)

  embedder = MockEmbedder()
  reranker = MockReranker()

  # Step 1: Initial vector search (retrieve more candidates)
  print("\n1. Initial Vector Search (broad retrieval):")
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  docs = [
    Document(content="How to train machine learning models with Python"),
    Document(content="Introduction to deep learning neural networks"),
    Document(content="Python basics for beginners"),
    Document(content="Advanced Python programming techniques"),
    Document(content="Machine learning with scikit-learn library"),
  ]

  for doc in docs:
    doc.embedding = embedder.get_embedding(doc.content)
  vector_db.add(docs)

  query = "machine learning Python tutorial"
  query_embedding = embedder.get_embedding(query)

  # Get more results than needed
  initial_results = vector_db.search(query_embedding, top_k=5)
  print(f"   Retrieved {len(initial_results)} candidates")

  # Step 2: Rerank
  print("\n2. Reranking (semantic relevance):")
  reranked = reranker.rerank(query, initial_results, top_k=3)

  for i, doc in enumerate(reranked, 1):
    score = doc.reranking_score or 0
    print(f"   {i}. Score: {score:.3f} - {doc.content[:40]}...")

  # Step 3: Use top results
  print("\n3. Final Results (top 3 after reranking):")
  for doc in reranked:
    print(f"   - {doc.content}")


def reranking_best_practices():
  """Best practices for using reranking."""
  print("\n" + "=" * 50)
  print("Reranking Best Practices")
  print("=" * 50)

  print("""
When to Use Reranking:
  ✓ When initial vector search returns mixed relevance
  ✓ For complex queries with multiple concepts
  ✓ When precision is more important than speed
  ✓ For user-facing search results

When NOT to Use Reranking:
  ✗ Simple, single-concept queries
  ✗ When latency is critical
  ✗ Small document sets (< 10 docs)
  ✗ When embedding model is already high quality

Configuration Tips:
  1. Retrieve 2-3x more documents than needed, then rerank
     - If you need top 5, retrieve top 15 first
  2. Use model appropriate for your language
     - rerank-english-v3.0 for English
     - rerank-multilingual-v3.0 for other languages
  3. Consider caching reranker results for repeated queries
  4. Monitor reranking latency in production

Cost Considerations:
  - Cohere reranking has per-query costs
  - Balance quality improvement vs. cost
  - Consider reranking only for important queries
""")


def knowledge_with_reranker():
  """Using reranker with Knowledge base."""
  print("\n" + "=" * 50)
  print("Knowledge Base with Reranker")
  print("=" * 50)

  embedder = MockEmbedder()
  reranker = MockReranker()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)

  # Create knowledge base with reranker
  kb = Knowledge(
    vector_db=vector_db,
    embedder=embedder,
    reranker=reranker,  # Attach reranker
  )

  # Add documents
  docs = [
    "Python is used for web development with Django and Flask.",
    "JavaScript runs in browsers and on servers with Node.js.",
    "Python's data science ecosystem includes pandas, numpy, and matplotlib.",
    "TypeScript adds static typing to JavaScript.",
    "Python machine learning uses TensorFlow and PyTorch frameworks.",
  ]

  for content in docs:
    kb.add(Document(content=content))

  # Search - reranking happens automatically if reranker is configured
  print("\nSearch with automatic reranking:")
  query = "Python for machine learning and data science"
  results = kb.search(query, top_k=3)

  for i, doc in enumerate(results, 1):
    score = f"{doc.reranking_score:.3f}" if doc.reranking_score else "N/A"
    print(f"  {i}. [{score}] {doc.content}")


def main():
  cohere_reranker_example()
  compare_with_without_reranking()
  reranking_workflow()
  reranking_best_practices()
  knowledge_with_reranker()


if __name__ == "__main__":
  main()
