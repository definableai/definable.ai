"""
Document chunking strategies.

This example shows how to:
- Use TextChunker for simple text splitting
- Use RecursiveChunker for smarter splitting
- Configure chunk size and overlap
- Handle long documents

Requirements:
    No API keys required for this example.
"""

from typing import List

from definable.knowledge import (
  Document,
  Embedder,
  InMemoryVectorDB,
  Knowledge,
  RecursiveChunker,
  TextChunker,
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


# Sample long document for chunking
SAMPLE_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building
systems that learn from data. Instead of being explicitly programmed, these
systems improve their performance through experience.

## Types of Machine Learning

### Supervised Learning

Supervised learning involves training a model on labeled data. The model learns
to map inputs to outputs based on example input-output pairs. Common algorithms:

- Linear Regression: Used for predicting continuous values
- Logistic Regression: Used for binary classification
- Decision Trees: Tree-based models for classification and regression
- Support Vector Machines: Effective for high-dimensional spaces

### Unsupervised Learning

Unsupervised learning works with unlabeled data. The model tries to find
patterns and structure in the data without guidance. Key techniques include:

- Clustering: Grouping similar data points together
- Dimensionality Reduction: Reducing features while preserving information
- Anomaly Detection: Identifying unusual patterns in data

### Reinforcement Learning

Reinforcement learning involves an agent learning to make decisions by
interacting with an environment. The agent receives rewards or penalties
based on its actions and learns to maximize cumulative reward over time.

## Applications of Machine Learning

Machine learning has numerous real-world applications:

1. Image Recognition: Identifying objects, faces, and scenes in images
2. Natural Language Processing: Understanding and generating human language
3. Recommendation Systems: Suggesting products, content, or connections
4. Autonomous Vehicles: Self-driving cars and drones
5. Medical Diagnosis: Assisting doctors in detecting diseases

## Conclusion

Machine learning continues to evolve rapidly, with new techniques and
applications emerging constantly. Understanding the fundamentals is essential
for anyone working in technology today.
"""


def text_chunker_example():
  """Demonstrate TextChunker."""
  print("TextChunker Example")
  print("=" * 50)

  # Create TextChunker with specific settings
  chunker = TextChunker(
    chunk_size=500,  # Maximum characters per chunk
    chunk_overlap=50,  # Characters to overlap between chunks
  )

  # Create a document
  doc = Document(
    content=SAMPLE_DOCUMENT,
    meta_data={"source": "ml_guide.md"},
  )

  # Chunk the document
  chunks = chunker.chunk(doc)

  print(f"Original document: {len(SAMPLE_DOCUMENT)} characters")
  print(f"Chunk size: {chunker.chunk_size}, Overlap: {chunker.chunk_overlap}")
  print(f"Number of chunks: {len(chunks)}")
  print()

  # Show chunks
  for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1} ({len(chunk.content)} chars):")
    print(f"  {chunk.content[:80]}...")
    print(f"  Meta: chunk_index={chunk.chunk_index}, total={chunk.chunk_total}")
    print()


def recursive_chunker_example():
  """Demonstrate RecursiveChunker."""
  print("\n" + "=" * 50)
  print("RecursiveChunker Example")
  print("=" * 50)

  # Create RecursiveChunker
  # This chunker tries to split on natural boundaries first
  chunker = RecursiveChunker(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
  )

  doc = Document(
    content=SAMPLE_DOCUMENT,
    meta_data={"source": "ml_guide.md"},
  )

  chunks = chunker.chunk(doc)

  print(f"Number of chunks: {len(chunks)}")
  print()

  for i, chunk in enumerate(chunks):
    # Check if chunk starts/ends at natural boundaries
    starts_with_header = chunk.content.strip().startswith("#")
    ends_with_period = chunk.content.strip().endswith(".")

    print(f"Chunk {i + 1} ({len(chunk.content)} chars):")
    print(f"  Starts with header: {starts_with_header}")
    print(f"  Ends with period: {ends_with_period}")
    print(f"  Content: {chunk.content[:60]}...")
    print()


def compare_chunkers():
  """Compare TextChunker vs RecursiveChunker."""
  print("\n" + "=" * 50)
  print("Comparing Chunkers")
  print("=" * 50)

  doc = Document(content=SAMPLE_DOCUMENT)

  # Same settings for fair comparison
  chunk_size = 400
  chunk_overlap = 40

  text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  recursive_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

  text_chunks = text_chunker.chunk(doc)
  recursive_chunks = recursive_chunker.chunk(doc)

  print(f"TextChunker produced: {len(text_chunks)} chunks")
  print(f"RecursiveChunker produced: {len(recursive_chunks)} chunks")

  print("\nTextChunker chunk boundaries:")
  for i, chunk in enumerate(text_chunks[:3]):
    end_sample = chunk.content[-30:].replace("\n", "\\n")
    print(f"  {i + 1}. Ends with: ...{end_sample}")

  print("\nRecursiveChunker chunk boundaries:")
  for i, chunk in enumerate(recursive_chunks[:3]):
    end_sample = chunk.content[-30:].replace("\n", "\\n")
    print(f"  {i + 1}. Ends with: ...{end_sample}")


def chunking_with_knowledge_base():
  """Use chunking with a knowledge base."""
  print("\n" + "=" * 50)
  print("Chunking with Knowledge Base")
  print("=" * 50)

  embedder = MockEmbedder()
  vector_db = InMemoryVectorDB(dimensions=embedder.dimensions)
  chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)

  # Create knowledge base with chunker
  kb = Knowledge(
    vector_db=vector_db,
    embedder=embedder,
    chunker=chunker,
  )

  # Add the long document - it will be automatically chunked
  doc = Document(
    content=SAMPLE_DOCUMENT,
    meta_data={"source": "ml_guide.md"},
  )

  kb.add(doc)
  print("Added document (will be chunked automatically)")

  # Search for specific topics
  queries = [
    "What is supervised learning?",
    "What are the applications of machine learning?",
    "Tell me about reinforcement learning",
  ]

  for query in queries:
    print(f"\nQuery: {query}")
    results = kb.search(query, top_k=2)
    for i, result in enumerate(results, 1):
      print(f"  {i}. {result.content[:80]}...")
      chunk_idx = (result.chunk_index or 0) + 1
      print(f"     Chunk: {chunk_idx}/{result.chunk_total}")


def chunk_size_comparison():
  """Compare different chunk sizes."""
  print("\n" + "=" * 50)
  print("Chunk Size Comparison")
  print("=" * 50)

  doc = Document(content=SAMPLE_DOCUMENT)

  sizes = [200, 500, 1000]

  for size in sizes:
    chunker = TextChunker(chunk_size=size, chunk_overlap=size // 10)
    chunks = chunker.chunk(doc)
    avg_len = sum(len(c.content) for c in chunks) / len(chunks)
    print(f"Chunk size {size}: {len(chunks)} chunks, avg length {avg_len:.0f} chars")


def main():
  text_chunker_example()
  recursive_chunker_example()
  compare_chunkers()
  chunking_with_knowledge_base()
  chunk_size_comparison()


if __name__ == "__main__":
  main()
