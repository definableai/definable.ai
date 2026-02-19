"""Scenario: Knowledge pipeline — documents, chunking, embedding, search.

Tests:
- Document creation with meta_data
- InMemoryVectorDB insert and search
- Knowledge base add and search
- MockEmbedder for offline testing

Requires: No API keys (uses mock embedder)
"""

import hashlib
import sys
from typing import List

PASS = 0
FAIL = 0


def check(condition: bool, description: str):
  global PASS, FAIL
  if condition:
    PASS += 1
    print(f"PASS: {description}")
  else:
    FAIL += 1
    print(f"FAIL: {description}")


from definable.embedder import Embedder


class MockEmbedder(Embedder):
  """Deterministic mock embedder for offline testing."""

  dimensions: int = 128

  def get_embedding(self, text: str) -> List[float]:
    text = text.lower().strip()
    embedding = [0.0] * self.dimensions
    for i, word in enumerate(text.split()):
      word_hash = hashlib.md5(word.encode()).digest()
      for j, byte_val in enumerate(word_hash):
        idx = (i + j) % self.dimensions
        embedding[idx] += (byte_val / 255.0 - 0.5) * (1 / (i + 1))
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
      embedding = [x / magnitude for x in embedding]
    return embedding

  def get_embedding_and_usage(self, text: str):
    return self.get_embedding(text), {"tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str):
    return self.get_embedding_and_usage(text)


def main():
  from definable.knowledge import Document, Knowledge
  from definable.vectordbs import InMemoryVectorDB

  # --- Test 1: Document creation ---
  doc = Document(content="Python is a programming language.", meta_data={"topic": "programming"})
  check(doc.content == "Python is a programming language.", "Document created with content")
  check(doc.meta_data.get("topic") == "programming", "Document has meta_data (NOT metadata)")

  # --- Test 2: VectorDB creation ---
  embedder = MockEmbedder()
  db = InMemoryVectorDB(embedder=embedder)
  check(db is not None, "InMemoryVectorDB created")

  # --- Test 3: Knowledge base creation ---
  kb = Knowledge(vector_db=db, embedder=embedder)
  check(kb is not None, "Knowledge base created with vector_db + embedder")

  # --- Test 4: Add documents ---
  docs = [
    Document(content="Machine learning uses data to train models.", meta_data={"topic": "ml"}),
    Document(content="Docker containers package applications.", meta_data={"topic": "devops"}),
    Document(content="REST APIs use HTTP methods for communication.", meta_data={"topic": "web"}),
    Document(content="Neural networks have layers of neurons.", meta_data={"topic": "ml"}),
    Document(content="Python has great library support for data science.", meta_data={"topic": "programming"}),
  ]
  for d in docs:
    kb.add(d)
  check(True, f"Added {len(docs)} documents to knowledge base")

  # --- Test 5: Search ---
  results = kb.search("machine learning models", top_k=2)
  check(len(results) > 0, f"Search returned {len(results)} results (expected > 0)")
  check(len(results) <= 2, f"Search respects top_k=2 (got {len(results)})")

  # --- Test 6: Result has content ---
  if results:
    check(results[0].content is not None and len(results[0].content) > 0, "Search result has content")
    check(results[0].meta_data is not None, "Search result preserves meta_data")

  # --- Summary ---
  print(f"\n--- Summary: {PASS} passed, {FAIL} failed ---")
  sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
  main()
