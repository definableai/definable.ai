#!/usr/bin/env python3
"""Review test: Knowledge (RAG) through Agent composition.

Tests Document, Knowledge, InMemoryVectorDB, MockEmbedder, and Agent + Knowledge.
"""

import sys, os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

passed, failed, skipped = 0, 0, 0


def check(name, condition, error=""):
  global passed, failed
  if condition:
    print(f"✅ PASS: {name}")
    passed += 1
  else:
    print(f"❌ FAIL: {name} — {error}")
    failed += 1


try:
  from definable.knowledge import Knowledge, Document
  from definable.vectordb import InMemoryVectorDB
  from definable.embedder import Embedder
  from definable.agent import Agent, MockModel

  check("Import knowledge + agent", True)
except Exception as e:
  check("Import knowledge + agent", False, str(e))
  print(f"\n{'=' * 60}\nRESULT: {passed} passed | {failed} failed | 0 skipped")
  sys.exit(1)


# ── MockEmbedder ────────────────────────────────────────────────
class MockEmbedder(Embedder):
  dimensions: int = 128

  def get_embedding(self, text: str) -> List[float]:
    import hashlib

    embedding = [0.0] * self.dimensions
    for i, word in enumerate(text.lower().split()):
      h = hashlib.md5(word.encode()).digest()
      for j, byte in enumerate(h):
        embedding[(i + j) % self.dimensions] += byte / 255.0 - 0.5
    mag = sum(x**2 for x in embedding) ** 0.5
    return [x / mag for x in embedding] if mag > 0 else embedding

  def get_embedding_and_usage(self, text):
    return self.get_embedding(text), {"tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> List[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text):
    return self.get_embedding_and_usage(text)


# ── Document ────────────────────────────────────────────────────
try:
  doc = Document(content="Test content about vacation policy.")
  check("Document(content=...) constructs", True)
except Exception as e:
  check("Document construction", False, str(e))

try:
  doc = Document(content="With metadata", meta_data={"source": "test.pdf"})
  check("Document with meta_data constructs", True)
  check("Document.meta_data accessible", doc.meta_data.get("source") == "test.pdf", str(doc.meta_data))
except Exception as e:
  check("Document meta_data", False, str(e))


# ── InMemoryVectorDB ───────────────────────────────────────────
try:
  vdb = InMemoryVectorDB(dimensions=128)
  check("InMemoryVectorDB(dimensions=128) constructs", True)
except Exception as e:
  check("InMemoryVectorDB construction", False, str(e))


# ── Knowledge pipeline ─────────────────────────────────────────
try:
  embedder = MockEmbedder()
  vdb = InMemoryVectorDB(dimensions=embedder.dimensions)
  kb = Knowledge(vector_db=vdb, embedder=embedder)
  check("Knowledge(vector_db, embedder) constructs", True)
except Exception as e:
  check("Knowledge construction", False, str(e))

try:
  docs = [
    Document(content="Vacation policy: 20 days PTO per year.", meta_data={"topic": "vacation"}),
    Document(content="Remote work: 3 days per week allowed.", meta_data={"topic": "remote"}),
    Document(content="Health insurance: company pays 80%.", meta_data={"topic": "benefits"}),
  ]
  for doc in docs:
    kb.add(doc)
  check("Knowledge.add() 3 documents", True)
except Exception as e:
  check("Knowledge.add()", False, str(e))

try:
  results = kb.search("vacation days", limit=2)
  check("Knowledge.search() returns results", results is not None and len(results) > 0, f"results: {len(results) if results else 0}")
except Exception as e:
  check("Knowledge.search()", False, str(e))


# ── Agent + Knowledge ───────────────────────────────────────────
try:
  embedder2 = MockEmbedder()
  vdb2 = InMemoryVectorDB(dimensions=embedder2.dimensions)
  kb2 = Knowledge(vector_db=vdb2, embedder=embedder2, top_k=3)
  kb2.add(Document(content="Company founded in 2018 by Sarah Chen."))

  agent = Agent(model=MockModel(), knowledge=kb2)
  check("Agent(knowledge=kb) constructs", True)

  output = agent.run("Tell me about the company")
  check("Agent + Knowledge runs", output.content is not None, "no content")
except Exception as e:
  check("Agent + Knowledge", False, str(e))


# ── Agent knowledge=True should raise ──────────────────────────
try:
  Agent(model=MockModel(), knowledge=True)
  check("Agent(knowledge=True) raises ValueError", False, "should have raised")
except ValueError:
  check("Agent(knowledge=True) raises ValueError", True)
except Exception as e:
  check("Agent(knowledge=True) error", False, f"{type(e).__name__}: {e}")


print(f"\n{'=' * 60}")
print(f"RESULT: {passed} passed | {failed} failed | 0 skipped")
sys.exit(1 if failed else 0)
