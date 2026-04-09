from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

try:
  from examples.docs.support import MockEmbedder, MockVectorDB
except ImportError:
  from support import MockEmbedder, MockVectorDB

from definable.chunker import RecursiveChunker, TextChunker
from definable.knowledge import Document, FallbackEmbedder, Knowledge
from definable.knowledge.fts import FTSIndex, HybridSearchConfig, HybridSearcher
from definable.knowledge.reader import TextReader
from definable.reranker import Reranker


class FlakyEmbedder(MockEmbedder):
  def get_embedding(self, text: str) -> list[float]:
    raise RuntimeError("429 rate limit")

  def get_embedding_and_usage(self, text: str) -> tuple[list[float], dict[str, object]]:
    raise RuntimeError("429 rate limit")

  async def async_get_embedding(self, text: str) -> list[float]:
    raise RuntimeError("429 rate limit")

  async def async_get_embedding_and_usage(self, text: str) -> tuple[list[float], dict[str, object]]:
    raise RuntimeError("429 rate limit")


class KeywordReranker(Reranker):
  def rerank(self, query: str, documents: list[Document]) -> list[Document]:
    terms = set(query.lower().split())
    for document in documents:
      overlap = len(terms & set(document.content.lower().split()))
      document.reranking_score = float(overlap)
    return sorted(documents, key=lambda document: document.reranking_score or 0.0, reverse=True)

  async def arerank(self, query: str, documents: list[Document]) -> list[Document]:
    return self.rerank(query, documents)


async def _hybrid_summary() -> dict[str, object]:
  fts = FTSIndex()
  await fts.initialize()

  vector_results = [
    Document(content="Python powers the agent runtime.", reranking_score=0.8),
    Document(content="Memory stores previous interactions.", reranking_score=0.3),
  ]
  await fts.add(
    "demo",
    [
      Document(content="Python powers the agent runtime."),
      Document(content="Python 3.12 is supported for local development."),
    ],
  )

  merged = await HybridSearcher(
    fts_index=fts,
    config=HybridSearchConfig(vector_weight=0.7, text_weight=0.3),
  ).merge(vector_results, query="Python runtime", limit=3)

  await fts.close()

  return {
    "hybrid_top_content": merged[0].content,
    "hybrid_result_count": len(merged),
  }


async def _knowledge_summary() -> dict[str, object]:
  fts = FTSIndex()
  await fts.initialize()

  knowledge = Knowledge(
    vector_db=MockVectorDB(embedder=MockEmbedder(dimensions=4)),
    embedder=MockEmbedder(dimensions=4),
    fts_index=fts,
    hybrid_config=HybridSearchConfig(vector_weight=0.7, text_weight=0.3),
  )

  await knowledge.aadd("Python powers the agent runtime.", chunk=False)
  await knowledge.aadd("Memory stores previous interactions.", chunk=False)
  results = await knowledge.asearch("Python runtime", limit=2)
  context = knowledge.format_context(results[:1])

  await fts.close()

  return {
    "basic_search_content": results[0].content,
    "context_contains_python": "Python powers the agent runtime." in context,
  }


def main() -> dict[str, object]:
  document = Document(
    content="Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.",
    name="guide",
  )
  text_chunks = TextChunker(chunk_size=20, chunk_overlap=0, separator="\n\n").chunk(document)
  recursive_chunks = RecursiveChunker(chunk_size=20, chunk_overlap=5).chunk(document)

  fallback = FallbackEmbedder(providers=[FlakyEmbedder(dimensions=4), MockEmbedder(dimensions=4)])
  embedding = fallback.get_embedding("agent memory knowledge")

  with TemporaryDirectory() as tmp_dir:
    note_path = Path(tmp_dir) / "note.txt"
    note_path.write_text("Plain text content for knowledge ingestion.", encoding="utf-8")
    reader_output = TextReader().read(note_path)

  reranked = KeywordReranker().rerank(
    "python runtime",
    [
      Document(content="Python powers the runtime."),
      Document(content="Agents use tools."),
    ],
  )

  summary = {
    "text_chunk_count": len(text_chunks),
    "recursive_chunk_count": len(recursive_chunks),
    "reader_name": reader_output[0].name,
    "reader_content": reader_output[0].content,
    "fallback_dimensions": len(embedding),
    "reranked_top_content": reranked[0].content,
    **asyncio.run(_hybrid_summary()),
    **asyncio.run(_knowledge_summary()),
  }

  assert summary["text_chunk_count"] == 3
  assert summary["recursive_chunk_count"] >= 3
  assert summary["reader_name"] == "note.txt"
  assert summary["reader_content"] == "Plain text content for knowledge ingestion."
  assert summary["fallback_dimensions"] == 4
  assert summary["reranked_top_content"] == "Python powers the runtime."
  assert summary["hybrid_top_content"] == "Python powers the agent runtime."
  assert summary["context_contains_python"] is True

  return summary


if __name__ == "__main__":
  print(main())
