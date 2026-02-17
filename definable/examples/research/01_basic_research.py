"""Standalone deep research — multi-wave web search with CKU compression."""

import asyncio
import os

from definable.models.openai import OpenAIChat
from definable.research import DeepResearch, DeepResearchConfig
from definable.research.search import create_search_provider


async def main():
  model = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
  search_provider = create_search_provider("duckduckgo")

  researcher = DeepResearch(
    model=model,
    search_provider=search_provider,
    config=DeepResearchConfig(depth="quick"),  # 1 wave, 8 sources
  )

  result = await researcher.arun("What are the latest developments in quantum computing?")

  print("=== Research Context (for system prompt) ===")
  print(result.context[:500], "..." if len(result.context) > 500 else "")

  print("\n=== Metrics ===")
  print(f"  Waves: {result.metrics.waves_executed}")
  print(f"  Sources read: {result.metrics.total_sources_read}")
  print(f"  Facts extracted: {result.metrics.total_facts_extracted}")
  print(f"  Unique facts: {result.metrics.unique_facts}")
  print(f"  Contradictions: {result.metrics.contradictions_found}")
  print(f"  Time: {result.metrics.total_time_ms:.0f}ms")

  print(f"\n=== Sources ({len(result.sources)}) ===")
  for src in result.sources[:5]:
    print(f"  [{src.relevance_score:.2f}] {src.title} — {src.url}")


if __name__ == "__main__":
  asyncio.run(main())
