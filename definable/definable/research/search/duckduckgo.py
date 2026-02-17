"""DuckDuckGo search backend via duckduckgo-search package."""

import asyncio
from typing import List

from definable.research.search.base import SearchResult
from definable.utils.log import log_warning


class DuckDuckGoSearchProvider:
  """Search provider using DuckDuckGo (no API key required).

  Wraps the synchronous DDGS().text() in asyncio.to_thread().
  Requires: pip install duckduckgo-search
  """

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
    """Search DuckDuckGo for the given query."""
    return await asyncio.to_thread(self._search_sync, query, max_results)

  def _search_sync(self, query: str, max_results: int) -> List[SearchResult]:
    try:
      from duckduckgo_search import DDGS
    except ImportError:
      raise ImportError("DuckDuckGo search requires 'duckduckgo-search'. Install it with: pip install duckduckgo-search")

    results: List[SearchResult] = []
    try:
      with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
          results.append(
            SearchResult(
              title=r.get("title", ""),
              url=r.get("href", ""),
              snippet=r.get("body", ""),
            )
          )
    except Exception as e:
      log_warning(f"DuckDuckGo search failed for query '{query}': {e}")

    return results
