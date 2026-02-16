"""Search provider factory and exports."""

from typing import Any, Awaitable, Callable, Dict, List, Optional

from definable.research.search.base import SearchProvider, SearchResult


class CallableSearchProvider:
  """Adapter that wraps a raw async callable as a SearchProvider."""

  def __init__(self, fn: Callable[[str, int], Awaitable[List[SearchResult]]]):
    self._fn = fn

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
    return await self._fn(query, max_results)


def create_search_provider(
  provider: str = "duckduckgo",
  config: Optional[Dict[str, Any]] = None,
  search_fn: Optional[Callable] = None,
) -> SearchProvider:
  """Create a search provider by name or from a custom callable.

  Args:
    provider: Provider name ("duckduckgo", "google", "serpapi").
    config: Provider-specific configuration (API keys, etc.).
    search_fn: Custom async search callable. Overrides provider name.

  Returns:
    A SearchProvider instance.
  """
  if search_fn is not None:
    return CallableSearchProvider(search_fn)  # type: ignore[return-value]

  if provider == "duckduckgo":
    from definable.research.search.duckduckgo import DuckDuckGoSearchProvider

    return DuckDuckGoSearchProvider()  # type: ignore[return-value]

  if provider == "google":
    from definable.research.search.google import create_google_provider

    return create_google_provider(config)  # type: ignore[return-value]

  if provider == "serpapi":
    from definable.research.search.serpapi import create_serpapi_provider

    return create_serpapi_provider(config)  # type: ignore[return-value]

  raise ValueError(f"Unknown search provider: {provider!r}. Use 'duckduckgo', 'google', or 'serpapi'.")


__all__ = [
  "SearchProvider",
  "SearchResult",
  "CallableSearchProvider",
  "create_search_provider",
]
