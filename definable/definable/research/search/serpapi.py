"""SerpAPI search backend via direct REST API."""

from typing import Any, Dict, List, Optional

from definable.research.search.base import SearchResult
from definable.utils.log import log_warning


class SerpAPISearchProvider:
  """Search provider using SerpAPI.

  Requires a SerpAPI key. No SDK dependency — uses httpx directly.

  Args:
    api_key: SerpAPI key.
  """

  def __init__(self, api_key: str):
    self._api_key = api_key

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
    """Search via SerpAPI."""
    import httpx

    params: Dict[str, Any] = {
      "api_key": self._api_key,
      "q": query,
      "num": max_results,
      "engine": "google",
    }

    results: List[SearchResult] = []
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          "https://serpapi.com/search.json",
          params=params,
          timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

      for item in data.get("organic_results", []):
        results.append(
          SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
          )
        )
    except Exception as e:
      log_warning(f"SerpAPI search failed for query '{query}': {e}")

    return results


def create_serpapi_provider(config: Optional[Dict[str, Any]] = None) -> SerpAPISearchProvider:
  """Factory for SerpAPISearchProvider from config dict."""
  config = config or {}
  api_key = config.get("api_key", "")
  if not api_key:
    raise ValueError("SerpAPI requires 'api_key' in search_provider_config")
  return SerpAPISearchProvider(api_key=api_key)
