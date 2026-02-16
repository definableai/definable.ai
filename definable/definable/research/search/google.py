"""Google Custom Search backend via direct REST API."""

from typing import Any, Dict, List, Optional

from definable.research.search.base import SearchResult
from definable.utils.log import log_warning


class GoogleSearchProvider:
  """Search provider using Google Custom Search JSON API.

  Requires an API key and Custom Search Engine ID.
  No SDK dependency — uses httpx directly.

  Args:
    api_key: Google API key with Custom Search API enabled.
    cse_id: Custom Search Engine ID.
  """

  def __init__(self, api_key: str, cse_id: str):
    self._api_key = api_key
    self._cse_id = cse_id

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
    """Search Google Custom Search for the given query."""
    import httpx

    params: Dict[str, Any] = {
      "key": self._api_key,
      "cx": self._cse_id,
      "q": query,
      "num": min(max_results, 10),  # Google CSE max is 10 per request
    }

    results: List[SearchResult] = []
    try:
      async with httpx.AsyncClient() as client:
        response = await client.get(
          "https://www.googleapis.com/customsearch/v1",
          params=params,
          timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()

      for item in data.get("items", []):
        results.append(
          SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
          )
        )
    except Exception as e:
      log_warning(f"Google Custom Search failed for query '{query}': {e}")

    return results


def create_google_provider(config: Optional[Dict[str, Any]] = None) -> GoogleSearchProvider:
  """Factory for GoogleSearchProvider from config dict."""
  config = config or {}
  api_key = config.get("api_key", "")
  cse_id = config.get("cse_id", "")
  if not api_key or not cse_id:
    raise ValueError("Google Custom Search requires 'api_key' and 'cse_id' in search_provider_config")
  return GoogleSearchProvider(api_key=api_key, cse_id=cse_id)
