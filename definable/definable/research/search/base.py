"""Search provider protocol and base types."""

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable


@dataclass
class SearchResult:
  """A single search result from any provider."""

  title: str
  url: str
  snippet: str


@runtime_checkable
class SearchProvider(Protocol):
  """Protocol for pluggable search backends."""

  async def search(self, query: str, max_results: int = 10) -> List[SearchResult]: ...
