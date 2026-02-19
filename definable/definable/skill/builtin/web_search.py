"""Web search skill — search the web and fetch page content.

Gives the agent the ability to search the web and read web pages.
Uses DuckDuckGo by default (no API key required), with an optional
provider parameter for custom search backends.

Requires: ``pip install duckduckgo-search``

Example:
    from definable.skill.builtin import WebSearch

    agent = Agent(
        model=model,
        skills=[WebSearch()],
    )
    output = agent.run("What are the latest Python 3.13 features?")
"""

from typing import Callable, Optional

from definable.skill.base import Skill
from definable.tool.decorator import tool


def _ddg_search(query: str, max_results: int = 5) -> str:
  """Search DuckDuckGo and return formatted results."""
  try:
    from duckduckgo_search import DDGS
  except ImportError:
    raise ImportError("WebSearch skill requires 'duckduckgo-search'. Install it with: pip install duckduckgo-search")

  results = []
  with DDGS() as ddgs:
    for r in ddgs.text(query, max_results=max_results):
      title = r.get("title", "")
      href = r.get("href", "")
      body = r.get("body", "")
      results.append(f"**{title}**\n{href}\n{body}")

  if not results:
    return "No results found."
  return "\n\n---\n\n".join(results)


def _ddg_fetch(url: str) -> str:
  """Fetch and extract text content from a URL."""
  try:
    from urllib.request import Request, urlopen
  except ImportError:
    return "Error: urllib not available."

  try:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DefinableBot/1.0)"})
    with urlopen(req, timeout=15) as resp:
      html = resp.read().decode("utf-8", errors="replace")

    # Simple HTML to text extraction
    import re

    # Remove scripts, styles, and tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    import html as html_mod

    text = html_mod.unescape(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to reasonable length
    if len(text) > 8000:
      text = text[:8000] + "... [truncated]"
    return text
  except Exception as e:
    return f"Error fetching URL: {e}"


class WebSearch(Skill):
  """Skill for searching the web and fetching page content.

  Uses DuckDuckGo by default (no API key required). Pass a custom
  ``search_fn`` to use a different search provider.

  Args:
    max_results: Maximum search results per query (default: 5).
    search_fn: Optional custom search function with signature
        ``(query: str, max_results: int) -> str``.
    enable_fetch: Whether to include the ``fetch_url`` tool (default: True).

  Example:
    # Default (DuckDuckGo, no API key)
    agent = Agent(model=model, skills=[WebSearch()])

    # Custom search provider
    def my_search(query: str, max_results: int = 5) -> str:
        return my_api.search(query, limit=max_results)

    agent = Agent(model=model, skills=[WebSearch(search_fn=my_search)])
  """

  name = "web_search"
  instructions = (
    "You have access to web search tools. Use search_web when you need "
    "current information, facts you're unsure about, or anything that "
    "may have changed after your training data. Use fetch_url to read "
    "the full content of a specific URL from search results. "
    "Always cite your sources when presenting information from search results."
  )

  def __init__(
    self,
    *,
    max_results: int = 5,
    search_fn: Optional[Callable[..., str]] = None,
    enable_fetch: bool = True,
  ):
    super().__init__()
    self._max_results = max_results
    self._search_fn = search_fn or _ddg_search
    self._enable_fetch = enable_fetch

  @property
  def tools(self) -> list:
    search_fn = self._search_fn
    max_results = self._max_results

    @tool
    def search_web(query: str) -> str:
      """Search the web for information on a topic.

      Args:
        query: The search query. Be specific and concise.

      Returns:
        Search results with titles, URLs, and snippets.
      """
      return search_fn(query, max_results)

    tools = [search_web]

    if self._enable_fetch:

      @tool
      def fetch_url(url: str) -> str:
        """Fetch and read the text content of a web page.

        Args:
          url: The full URL to fetch (e.g. "https://example.com/page").

        Returns:
          The extracted text content from the page.
        """
        return _ddg_fetch(url)

      tools.append(fetch_url)

    return tools
