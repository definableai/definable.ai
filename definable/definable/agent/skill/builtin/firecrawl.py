"""Firecrawl skill — LLM-optimized web scraping and crawling."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from definable.agent.skill.base import Skill
from definable.tool.decorator import tool


class Firecrawl(Skill):
  """Scrape and crawl websites with LLM-optimized markdown output.

  Requires ``firecrawl-py``: ``pip install firecrawl-py``

  Args:
      api_key: Firecrawl API key. Falls back to FIRECRAWL_API_KEY env var.
      api_url: Custom API URL. Default "https://api.firecrawl.dev".
      formats: Output formats. Default ["markdown"].
      max_pages: Maximum pages for crawling. Default 10.
      enable_crawl: Enable site crawling. Default False.
      enable_map: Enable site mapping. Default False.
      enable_search: Enable web search. Default False.

  Example::

      from definable.agent.skill.builtin import Firecrawl
      agent = Agent(model=model, skills=[Firecrawl(api_key="fc-...")])
  """

  name = "firecrawl"
  instructions = (
    "You have access to Firecrawl for web scraping. Use scrape_page to extract "
    "clean, LLM-friendly content from any URL. The content is returned as markdown. "
    "Use crawl_site to scrape multiple pages from a website."
  )

  def __init__(
    self,
    *,
    api_key: Optional[str] = None,
    api_url: str = "https://api.firecrawl.dev",
    formats: Optional[List[str]] = None,
    max_pages: int = 10,
    enable_crawl: bool = False,
    enable_map: bool = False,
    enable_search: bool = False,
  ):
    super().__init__()
    self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
    self._api_url = api_url
    self._formats = formats or ["markdown"]
    self._max_pages = max_pages
    self._enable_crawl = enable_crawl
    self._enable_map = enable_map
    self._enable_search = enable_search
    self._app: Any = None

  @property
  def app(self) -> Any:
    if self._app is not None:
      return self._app
    try:
      from firecrawl import FirecrawlApp
    except ImportError:
      raise ImportError("`firecrawl-py` not installed. Run: pip install firecrawl-py")
    if not self._api_key:
      raise ValueError("Firecrawl API key required. Set api_key or FIRECRAWL_API_KEY env var.")
    self._app = FirecrawlApp(api_key=self._api_key, api_url=self._api_url)
    return self._app

  @property
  def tools(self) -> list:
    skill = self
    result: list = []

    @tool
    def scrape_page(url: str) -> str:
      """Scrape a web page and return clean markdown content optimized for LLM consumption."""
      try:
        scrape_result = skill.app.scrape_url(url, params={"formats": skill._formats})
        if isinstance(scrape_result, dict):
          content = scrape_result.get("markdown") or scrape_result.get("content", "")
          metadata = scrape_result.get("metadata", {})
          return json.dumps({"url": url, "title": metadata.get("title", ""), "content": content[:50000]}, indent=2, default=str)
        return json.dumps({"url": url, "content": str(scrape_result)[:50000]}, default=str)
      except Exception as e:
        return json.dumps({"error": str(e)})

    result.append(scrape_page)

    if self._enable_crawl:

      @tool
      def crawl_site(url: str, max_pages: int = 0) -> str:
        """Crawl a website and return content from multiple pages."""
        try:
          limit = max_pages or skill._max_pages
          crawl_result = skill.app.crawl_url(url, params={"limit": limit, "scrapeOptions": {"formats": skill._formats}})
          if isinstance(crawl_result, dict) and "data" in crawl_result:
            pages = []
            for page in crawl_result["data"][:limit]:
              pages.append({
                "url": page.get("metadata", {}).get("sourceURL", ""),
                "title": page.get("metadata", {}).get("title", ""),
                "content": (page.get("markdown", "") or "")[:10000],
              })
            return json.dumps({"pages": pages, "count": len(pages)}, indent=2, default=str)
          return json.dumps({"result": str(crawl_result)[:10000]}, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(crawl_site)

    if self._enable_map:

      @tool
      def map_site(url: str) -> str:
        """Get a map of all pages/links on a website."""
        try:
          map_result = skill.app.map_url(url)
          if isinstance(map_result, dict):
            links = map_result.get("links", [])
            return json.dumps({"url": url, "links": links[:200], "total": len(links)}, indent=2)
          return json.dumps({"links": list(map_result)[:200] if map_result else []}, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(map_site)

    if self._enable_search:

      @tool
      def search_web(query: str, max_results: int = 5) -> str:
        """Search the web and return LLM-friendly results with content."""
        try:
          search_result = skill.app.search(query, params={"limit": max_results})
          if isinstance(search_result, dict) and "data" in search_result:
            results = []
            for r in search_result["data"][:max_results]:
              results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": (r.get("markdown", "") or r.get("description", ""))[:5000],
              })
            return json.dumps({"results": results}, indent=2, default=str)
          return json.dumps({"results": str(search_result)[:10000]}, default=str)
        except Exception as e:
          return json.dumps({"error": str(e)})

      result.append(search_web)

    return result
