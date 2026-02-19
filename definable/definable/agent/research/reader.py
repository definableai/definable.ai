"""Concurrent page reader — fetches and extracts text from URLs."""

import asyncio
from typing import List

from definable.agent.research.models import PageContent
from definable.utils.log import log_debug, log_warning


async def read_pages(
  urls: List[str],
  *,
  max_concurrent: int = 10,
  max_chars: int = 15000,
  timeout: float = 15.0,
) -> List[PageContent]:
  """Fetch multiple URLs concurrently and extract text content.

  Uses the existing URLReader from definable.knowledge.reader.url.
  Non-fatal on individual failures — returns PageContent with error set.

  Args:
    urls: List of URLs to fetch.
    max_concurrent: Maximum concurrent requests.
    max_chars: Truncate extracted text to this many characters.
    timeout: Per-request timeout in seconds.

  Returns:
    List of PageContent (one per URL, preserving order).
  """
  semaphore = asyncio.Semaphore(max_concurrent)

  async def _read_one(url: str) -> PageContent:
    async with semaphore:
      try:
        from definable.knowledge.reader.url import URLReader

        reader = URLReader()
        # Override timeout via reader config
        reader.config.timeout = timeout  # type: ignore[union-attr]
        docs = await reader.aread(url)
        if docs:
          doc = docs[0]
          content = doc.content or ""
          if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"
          title = doc.name or url
          log_debug(f"Read {len(content)} chars from {url}")
          return PageContent(url=url, title=title, content=content)
        return PageContent(url=url, error="No content extracted")
      except Exception as e:
        log_warning(f"Failed to read {url}: {e}")
        return PageContent(url=url, error=str(e))

  tasks = [_read_one(url) for url in urls]
  return await asyncio.gather(*tasks)
