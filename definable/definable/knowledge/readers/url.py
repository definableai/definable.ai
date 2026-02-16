"""URL reader implementation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

from definable.knowledge.document import Document
from definable.knowledge.readers.base import Reader


@dataclass
class URLReader(Reader):
  """Reader for web pages. Uses httpx + beautifulsoup."""

  user_agent: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
  )
  impersonate: Optional[str] = "chrome"
  extract_links: bool = False
  remove_tags: List[str] = field(default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside"])

  @property
  def _request_headers(self) -> dict:
    return {
      "User-Agent": self.user_agent,
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9",
      "Accept-Encoding": "gzip, deflate, br",
      "Connection": "keep-alive",
      "Upgrade-Insecure-Requests": "1",
    }

  def __post_init__(self) -> None:
    super().__post_init__()

  def _fetch(self, url: str) -> Tuple[str, int, str]:
    """Fetch URL content. Tries curl-cffi with TLS impersonation first, falls back to httpx."""
    if self.impersonate:
      try:
        from curl_cffi.requests import Session

        with Session(impersonate=self.impersonate) as s:  # type: ignore[arg-type]
          r = s.get(url, timeout=self.config.timeout, allow_redirects=True)
          r.raise_for_status()
          return r.text, r.status_code, r.headers.get("content-type", "")
      except ImportError:
        pass
    import httpx

    r = httpx.get(url, headers=self._request_headers, timeout=self.config.timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text, r.status_code, r.headers.get("content-type", "")

  async def _afetch(self, url: str) -> Tuple[str, int, str]:
    """Async fetch URL content. Tries curl-cffi with TLS impersonation first, falls back to httpx."""
    if self.impersonate:
      try:
        from curl_cffi.requests import AsyncSession

        async with AsyncSession(impersonate=self.impersonate) as s:  # type: ignore[arg-type]
          r = await s.get(url, timeout=self.config.timeout, allow_redirects=True)
          r.raise_for_status()
          return r.text, r.status_code, r.headers.get("content-type", "")
      except ImportError:
        pass
    import httpx

    async with httpx.AsyncClient() as client:
      r = await client.get(url, headers=self._request_headers, timeout=self.config.timeout, follow_redirects=True)
      r.raise_for_status()
      return r.text, r.status_code, r.headers.get("content-type", "")

  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read a web page and return as Document."""
    from bs4 import BeautifulSoup

    url = str(source)
    html, status_code, content_type = self._fetch(url)

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(self.remove_tags):
      tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Get title
    title = soup.title.string if soup.title else url

    # Extract links if requested
    links: List[str] = []
    if self.extract_links:
      for a in soup.find_all("a", href=True):
        href = cast(Any, a).get("href")
        if href is not None:
          links.append(str(href))

    meta_data = {
      "url": url,
      "status_code": status_code,
      "content_type": content_type,
      **self.config.metadata,
    }

    if self.extract_links:
      meta_data["links"] = links

    return [
      Document(
        content=text,
        name=str(title)[:200] if title else url,
        source=url,
        source_type="url",
        size=len(text),
        meta_data=meta_data,
      )
    ]

  async def aread(self, source: Union[str, Path]) -> List[Document]:
    """Async read a web page."""
    from bs4 import BeautifulSoup

    url = str(source)
    html, status_code, content_type = await self._afetch(url)

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(self.remove_tags):
      tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Get title
    title = soup.title.string if soup.title else url

    # Extract links if requested
    links: List[str] = []
    if self.extract_links:
      for a in soup.find_all("a", href=True):
        href = cast(Any, a).get("href")
        if href is not None:
          links.append(str(href))

    meta_data = {
      "url": url,
      "status_code": status_code,
      "content_type": content_type,
      **self.config.metadata,
    }

    if self.extract_links:
      meta_data["links"] = links

    return [
      Document(
        content=text,
        name=str(title)[:200] if title else url,
        source=url,
        source_type="url",
        size=len(text),
        meta_data=meta_data,
      )
    ]

  def can_read(self, source: Union[str, Path]) -> bool:
    """Check if this reader can handle the source."""
    url = str(source).lower()
    return url.startswith("http://") or url.startswith("https://")
