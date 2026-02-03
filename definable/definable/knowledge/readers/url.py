"""URL reader implementation."""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from definable.knowledge.document import Document
from definable.knowledge.readers.base import Reader


@dataclass
class URLReader(Reader):
  """Reader for web pages. Uses httpx + beautifulsoup."""

  user_agent: str = "Mozilla/5.0 (compatible; DefinableBot/1.0)"
  extract_links: bool = False
  remove_tags: List[str] = None

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.remove_tags is None:
      self.remove_tags = ["script", "style", "nav", "footer", "header", "aside"]

  def read(self, source: Union[str, Path]) -> List[Document]:
    """Read a web page and return as Document."""
    try:
      import httpx
      from bs4 import BeautifulSoup
    except ImportError:
      raise ImportError("httpx and beautifulsoup4 required. Run: pip install httpx beautifulsoup4")

    url = str(source)
    response = httpx.get(
      url,
      headers={"User-Agent": self.user_agent},
      timeout=self.config.timeout,
      follow_redirects=True,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

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
        links.append(a["href"])

    meta_data = {
      "url": url,
      "status_code": response.status_code,
      "content_type": response.headers.get("content-type", ""),
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
    try:
      import httpx
      from bs4 import BeautifulSoup
    except ImportError:
      raise ImportError("httpx and beautifulsoup4 required. Run: pip install httpx beautifulsoup4")

    url = str(source)
    async with httpx.AsyncClient() as client:
      response = await client.get(
        url,
        headers={"User-Agent": self.user_agent},
        timeout=self.config.timeout,
        follow_redirects=True,
      )
      response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

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
        links.append(a["href"])

    meta_data = {
      "url": url,
      "status_code": response.status_code,
      "content_type": response.headers.get("content-type", ""),
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
