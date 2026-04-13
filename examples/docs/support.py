from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

from definable.browser.base import BaseBrowser
from definable.knowledge.document import Document
from definable.knowledge.embedder import Embedder
from definable.model.metrics import Metrics


class MockVectorDB:
  """Small in-memory vector store for docs examples."""

  def __init__(self, embedder: Any = None):
    self.embedder = embedder
    self._docs: List[Document] = []

  def create(self) -> None:
    return None

  async def async_create(self) -> None:
    return None

  def content_hash_exists(self, content_hash: str) -> bool:
    return False

  def upsert_available(self) -> bool:
    return False

  def insert(self, content_hash: str, documents: List[Document]) -> None:
    self._docs.extend(documents)

  async def ainsert(self, content_hash: str, documents: List[Document]) -> None:
    self.insert(content_hash, documents)

  def search(self, query: str, limit: int = 10, filters: Any = None) -> List[Document]:
    return self._docs[:limit]

  async def asearch(self, query: str, limit: int = 10, filters: Any = None) -> List[Document]:
    return self.search(query, limit=limit, filters=filters)

  def delete_by_id(self, doc_id: str) -> None:
    self._docs = [doc for doc in self._docs if doc.id != doc_id]

  def delete(self) -> None:
    self._docs.clear()

  def count(self) -> int:
    return len(self._docs)


@dataclass
class MockEmbedder(Embedder):
  """Deterministic embedder for documentation examples."""

  dimensions: int = 4

  def _vectorize(self, text: str) -> list[float]:
    normalized = text.lower()
    return [
      float(normalized.count("agent") + normalized.count("workflow")),
      float(normalized.count("tool") + normalized.count("browser")),
      float(normalized.count("memory") + normalized.count("context")),
      float(normalized.count("knowledge") + normalized.count("document")),
    ]

  def get_embedding(self, text: str) -> list[float]:
    return self._vectorize(text)

  def get_embedding_and_usage(self, text: str) -> tuple[list[float], dict[str, Any]]:
    return self._vectorize(text), {"input_tokens": len(text.split())}

  async def async_get_embedding(self, text: str) -> list[float]:
    return self.get_embedding(text)

  async def async_get_embedding_and_usage(self, text: str) -> tuple[list[float], dict[str, Any]]:
    return self.get_embedding_and_usage(text)


@dataclass
class MockBrowser(BaseBrowser):
  """Deterministic browser implementation for documentation examples."""

  current_url: str = "https://example.com"
  title: str = "Mock Page"
  started: bool = False
  tabs: list[str] = field(default_factory=lambda: ["https://example.com"])
  cookies: list[dict[str, Any]] = field(default_factory=list)

  async def start(self) -> None:
    self.started = True

  async def stop(self) -> None:
    self.started = False

  async def snapshot(self, options: Any = None, selector: str | None = None, frame_selector: str | None = None) -> str:
    return '- heading "Mock Page" [level=1]\n- link "Docs" [ref=e1]\n[1 refs, 1 interactive]'

  async def screenshot(self, name: str = "screenshot", ref: str | None = None, full_page: bool = False) -> str:
    return f"/tmp/{name}.png"

  async def get_page_info(self) -> str:
    return f"URL: {self.current_url}\nTitle: {self.title}"

  async def navigate(self, url: str) -> str:
    self.current_url = url
    self.title = "Example Domain" if "example.com" in url else "Mock Page"
    if not self.tabs:
      self.tabs = [url]
    else:
      self.tabs[0] = url
    return f"Navigated to {url}"

  async def go_back(self) -> str:
    return "Went back"

  async def go_forward(self) -> str:
    return "Went forward"

  async def refresh(self) -> str:
    return f"Refreshed {self.current_url}"

  async def get_url(self) -> str:
    return self.current_url

  async def get_title(self) -> str:
    return self.title

  async def get_page_source(self, max_chars: int = 20000) -> str:
    return "<html><body>Mock browser page</body></html>"[:max_chars]

  async def get_text(self, ref_or_selector: str = "body") -> str:
    return "Mock browser page"

  async def get_attribute(self, ref_or_selector: str, attribute: str) -> str:
    return f"{attribute}-value"

  async def is_element_visible(self, ref_or_selector: str) -> str:
    return "true"

  async def click(self, ref_or_selector: str) -> str:
    return f"Clicked {ref_or_selector}"

  async def click_if_visible(self, ref_or_selector: str) -> str:
    return f"Clicked {ref_or_selector}"

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    return f"Clicked text {text}"

  async def hover(self, ref_or_selector: str) -> str:
    return f"Hovered {ref_or_selector}"

  async def drag(self, from_ref: str, to_ref: str) -> str:
    return f"Dragged {from_ref} to {to_ref}"

  async def type_text(self, ref_or_selector: str, text: str, submit: bool = False) -> str:
    return f"Typed into {ref_or_selector}: {text}"

  async def type_slowly(self, ref_or_selector: str, text: str, delay: float = 75.0) -> str:
    return f"Typed slowly into {ref_or_selector}: {text}"

  async def press_key(self, key: str) -> str:
    return f"Pressed {key}"

  async def press_keys(self, ref_or_selector: str, keys: str) -> str:
    return f"Pressed {keys} on {ref_or_selector}"

  async def clear_input(self, ref_or_selector: str) -> str:
    return f"Cleared {ref_or_selector}"

  async def select_option(self, ref_or_selector: str, text: str) -> str:
    return f"Selected {text} on {ref_or_selector}"

  async def check_element(self, ref_or_selector: str) -> str:
    return f"Checked {ref_or_selector}"

  async def uncheck_element(self, ref_or_selector: str) -> str:
    return f"Unchecked {ref_or_selector}"

  async def is_checked(self, ref_or_selector: str) -> str:
    return "false"

  async def set_value(self, ref_or_selector: str, value: str) -> str:
    return f"Set {ref_or_selector} to {value}"

  async def set_input_files(self, ref_or_selector: str, paths: list[str]) -> str:
    return f"Attached {len(paths)} files to {ref_or_selector}"

  async def fill_form(self, fields: list[dict[str, Any]]) -> str:
    return f"Filled {len(fields)} fields"

  async def execute_js(self, code: str, ref: str | None = None, timeout: float | None = None) -> str:
    return "JavaScript executed"

  async def highlight(self, ref_or_selector: str) -> str:
    return f"Highlighted {ref_or_selector}"

  async def remove_elements(self, selector: str) -> str:
    return f"Removed {selector}"

  async def scroll_down(self, amount: int = 3) -> str:
    return f"Scrolled down {amount}"

  async def scroll_up(self, amount: int = 3) -> str:
    return f"Scrolled up {amount}"

  async def scroll_to_element(self, ref_or_selector: str) -> str:
    return f"Scrolled to {ref_or_selector}"

  async def wait(self, seconds: float = 2.0) -> str:
    return f"Waited {seconds} seconds"

  async def wait_for_element(self, ref_or_selector: str, timeout: float = 10.0) -> str:
    return f"Found {ref_or_selector}"

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    return f"Found text {text}"

  async def wait_for(
    self,
    text: str | None = None,
    text_gone: str | None = None,
    selector: str | None = None,
    url: str | None = None,
    load_state: str | None = None,
    fn: str | None = None,
    timeout: float | None = None,
  ) -> str:
    return "Wait condition satisfied"

  async def open_tab(self, url: str = "") -> str:
    new_url = url or "about:blank"
    self.tabs.append(new_url)
    return f"Opened tab {len(self.tabs) - 1}: {new_url}"

  async def close_tab(self) -> str:
    if len(self.tabs) > 1:
      closed = self.tabs.pop()
      return f"Closed {closed}"
    return "No tab closed"

  async def get_tabs(self) -> str:
    return "\n".join(f"{idx}: {url}" for idx, url in enumerate(self.tabs))

  async def switch_to_tab(self, index: int) -> str:
    self.current_url = self.tabs[index]
    return f"Switched to tab {index}"

  async def get_cookies(self) -> str:
    return str(self.cookies)

  async def set_cookie(self, name: str, value: str) -> str:
    self.cookies.append({"name": name, "value": value, "domain": "example.com", "path": "/"})
    return f"Set cookie {name}"

  async def clear_cookies(self) -> str:
    self.cookies.clear()
    return "Cleared cookies"

  async def get_storage(self, key: str | None = None, kind: str = "local") -> str:
    return "{}"

  async def set_storage(self, key: str, value: str, kind: str = "local") -> str:
    return f"Set {kind}Storage[{key}]"

  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    action = "accepted" if accept else "dismissed"
    return f"Dialog {action}"

  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    return f"Set geolocation to {latitude},{longitude}"

  async def print_to_pdf(self, name: str = "page") -> str:
    return f"/tmp/{name}.pdf"

  async def get_console(self) -> str:
    return "No console messages"

  async def get_network(self) -> str:
    return "No captured network requests"

  async def get_errors(self) -> str:
    return "No page errors"


def mock_mcp_server_path() -> Path:
  return Path(__file__).with_name("mock_mcp_server.py")


def mock_model_response(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> MagicMock:
  """Build a model response object for MockModel side effects."""

  response = MagicMock()
  response.content = content
  response.tool_calls = tool_calls or []
  response.tool_executions = []
  response.response_usage = Metrics()
  response.reasoning_content = None
  response.citations = None
  response.images = None
  response.videos = None
  response.audios = None
  return response
