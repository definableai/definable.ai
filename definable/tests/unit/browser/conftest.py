"""Shared fixtures for browser unit tests."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class MockLocator:
  """Minimal mock for Playwright Locator."""

  _text: str = ""
  _visible: bool = True
  _checked: bool = False
  _attribute: str | None = None
  _calls: list[tuple[str, tuple, dict]] = field(default_factory=list)

  async def click(self, **kwargs: Any) -> None:
    self._calls.append(("click", (), kwargs))

  async def dblclick(self, **kwargs: Any) -> None:
    self._calls.append(("dblclick", (), kwargs))

  async def fill(self, value: str, **kwargs: Any) -> None:
    self._calls.append(("fill", (value,), kwargs))
    self._text = value

  async def press(self, key: str, **kwargs: Any) -> None:
    self._calls.append(("press", (key,), kwargs))

  async def press_sequentially(self, text: str, **kwargs: Any) -> None:
    self._calls.append(("press_sequentially", (text,), kwargs))

  async def type(self, text: str, **kwargs: Any) -> None:
    self._calls.append(("type", (text,), kwargs))

  async def hover(self, **kwargs: Any) -> None:
    self._calls.append(("hover", (), kwargs))

  async def drag_to(self, target: Any, **kwargs: Any) -> None:
    self._calls.append(("drag_to", (target,), kwargs))

  async def select_option(self, **kwargs: Any) -> None:
    self._calls.append(("select_option", (), kwargs))

  async def check(self, **kwargs: Any) -> None:
    self._calls.append(("check", (), kwargs))
    self._checked = True

  async def uncheck(self, **kwargs: Any) -> None:
    self._calls.append(("uncheck", (), kwargs))
    self._checked = False

  async def set_checked(self, checked: bool, **kwargs: Any) -> None:
    self._calls.append(("set_checked", (checked,), kwargs))
    self._checked = checked

  async def set_input_files(self, paths: list[str], **kwargs: Any) -> None:
    self._calls.append(("set_input_files", (paths,), kwargs))

  async def highlight(self) -> None:
    self._calls.append(("highlight", (), {}))

  async def screenshot(self, **kwargs: Any) -> bytes:
    self._calls.append(("screenshot", (), kwargs))
    return b"PNG"

  async def scroll_into_view_if_needed(self, **kwargs: Any) -> None:
    self._calls.append(("scroll_into_view_if_needed", (), kwargs))

  async def inner_text(self) -> str:
    return self._text or "mock text"

  async def get_attribute(self, name: str) -> str | None:
    return self._attribute

  async def is_visible(self) -> bool:
    return self._visible

  async def is_checked(self) -> bool:
    return self._checked

  async def wait_for(self, **kwargs: Any) -> None:
    self._calls.append(("wait_for", (), kwargs))

  async def evaluate(self, expr: Any, arg: Any = None) -> Any:
    self._calls.append(("evaluate", (expr,), {"arg": arg}))
    return None

  async def dispatch_event(self, event: str) -> None:
    self._calls.append(("dispatch_event", (event,), {}))

  def nth(self, index: int) -> "MockLocator":
    return self

  @property
  def first(self) -> "MockLocator":
    return self


class MockKeyboard:
  def __init__(self) -> None:
    self.calls: list[tuple[str, tuple, dict]] = []

  async def press(self, key: str, **kwargs: Any) -> None:
    self.calls.append(("press", (key,), kwargs))


class MockPage:
  """Minimal mock for Playwright Page."""

  def __init__(self, url: str = "https://example.com") -> None:
    self._url = url
    self._title = "Example"
    self._locator = MockLocator()
    self.keyboard = MockKeyboard()
    self._context = MockContext(self)
    self._calls: list[tuple[str, tuple, dict]] = []
    self._event_handlers: dict[str, list] = {}

  @property
  def url(self) -> str:
    return self._url

  @property
  def context(self) -> "MockContext":
    return self._context

  async def title(self) -> str:
    return self._title

  async def goto(self, url: str, **kwargs: Any) -> None:
    self._url = url
    self._calls.append(("goto", (url,), kwargs))

  async def go_back(self, **kwargs: Any) -> None:
    self._calls.append(("go_back", (), kwargs))

  async def go_forward(self, **kwargs: Any) -> None:
    self._calls.append(("go_forward", (), kwargs))

  async def reload(self, **kwargs: Any) -> None:
    self._calls.append(("reload", (), kwargs))

  async def content(self) -> str:
    return "<html><body>Mock</body></html>"

  async def screenshot(self, **kwargs: Any) -> bytes:
    self._calls.append(("screenshot", (), kwargs))
    return b"PNG"

  async def pdf(self, **kwargs: Any) -> bytes:
    self._calls.append(("pdf", (), kwargs))
    return b"PDF"

  async def evaluate(self, expr: Any, arg: Any = None) -> Any:
    self._calls.append(("evaluate", (expr,), {"arg": arg}))
    return {"scrollX": 0, "scrollY": 0, "scrollHeight": 1000, "viewportWidth": 1280, "viewportHeight": 720}

  async def set_viewport_size(self, size: dict) -> None:
    self._calls.append(("set_viewport_size", (size,), {}))

  async def emulate_media(self, **kwargs: Any) -> None:
    self._calls.append(("emulate_media", (), kwargs))

  async def wait_for_url(self, url: str, **kwargs: Any) -> None:
    self._calls.append(("wait_for_url", (url,), kwargs))

  async def wait_for_load_state(self, state: str, **kwargs: Any) -> None:
    self._calls.append(("wait_for_load_state", (state,), kwargs))

  async def wait_for_function(self, fn: str, **kwargs: Any) -> None:
    self._calls.append(("wait_for_function", (fn,), kwargs))

  async def close(self) -> None:
    self._calls.append(("close", (), {}))

  async def bring_to_front(self) -> None:
    self._calls.append(("bring_to_front", (), {}))

  def locator(self, selector: str, **kwargs: Any) -> MockLocator:
    return self._locator

  def frame_locator(self, selector: str) -> "MockFrameLocator":
    return MockFrameLocator(self._locator)

  def get_by_role(self, role: str, **kwargs: Any) -> MockLocator:
    return self._locator

  def get_by_text(self, text: str) -> MockLocator:
    return self._locator

  def on(self, event: str, handler: Any) -> None:
    if event not in self._event_handlers:
      self._event_handlers[event] = []
    self._event_handlers[event].append(handler)

  def once(self, event: str, handler: Any) -> None:
    self.on(event, handler)

  def remove_listener(self, event: str, handler: Any) -> None:
    if event in self._event_handlers:
      with contextlib.suppress(ValueError):
        self._event_handlers[event].remove(handler)


class MockFrameLocator:
  def __init__(self, locator: MockLocator) -> None:
    self._locator = locator

  def locator(self, selector: str) -> MockLocator:
    return self._locator

  def get_by_role(self, role: str, **kwargs: Any) -> MockLocator:
    return self._locator


class MockContext:
  """Minimal mock for Playwright BrowserContext."""

  def __init__(self, page: MockPage | None = None) -> None:
    self._pages = [page] if page else []
    self._cookies: list[dict] = []
    self._calls: list[tuple[str, tuple, dict]] = []

  @property
  def pages(self) -> list[MockPage]:
    return self._pages  # type: ignore[return-value]

  async def cookies(self) -> list[dict]:
    return self._cookies

  async def add_cookies(self, cookies: list[dict]) -> None:
    self._cookies.extend(cookies)

  async def clear_cookies(self) -> None:
    self._cookies.clear()

  async def set_offline(self, offline: bool) -> None:
    self._calls.append(("set_offline", (offline,), {}))

  async def set_extra_http_headers(self, headers: dict) -> None:
    self._calls.append(("set_extra_http_headers", (headers,), {}))

  async def set_geolocation(self, geo: dict | None) -> None:
    self._calls.append(("set_geolocation", (geo,), {}))

  async def grant_permissions(self, perms: list[str], **kwargs: Any) -> None:
    self._calls.append(("grant_permissions", (perms,), kwargs))

  async def new_page(self) -> MockPage:
    page = MockPage()
    self._pages.append(page)
    return page  # type: ignore[return-value]


class MockBrowser:
  """Minimal mock for Playwright Browser."""

  def __init__(self) -> None:
    self._context = MockContext()
    self._calls: list[str] = []

  @property
  def contexts(self) -> list[MockContext]:
    return [self._context]

  async def new_context(self, **kwargs: Any) -> MockContext:
    return self._context

  async def close(self) -> None:
    self._calls.append("close")


@pytest.fixture
def mock_page() -> MockPage:
  return MockPage()


@pytest.fixture
def mock_browser() -> MockBrowser:
  return MockBrowser()


@pytest.fixture
def browser_config():
  from definable.browser.config import BrowserConfig

  return BrowserConfig()
