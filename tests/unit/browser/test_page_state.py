"""Tests for PageState — console, error, network ring buffers."""

from __future__ import annotations

import pytest

from definable.browser.config import BrowserConfig
from definable.browser.page_state import ConsoleMessage, NetworkRequest, PageError, PageState


@pytest.fixture
def page_state():
  config = BrowserConfig(max_console_messages=5, max_page_errors=3, max_network_requests=5)
  return PageState(config)


class TestConsole:
  def test_empty(self, page_state: PageState):
    assert "No console messages" in page_state.format_console()

  def test_append(self, page_state: PageState):
    page_state._console.append(ConsoleMessage(type="log", text="hello", timestamp="2026-01-01T00:00:00Z"))
    result = page_state.format_console()
    assert "[LOG] hello" in result

  def test_ring_buffer_overflow(self, page_state: PageState):
    for i in range(10):
      page_state._console.append(ConsoleMessage(type="log", text=f"msg-{i}", timestamp=""))
    # max is 5, so only last 5 should remain
    assert len(page_state._console) == 5
    assert "msg-5" in page_state.format_console()

  def test_level_filter(self, page_state: PageState):
    page_state._console.append(ConsoleMessage(type="log", text="info msg", timestamp=""))
    page_state._console.append(ConsoleMessage(type="error", text="error msg", timestamp=""))
    page_state._console.append(ConsoleMessage(type="debug", text="debug msg", timestamp=""))

    errors_only = page_state.format_console(level="error")
    assert "error msg" in errors_only
    assert "info msg" not in errors_only
    assert "debug msg" not in errors_only

  def test_limit(self, page_state: PageState):
    for i in range(5):
      page_state._console.append(ConsoleMessage(type="log", text=f"msg-{i}", timestamp=""))
    result = page_state.format_console(limit=2)
    lines = result.strip().split("\n")
    assert len(lines) == 2


class TestErrors:
  def test_empty(self, page_state: PageState):
    assert "No page errors" in page_state.format_errors()

  def test_append(self, page_state: PageState):
    page_state._errors.append(PageError(message="TypeError: x is not a function", timestamp=""))
    result = page_state.format_errors()
    assert "TypeError" in result

  def test_ring_buffer_overflow(self, page_state: PageState):
    for i in range(10):
      page_state._errors.append(PageError(message=f"error-{i}", timestamp=""))
    assert len(page_state._errors) == 3

  def test_stack_truncated(self, page_state: PageState):
    stack = "  at foo (file.js:1:1)\n  at bar (file.js:2:2)\n  at baz (file.js:3:3)"
    page_state._errors.append(PageError(message="err", stack=stack, timestamp=""))
    result = page_state.format_errors()
    # Should show at most 2 stack lines
    assert "foo" in result
    assert "bar" in result


class TestNetwork:
  def test_empty(self, page_state: PageState):
    assert "No network requests" in page_state.format_network()

  def test_append(self, page_state: PageState):
    page_state._requests.append(NetworkRequest(id="r1", method="GET", url="https://api.example.com/data", timestamp=""))
    result = page_state.format_network()
    assert "[r1] GET" in result
    assert "api.example.com" in result

  def test_url_filter(self, page_state: PageState):
    page_state._requests.append(NetworkRequest(id="r1", method="GET", url="https://cdn.example.com/img.png", timestamp=""))
    page_state._requests.append(NetworkRequest(id="r2", method="POST", url="https://example.com/api/data", timestamp=""))

    result = page_state.format_network(url_filter="/api/")
    assert "r2" in result
    assert "r1" not in result

  def test_ring_buffer_overflow(self, page_state: PageState):
    for i in range(10):
      page_state._requests.append(NetworkRequest(id=f"r{i}", method="GET", url=f"https://example.com/{i}", timestamp=""))
    assert len(page_state._requests) == 5

  def test_status_and_failure(self, page_state: PageState):
    req = NetworkRequest(id="r1", method="GET", url="https://example.com", status=404, ok=False, timestamp="")
    page_state._requests.append(req)
    result = page_state.format_network()
    assert "404" in result

    req2 = NetworkRequest(id="r2", method="GET", url="https://example.com", failure_text="net::ERR_FAILED", ok=False, timestamp="")
    page_state._requests.append(req2)
    result = page_state.format_network()
    assert "ERR_FAILED" in result


class TestClear:
  def test_clear_all(self, page_state: PageState):
    page_state._console.append(ConsoleMessage(type="log", text="x", timestamp=""))
    page_state._errors.append(PageError(message="y", timestamp=""))
    page_state._requests.append(NetworkRequest(id="r1", method="GET", url="z", timestamp=""))
    page_state.clear()
    assert len(page_state._console) == 0
    assert len(page_state._errors) == 0
    assert len(page_state._requests) == 0
