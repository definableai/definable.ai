"""Page state tracking — console, errors, network via Playwright event listeners.

Uses collections.deque(maxlen=N) for O(1) ring buffers (vs JS Array.shift() O(n)).
Ported from openclaw pw-session.ts ensurePageState().
"""

from __future__ import annotations

import weakref
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from playwright.async_api import Page, Request, Response

  from definable.browser.config import BrowserConfig


def _now_iso() -> str:
  return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class ConsoleMessage:
  """A captured browser console message."""

  type: str  # "log", "error", "warning", "info", "debug"
  text: str
  timestamp: str  # ISO 8601


@dataclass
class PageError:
  """A captured browser page error."""

  message: str
  stack: str | None = None
  timestamp: str = ""  # ISO 8601


@dataclass
class NetworkRequest:
  """A captured browser network request with optional response data."""

  id: str  # "r1", "r2", ...
  method: str
  url: str
  resource_type: str | None = None
  status: int | None = None
  ok: bool | None = None
  failure_text: str | None = None
  timestamp: str = ""  # ISO 8601


class PageState:
  """Tracks console, errors, network via Playwright page event listeners.

  Attach to a Page to start capturing events. Detach to stop.
  Ring buffers auto-evict oldest entries when full.
  """

  def __init__(self, config: "BrowserConfig") -> None:
    self._console: deque[ConsoleMessage] = deque(maxlen=config.max_console_messages)
    self._errors: deque[PageError] = deque(maxlen=config.max_page_errors)
    self._requests: deque[NetworkRequest] = deque(maxlen=config.max_network_requests)
    self._request_ids: weakref.WeakKeyDictionary[object, str] = weakref.WeakKeyDictionary()
    self._next_request_id: int = 0
    self._page: Page | None = None
    self._attached = False

  def attach(self, page: "Page") -> None:
    """Register event listeners on a Playwright Page."""
    if self._attached:
      self.detach()

    self._page = page
    self._attached = True

    page.on("console", self._on_console)
    page.on("pageerror", self._on_page_error)
    page.on("request", self._on_request)
    page.on("response", self._on_response)
    page.on("requestfailed", self._on_request_failed)

  def detach(self) -> None:
    """Remove all event listeners."""
    if self._page and self._attached:
      try:
        self._page.remove_listener("console", self._on_console)
        self._page.remove_listener("pageerror", self._on_page_error)
        self._page.remove_listener("request", self._on_request)
        self._page.remove_listener("response", self._on_response)
        self._page.remove_listener("requestfailed", self._on_request_failed)
      except Exception:
        pass
    self._page = None
    self._attached = False

  # -- Event handlers (bound methods for easy removal) ---------------------

  def _on_console(self, msg: object) -> None:
    entry = ConsoleMessage(
      type=getattr(msg, "type", "log"),
      text=getattr(msg, "text", str(msg)),
      timestamp=_now_iso(),
    )
    self._console.append(entry)

  def _on_page_error(self, error: object) -> None:
    entry = PageError(
      message=getattr(error, "message", str(error)),
      stack=getattr(error, "stack", None),
      timestamp=_now_iso(),
    )
    self._errors.append(entry)

  def _on_request(self, request: "Request") -> None:
    self._next_request_id += 1
    rid = f"r{self._next_request_id}"
    self._request_ids[request] = rid
    entry = NetworkRequest(
      id=rid,
      method=request.method,
      url=request.url,
      resource_type=request.resource_type,
      timestamp=_now_iso(),
    )
    self._requests.append(entry)

  def _on_response(self, response: "Response") -> None:
    request = response.request
    rid = self._request_ids.get(request)
    if not rid:
      return
    rec = self._find_request(rid)
    if rec:
      rec.status = response.status
      rec.ok = response.ok

  def _on_request_failed(self, request: "Request") -> None:
    rid = self._request_ids.get(request)
    if not rid:
      return
    rec = self._find_request(rid)
    if rec:
      failure = request.failure
      rec.failure_text = failure if isinstance(failure, str) else str(failure) if failure else None
      rec.ok = False

  def _find_request(self, rid: str) -> NetworkRequest | None:
    """Find a request by ID (search from end for efficiency)."""
    for i in range(len(self._requests) - 1, -1, -1):
      if self._requests[i].id == rid:
        return self._requests[i]
    return None

  # -- Query methods -------------------------------------------------------

  _LEVEL_ORDER = {"error": 0, "warning": 1, "info": 2, "log": 3, "debug": 4}

  def format_console(self, limit: int = 50, level: str | None = None) -> str:
    """Format console messages for AI consumption.

    Args:
        limit: Max number of messages to return.
        level: Minimum level filter (error > warning > info > log > debug).
    """
    messages = list(self._console)
    if level:
      min_order = self._LEVEL_ORDER.get(level, 3)
      messages = [m for m in messages if self._LEVEL_ORDER.get(m.type, 3) <= min_order]
    messages = messages[-limit:]
    if not messages:
      return "No console messages."
    lines = [f"[{m.type.upper()}] {m.text}" for m in messages]
    return "\n".join(lines)

  def format_errors(self, limit: int = 20) -> str:
    """Format page errors for AI consumption."""
    errors = list(self._errors)[-limit:]
    if not errors:
      return "No page errors."
    lines: list[str] = []
    for e in errors:
      line = f"[ERROR] {e.message}"
      if e.stack:
        # Just first 2 lines of stack
        stack_lines = e.stack.strip().split("\n")[:2]
        line += "\n  " + "\n  ".join(stack_lines)
      lines.append(line)
    return "\n".join(lines)

  def format_network(self, limit: int = 50, url_filter: str | None = None) -> str:
    """Format network requests for AI consumption.

    Args:
        limit: Max number of requests to return.
        url_filter: Only show requests whose URL contains this substring.
    """
    requests = list(self._requests)
    if url_filter:
      requests = [r for r in requests if url_filter in r.url]
    requests = requests[-limit:]
    if not requests:
      return "No network requests."
    lines: list[str] = []
    for r in requests:
      status = str(r.status) if r.status is not None else "..."
      fail = f" FAILED: {r.failure_text}" if r.failure_text else ""
      lines.append(f"[{r.id}] {r.method} {status} {r.url}{fail}")
    return "\n".join(lines)

  def clear(self) -> None:
    """Clear all captured state."""
    self._console.clear()
    self._errors.clear()
    self._requests.clear()
