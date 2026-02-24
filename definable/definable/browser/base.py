"""BaseBrowser — abstract interface for Playwright-based browser implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBrowser(ABC):
  """Abstract base class for browser implementations.

  All methods are async. All interaction methods accept a ``ref_or_selector``
  parameter that can be either a role-based ref (e.g. ``"e1"``) from the
  last ``snapshot()`` call, or a CSS selector (e.g. ``"button.submit"``).

  All methods return a plain ``str`` result that the agent can read.
  Error conditions are returned as ``"Error: <message>"`` strings rather
  than raising exceptions, so the agent can decide how to recover.
  """

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def start(self) -> None:
    """Launch (or attach to) the browser. Must be called before any other method."""

  @abstractmethod
  async def stop(self) -> None:
    """Close the browser and release all resources."""

  async def __aenter__(self) -> "BaseBrowser":
    await self.start()
    return self

  async def __aexit__(self, *_: object) -> None:
    await self.stop()

  # ---------------------------------------------------------------------------
  # Perception
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def snapshot(
    self,
    options: Any = None,
    selector: str | None = None,
    frame_selector: str | None = None,
  ) -> str:
    """Get role-based accessibility snapshot with element refs (e1, e2, ...)."""

  @abstractmethod
  async def screenshot(
    self,
    name: str = "screenshot",
    ref: str | None = None,
    full_page: bool = False,
  ) -> str:
    """Save a screenshot and return its file path."""

  @abstractmethod
  async def get_page_info(self) -> str:
    """Return situational snapshot: URL, title, scroll position, viewport."""

  # ---------------------------------------------------------------------------
  # Navigation
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def navigate(self, url: str) -> str:
    """Navigate to ``url``. Returns current URL and page title on success."""

  @abstractmethod
  async def go_back(self) -> str:
    """Navigate to the previous page in history."""

  @abstractmethod
  async def go_forward(self) -> str:
    """Navigate forward in history."""

  @abstractmethod
  async def refresh(self) -> str:
    """Reload the current page."""

  # ---------------------------------------------------------------------------
  # Page state
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def get_url(self) -> str:
    """Return the current page URL."""

  @abstractmethod
  async def get_title(self) -> str:
    """Return the current page title."""

  @abstractmethod
  async def get_page_source(self, max_chars: int = 20000) -> str:
    """Return the full page HTML source (capped at ``max_chars``)."""

  @abstractmethod
  async def get_text(self, ref_or_selector: str = "body") -> str:
    """Return the visible text content of the element."""

  @abstractmethod
  async def get_attribute(self, ref_or_selector: str, attribute: str) -> str:
    """Return the value of ``attribute`` on the element."""

  @abstractmethod
  async def is_element_visible(self, ref_or_selector: str) -> str:
    """Return ``"true"`` or ``"false"`` depending on element visibility."""

  # ---------------------------------------------------------------------------
  # Interaction
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def click(self, ref_or_selector: str) -> str:
    """Click the element."""

  @abstractmethod
  async def click_if_visible(self, ref_or_selector: str) -> str:
    """Click the element only if it is currently visible."""

  @abstractmethod
  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    """Click the first element whose visible text contains ``text``."""

  @abstractmethod
  async def hover(self, ref_or_selector: str) -> str:
    """Hover the mouse over the element."""

  @abstractmethod
  async def drag(self, from_ref: str, to_ref: str) -> str:
    """Drag from one element to another."""

  @abstractmethod
  async def type_text(self, ref_or_selector: str, text: str, submit: bool = False) -> str:
    """Clear the element and type ``text`` into it."""

  @abstractmethod
  async def type_slowly(self, ref_or_selector: str, text: str, delay: float = 75.0) -> str:
    """Type ``text`` char-by-char with human-like keystroke delays."""

  @abstractmethod
  async def press_key(self, key: str) -> str:
    """Press a keyboard key on the currently focused element."""

  @abstractmethod
  async def press_keys(self, ref_or_selector: str, keys: str) -> str:
    """Send ``keys`` to the element."""

  @abstractmethod
  async def clear_input(self, ref_or_selector: str) -> str:
    """Clear the text from an input or textarea element."""

  @abstractmethod
  async def select_option(self, ref_or_selector: str, text: str) -> str:
    """Select an option from a ``<select>`` element by its visible text."""

  @abstractmethod
  async def check_element(self, ref_or_selector: str) -> str:
    """Check a checkbox or radio button."""

  @abstractmethod
  async def uncheck_element(self, ref_or_selector: str) -> str:
    """Uncheck a checkbox."""

  @abstractmethod
  async def is_checked(self, ref_or_selector: str) -> str:
    """Return ``"true"`` or ``"false"`` for checkbox/radio state."""

  @abstractmethod
  async def set_value(self, ref_or_selector: str, value: str) -> str:
    """Set the value of an element directly (works for sliders, range inputs)."""

  @abstractmethod
  async def set_input_files(self, ref_or_selector: str, paths: list[str]) -> str:
    """Set files on a file input element."""

  @abstractmethod
  async def fill_form(self, fields: list[dict[str, Any]]) -> str:
    """Batch form fill. Each field: {ref, type, value}."""

  @abstractmethod
  async def execute_js(self, code: str, ref: str | None = None, timeout: float | None = None) -> str:
    """Execute ``code`` as JavaScript in the page context."""

  @abstractmethod
  async def highlight(self, ref_or_selector: str) -> str:
    """Briefly highlight the element for visual debugging."""

  @abstractmethod
  async def remove_elements(self, selector: str) -> str:
    """Remove ALL elements matching ``selector`` from the DOM."""

  # ---------------------------------------------------------------------------
  # Scrolling
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def scroll_down(self, amount: int = 3) -> str:
    """Scroll down by ``amount`` screen-heights."""

  @abstractmethod
  async def scroll_up(self, amount: int = 3) -> str:
    """Scroll up by ``amount`` screen-heights."""

  @abstractmethod
  async def scroll_to_element(self, ref_or_selector: str) -> str:
    """Scroll until the element is in the viewport."""

  # ---------------------------------------------------------------------------
  # Waiting
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def wait(self, seconds: float = 2.0) -> str:
    """Pause for ``seconds`` seconds."""

  @abstractmethod
  async def wait_for_element(self, ref_or_selector: str, timeout: float = 10.0) -> str:
    """Wait up to ``timeout`` seconds for the element to appear."""

  @abstractmethod
  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    """Wait up to ``timeout`` seconds for ``text`` to appear."""

  @abstractmethod
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
    """Unified wait with multiple condition types."""

  # ---------------------------------------------------------------------------
  # Tabs
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def open_tab(self, url: str = "") -> str:
    """Open a new tab, optionally navigating to ``url``."""

  @abstractmethod
  async def close_tab(self) -> str:
    """Close the current tab."""

  @abstractmethod
  async def get_tabs(self) -> str:
    """Return the list of open browser tabs."""

  @abstractmethod
  async def switch_to_tab(self, index: int) -> str:
    """Switch focus to the tab at ``index`` (0-based)."""

  # ---------------------------------------------------------------------------
  # Cookies
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def get_cookies(self) -> str:
    """Return all cookies for the current page as a JSON array."""

  @abstractmethod
  async def set_cookie(self, name: str, value: str) -> str:
    """Set a cookie on the current domain."""

  @abstractmethod
  async def clear_cookies(self) -> str:
    """Delete all cookies for the current session."""

  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def get_storage(self, key: str | None = None, kind: str = "local") -> str:
    """Get a value from ``localStorage`` or ``sessionStorage``."""

  @abstractmethod
  async def set_storage(self, key: str, value: str, kind: str = "local") -> str:
    """Set a key/value pair in ``localStorage`` or ``sessionStorage``."""

  # ---------------------------------------------------------------------------
  # Dialogs
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    """Accept or dismiss a browser dialog (alert/confirm/prompt)."""

  # ---------------------------------------------------------------------------
  # Emulation
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    """Override the browser's GPS coordinates."""

  # ---------------------------------------------------------------------------
  # PDF
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def print_to_pdf(self, name: str = "page") -> str:
    """Save the current page as a PDF and return the file path."""

  # ---------------------------------------------------------------------------
  # Page state diagnostics
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def get_console(self, limit: int = 50, level: str | None = None) -> str:
    """Return captured browser console messages."""

  @abstractmethod
  async def get_errors(self, limit: int = 20) -> str:
    """Return captured browser page errors."""

  @abstractmethod
  async def get_network(self, limit: int = 50, url_filter: str | None = None) -> str:
    """Return captured network requests."""
