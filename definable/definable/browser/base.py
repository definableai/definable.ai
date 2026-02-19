"""BaseBrowser — abstract interface for SeleniumBase CDP browser implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBrowser(ABC):
  """Abstract base class for CDP-backed browser implementations.

  All methods are async — concrete implementations wrap SeleniumBase's
  synchronous API with ``asyncio.to_thread`` or a dedicated executor so the
  agent event loop is never blocked.

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
  async def get_page_source(self) -> str:
    """Return the full page HTML source (capped at 20 000 chars)."""

  @abstractmethod
  async def get_text(self, selector: str = "body") -> str:
    """Return the visible text content of the element at ``selector``."""

  @abstractmethod
  async def get_attribute(self, selector: str, attribute: str) -> str:
    """Return the value of ``attribute`` on the element at ``selector``."""

  @abstractmethod
  async def is_element_visible(self, selector: str) -> str:
    """Return ``"true"`` or ``"false"`` depending on element visibility."""

  # ---------------------------------------------------------------------------
  # Interaction
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def click(self, selector: str) -> str:
    """Click the element at ``selector``."""

  @abstractmethod
  async def click_if_visible(self, selector: str) -> str:
    """Click the element only if it is currently visible."""

  @abstractmethod
  async def type_text(self, selector: str, text: str) -> str:
    """Clear ``selector`` and type ``text`` into it."""

  @abstractmethod
  async def press_keys(self, selector: str, keys: str) -> str:
    """Send ``keys`` to the element (e.g. ``"\\n"`` to press Enter)."""

  @abstractmethod
  async def clear_input(self, selector: str) -> str:
    """Clear the text from an input or textarea element."""

  @abstractmethod
  async def execute_js(self, code: str) -> str:
    """Execute ``code`` as JavaScript in the page context and return the result."""

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
  async def scroll_to_element(self, selector: str) -> str:
    """Scroll until ``selector`` is in the viewport."""

  # ---------------------------------------------------------------------------
  # Waiting
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def wait(self, seconds: float = 2.0) -> str:
    """Pause for ``seconds`` seconds."""

  @abstractmethod
  async def wait_for_element(self, selector: str, timeout: float = 10.0) -> str:
    """Wait up to ``timeout`` seconds for ``selector`` to appear in the DOM."""

  @abstractmethod
  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    """Wait up to ``timeout`` seconds for ``text`` to appear inside ``selector``."""

  # ---------------------------------------------------------------------------
  # Screenshot
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def screenshot(self, name: str = "screenshot") -> str:
    """Save a screenshot and return its file path."""

  # ---------------------------------------------------------------------------
  # Tabs
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def open_tab(self, url: str = "") -> str:
    """Open a new tab, optionally navigating to ``url``."""

  @abstractmethod
  async def close_tab(self) -> str:
    """Close the current tab."""

  # ---------------------------------------------------------------------------
  # Advanced interaction
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def hover(self, selector: str) -> str:
    """Hover the mouse over ``selector`` (reveals dropdown menus, tooltips)."""

  @abstractmethod
  async def drag(self, from_selector: str, to_selector: str) -> str:
    """Dispatch HTML5 drag-and-drop events from ``from_selector`` to ``to_selector``."""

  @abstractmethod
  async def type_slowly(self, selector: str, text: str) -> str:
    """Type ``text`` char-by-char with human-like 75 ms keystroke delays."""

  @abstractmethod
  async def select_option(self, selector: str, text: str) -> str:
    """Select an option from a ``<select>`` element by its visible text."""

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
  # Dialogs
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    """Accept or dismiss a browser dialog (alert/confirm/prompt).
    Pass ``prompt_text`` to fill a ``prompt()`` dialog before accepting."""

  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def get_storage(self, key: str, storage_type: str = "local") -> str:
    """Get a value from ``localStorage`` or ``sessionStorage``.
    ``storage_type``: ``"local"`` (default) or ``"session"``."""

  @abstractmethod
  async def set_storage(self, key: str, value: str, storage_type: str = "local") -> str:
    """Set a key/value pair in ``localStorage`` or ``sessionStorage``."""

  # ---------------------------------------------------------------------------
  # Browser state
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    """Override the browser's GPS coordinates via CDP Emulation."""

  @abstractmethod
  async def highlight(self, selector: str) -> str:
    """Briefly highlight ``selector`` with a gold border (2 s) for visual debugging."""

  @abstractmethod
  async def get_page_info(self) -> str:
    """Return situational snapshot: URL, title, scroll position, interactive element counts."""

  @abstractmethod
  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    """Click the first element whose visible text contains ``text``.
    Optionally filter to ``tag_name`` (e.g. ``"button"``, ``"a"``)."""

  @abstractmethod
  async def remove_elements(self, selector: str) -> str:
    """Remove ALL elements matching ``selector`` from the DOM.
    Use to dismiss cookie banners, ads, and overlay popups."""

  @abstractmethod
  async def is_checked(self, selector: str) -> str:
    """Return ``"true"`` or ``"false"`` depending on whether a checkbox/radio is checked."""

  @abstractmethod
  async def check_element(self, selector: str) -> str:
    """Check a checkbox or radio button if it is not already checked."""

  @abstractmethod
  async def uncheck_element(self, selector: str) -> str:
    """Uncheck a checkbox if it is currently checked."""

  @abstractmethod
  async def set_value(self, selector: str, value: str) -> str:
    """Set the value of an element directly (works for sliders and range inputs)."""

  @abstractmethod
  async def get_tabs(self) -> str:
    """Return the number of open browser tabs."""

  @abstractmethod
  async def switch_to_tab(self, index: int) -> str:
    """Switch focus to the tab at ``index`` (0-based)."""

  @abstractmethod
  async def print_to_pdf(self, name: str = "page") -> str:
    """Save the current page as a PDF and return the file path."""

  # ---------------------------------------------------------------------------
  # CAPTCHA
  # ---------------------------------------------------------------------------

  @abstractmethod
  async def solve_captcha(self) -> str:
    """Attempt to solve any CAPTCHA on the current page."""
