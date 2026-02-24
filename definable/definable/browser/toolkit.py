"""BrowserToolkit — exposes Playwright browser as agent tools.

Follows the same AsyncLifecycleToolkit pattern as MCPToolkit:
- Call ``await toolkit.initialize()`` (or ``async with toolkit:``) before
  passing the toolkit to an agent.
- The toolkit manages the browser lifecycle automatically.

Tools exposed (all support refs or CSS selectors):
  Navigation  : browser_navigate, browser_go_back, browser_go_forward, browser_refresh
  Page state  : browser_get_url, browser_get_title, browser_get_text,
                browser_get_source, browser_get_attribute, browser_is_visible,
                browser_get_page_info
  Perception  : browser_snapshot, browser_screenshot
  Interaction : browser_click, browser_click_if_visible, browser_click_by_text,
                browser_type, browser_type_slowly, browser_press_keys, browser_press_key,
                browser_clear_input, browser_execute_js, browser_hover, browser_drag,
                browser_select_option, browser_fill_form, browser_set_value,
                browser_set_input_files
  Checkboxes  : browser_check, browser_uncheck, browser_is_checked
  Scrolling   : browser_scroll_down, browser_scroll_up, browser_scroll_to
  Waiting     : browser_wait, browser_wait_for_element, browser_wait_for_text, browser_wait_for
  Tabs        : browser_open_tab, browser_close_tab, browser_get_tabs, browser_switch_to_tab
  Cookies     : browser_get_cookies, browser_set_cookie, browser_clear_cookies
  Storage     : browser_get_storage, browser_set_storage
  Dialogs     : browser_handle_dialog
  DOM mutation : browser_highlight, browser_remove_elements
  Emulation   : browser_set_geolocation
  Diagnostics : browser_get_console, browser_get_errors, browser_get_network
  PDF         : browser_print_to_pdf
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from definable.agent.toolkit import Toolkit
from definable.browser.config import BrowserConfig
from definable.tool.function import Function
from definable.utils.log import log_debug, log_info

if TYPE_CHECKING:
  from definable.browser.events import BrowserActionEvent
  from definable.browser.playwright_browser import PlaywrightBrowser


def _make_tools(browser: "PlaywrightBrowser") -> list[Function]:
  """Build all browser tool Function objects as closures over ``browser``."""

  # -------------------------------------------------------------------------
  # Navigation (4)
  # -------------------------------------------------------------------------

  async def browser_navigate(url: str) -> str:
    """Navigate to a URL.
    REQUIRED: url (str) — must be a full URL including the scheme, e.g. 'https://example.com'.
    Returns the final URL and page title."""
    return await browser.navigate(url)

  async def browser_go_back() -> str:
    """Navigate to the previous page in browser history."""
    return await browser.go_back()

  async def browser_go_forward() -> str:
    """Navigate forward in browser history."""
    return await browser.go_forward()

  async def browser_refresh() -> str:
    """Reload the current page."""
    return await browser.refresh()

  # -------------------------------------------------------------------------
  # Page state (7)
  # -------------------------------------------------------------------------

  async def browser_get_url() -> str:
    """Return the current page URL."""
    return await browser.get_url()

  async def browser_get_title() -> str:
    """Return the current page title."""
    return await browser.get_title()

  async def browser_get_text(ref_or_selector: str = "body") -> str:
    """Return the visible text content of an element.
    Use a ref (e.g. "e1") from the last snapshot, or a CSS selector (e.g. "h1", "#main").
    Defaults to the entire page body."""
    return await browser.get_text(ref_or_selector)

  async def browser_get_source() -> str:
    """Return the full page HTML source (capped at 20 000 chars)."""
    return await browser.get_page_source()

  async def browser_get_attribute(ref_or_selector: str, attribute: str) -> str:
    """Return the value of an HTML attribute on an element.
    Example: browser_get_attribute("e1", "href") or browser_get_attribute("a.logo", "href")"""
    return await browser.get_attribute(ref_or_selector, attribute)

  async def browser_is_visible(ref_or_selector: str) -> str:
    """Check if an element is currently visible on the page.
    Returns "true" or "false"."""
    return await browser.is_element_visible(ref_or_selector)

  async def browser_get_page_info() -> str:
    """Return situational snapshot: URL, title, scroll position, viewport size.
    Call this to understand page context before deciding your next action."""
    return await browser.get_page_info()

  # -------------------------------------------------------------------------
  # Perception (2)
  # -------------------------------------------------------------------------

  async def browser_snapshot() -> str:
    """Return a role-based accessibility view of the page with element refs.
    Each interactive element gets a ref like [ref=e1] that you can use with
    other browser tools (e.g. browser_click("e1"), browser_type("e2", "hello")).
    This is the BEST way to understand page structure. Use it BEFORE interacting."""
    return await browser.snapshot()

  async def browser_screenshot(name: str = "screenshot") -> str:
    """Take a screenshot of the current page and save it to a file.
    Returns the file path. Use this to visually inspect the page."""
    return await browser.screenshot(name)

  # -------------------------------------------------------------------------
  # Interaction (15)
  # -------------------------------------------------------------------------

  async def browser_click(ref_or_selector: str) -> str:
    """Click an element using a ref (e.g. "e1") or CSS selector (e.g. "button#submit").
    Use browser_snapshot() first to get element refs."""
    return await browser.click(ref_or_selector)

  async def browser_click_if_visible(ref_or_selector: str) -> str:
    """Click an element only if it is currently visible. Safe to call on
    conditionally-shown elements like popups, cookie banners, etc."""
    return await browser.click_if_visible(ref_or_selector)

  async def browser_click_by_text(text: str, tag_name: str = "") -> str:
    """Click the first element whose visible text contains 'text'.
    More reliable than selectors on dynamic sites.
    Optionally restrict to a tag: browser_click_by_text("Sign in", "button")"""
    return await browser.click_by_text(text, tag_name)

  async def browser_type(ref_or_selector: str, text: str) -> str:
    """Clear an input field and type text into it.
    Use a ref (e.g. "e2") or CSS selector. Example: browser_type("e2", "hello@example.com")"""
    return await browser.type_text(ref_or_selector, text)

  async def browser_type_slowly(ref_or_selector: str, text: str) -> str:
    """Type text with human-like 75 ms delays between keystrokes.
    Use on sensitive form fields to avoid bot-detection triggers."""
    return await browser.type_slowly(ref_or_selector, text)

  async def browser_press_keys(ref_or_selector: str, keys: str) -> str:
    """Send keystrokes to a specific element (requires ref or CSS selector).
    Use "Enter" for Enter, "Tab" for Tab. Example: browser_press_keys("e1", "Enter")
    If you don't have a selector, use browser_press_key(key) instead."""
    return await browser.press_keys(ref_or_selector, keys)

  async def browser_press_key(key: str) -> str:
    """Press a keyboard key on the currently focused element (no selector needed).
    Use for Enter, Tab, Escape, Backspace, ArrowDown, ArrowUp, etc.
    Example: browser_press_key("Enter"), browser_press_key("Escape")"""
    return await browser.press_key(key)

  async def browser_clear_input(ref_or_selector: str) -> str:
    """Clear the contents of an input field or textarea."""
    return await browser.clear_input(ref_or_selector)

  async def browser_execute_js(code: str) -> str:
    """Execute JavaScript in the page context and return the result.
    Example: browser_execute_js("document.title")"""
    return await browser.execute_js(code)

  async def browser_hover(ref_or_selector: str) -> str:
    """Hover the mouse over an element.
    Reveals dropdown menus, tooltips, and hover-triggered content."""
    return await browser.hover(ref_or_selector)

  async def browser_drag(from_ref: str, to_ref: str) -> str:
    """Drag an element to another element using native drag-and-drop.
    Use for reordering lists, sliders, Kanban boards, and canvas apps."""
    return await browser.drag(from_ref, to_ref)

  async def browser_select_option(ref_or_selector: str, text: str) -> str:
    """Select an option from a <select> dropdown by its visible text.
    Example: browser_select_option("e3", "United States")"""
    return await browser.select_option(ref_or_selector, text)

  async def browser_fill_form(fields: list[dict[str, Any]]) -> str:
    """Fill multiple form fields at once. Each field: {ref, type, value}.
    Checkboxes use setChecked(), others use fill().
    Example: browser_fill_form([{"ref":"e1","type":"text","value":"John"},
    {"ref":"e2","type":"checkbox","value":true}])"""
    return await browser.fill_form(fields)

  async def browser_set_value(ref_or_selector: str, value: str) -> str:
    """Set an element's value directly — works for sliders and range inputs.
    Example: browser_set_value("e4", "75")"""
    return await browser.set_value(ref_or_selector, value)

  async def browser_set_input_files(ref_or_selector: str, paths: list[str]) -> str:
    """Set files on a file input element.
    Example: browser_set_input_files("e5", ["/path/to/file.pdf"])"""
    return await browser.set_input_files(ref_or_selector, paths)

  # -------------------------------------------------------------------------
  # Checkboxes (3)
  # -------------------------------------------------------------------------

  async def browser_is_checked(ref_or_selector: str) -> str:
    """Return 'true' or 'false' indicating whether a checkbox or radio is checked."""
    return await browser.is_checked(ref_or_selector)

  async def browser_check(ref_or_selector: str) -> str:
    """Check a checkbox or radio button if not already checked. Safe to call repeatedly."""
    return await browser.check_element(ref_or_selector)

  async def browser_uncheck(ref_or_selector: str) -> str:
    """Uncheck a checkbox if currently checked. Safe to call repeatedly."""
    return await browser.uncheck_element(ref_or_selector)

  # -------------------------------------------------------------------------
  # Scrolling (3)
  # -------------------------------------------------------------------------

  async def browser_scroll_down(amount: int = 3) -> str:
    """Scroll down by 'amount' screen-heights (default 3)."""
    return await browser.scroll_down(amount)

  async def browser_scroll_up(amount: int = 3) -> str:
    """Scroll up by 'amount' screen-heights (default 3)."""
    return await browser.scroll_up(amount)

  async def browser_scroll_to(ref_or_selector: str) -> str:
    """Scroll the page until the element is in view."""
    return await browser.scroll_to_element(ref_or_selector)

  # -------------------------------------------------------------------------
  # Waiting (4)
  # -------------------------------------------------------------------------

  async def browser_wait(seconds: float = 2.0) -> str:
    """Pause execution for the given number of seconds.
    PARAMETER: seconds (float) — number of seconds to wait, e.g. browser_wait(seconds=2.0).
    Do NOT use 'ms', 'timeout', or 'delay' — the only valid parameter name is 'seconds'."""
    return await browser.wait(seconds)

  async def browser_wait_for_element(ref_or_selector: str, timeout: float = 10.0) -> str:
    """Wait up to 'timeout' seconds for an element to appear on the page."""
    return await browser.wait_for_element(ref_or_selector, timeout)

  async def browser_wait_for_text(text: str, selector: str = "body", timeout: float = 10.0) -> str:
    """Wait up to 'timeout' seconds for text to appear inside a selector."""
    return await browser.wait_for_text(text, selector, timeout)

  async def browser_wait_for(
    text: str | None = None,
    text_gone: str | None = None,
    selector: str | None = None,
    url: str | None = None,
    load_state: str | None = None,
    fn: str | None = None,
    timeout: float | None = None,
  ) -> str:
    """Unified wait with multiple conditions. All parameters optional, checked sequentially.
    text: wait for text to appear. text_gone: wait for text to disappear.
    selector: wait for element. url: wait for URL match. load_state: "load"/"networkidle".
    fn: wait for JS function to return truthy. timeout: seconds (default 20)."""
    return await browser.wait_for(
      text=text,
      text_gone=text_gone,
      selector=selector,
      url=url,
      load_state=load_state,
      fn=fn,
      timeout=timeout,
    )

  # -------------------------------------------------------------------------
  # Tabs (4)
  # -------------------------------------------------------------------------

  async def browser_open_tab(url: str = "") -> str:
    """Open a new browser tab, optionally navigating to 'url'."""
    return await browser.open_tab(url)

  async def browser_close_tab() -> str:
    """Close the currently active browser tab."""
    return await browser.close_tab()

  async def browser_get_tabs() -> str:
    """Return the list of open browser tabs with URLs.
    Use before browser_switch_to_tab to know valid indices (0 to N-1)."""
    return await browser.get_tabs()

  async def browser_switch_to_tab(index: int) -> str:
    """Switch focus to the tab at zero-based index.
    Example: browser_switch_to_tab(0) goes to the first tab."""
    return await browser.switch_to_tab(index)

  # -------------------------------------------------------------------------
  # Cookies (3)
  # -------------------------------------------------------------------------

  async def browser_get_cookies() -> str:
    """Return all cookies for the current page as a JSON array."""
    return await browser.get_cookies()

  async def browser_set_cookie(name: str, value: str) -> str:
    """Set a cookie (name + value) on the current domain."""
    return await browser.set_cookie(name, value)

  async def browser_clear_cookies() -> str:
    """Delete all cookies for the current session."""
    return await browser.clear_cookies()

  # -------------------------------------------------------------------------
  # Dialogs (1)
  # -------------------------------------------------------------------------

  async def browser_handle_dialog(accept: bool = True, prompt_text: str = "") -> str:
    """Accept or dismiss a browser dialog (alert / confirm / prompt).
    Set accept=false to dismiss. Use prompt_text to fill a prompt() dialog."""
    return await browser.handle_dialog(accept, prompt_text)

  # -------------------------------------------------------------------------
  # Storage (2)
  # -------------------------------------------------------------------------

  async def browser_get_storage(key: str, storage_type: str = "local") -> str:
    """Get a value from localStorage or sessionStorage.
    storage_type: 'local' (default) or 'session'."""
    return await browser.get_storage(key, storage_type)

  async def browser_set_storage(key: str, value: str, storage_type: str = "local") -> str:
    """Set a key/value pair in localStorage or sessionStorage.
    storage_type: 'local' (default) or 'session'."""
    return await browser.set_storage(key, value, storage_type)

  # -------------------------------------------------------------------------
  # DOM mutation (2)
  # -------------------------------------------------------------------------

  async def browser_highlight(ref_or_selector: str) -> str:
    """Flash a visual highlight on an element for debugging.
    Use to visually confirm the correct element before acting on it."""
    return await browser.highlight(ref_or_selector)

  async def browser_remove_elements(selector: str) -> str:
    """Remove ALL elements matching selector from the DOM.
    Use to dismiss cookie banners, overlays, and popups before interacting.
    Example: browser_remove_elements(".cookie-notice")"""
    return await browser.remove_elements(selector)

  # -------------------------------------------------------------------------
  # Emulation (1)
  # -------------------------------------------------------------------------

  async def browser_set_geolocation(latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    """Override the browser's GPS coordinates.
    Example: browser_set_geolocation(37.7749, -122.4194) sets San Francisco."""
    return await browser.set_geolocation(latitude, longitude, accuracy)

  # -------------------------------------------------------------------------
  # Diagnostics (3)
  # -------------------------------------------------------------------------

  async def browser_get_console(limit: int = 50, level: str | None = None) -> str:
    """Return captured browser console messages.
    Optionally filter by minimum level: 'error', 'warning', 'info', 'log', 'debug'."""
    return await browser.get_console(limit=limit, level=level)

  async def browser_get_errors(limit: int = 20) -> str:
    """Return captured browser page errors (JS exceptions, unhandled rejections).
    Useful for debugging when page behavior is unexpected."""
    return await browser.get_errors(limit=limit)

  async def browser_get_network(limit: int = 50, url_filter: str | None = None) -> str:
    """Return captured network requests with status codes.
    Filter by URL substring: browser_get_network(url_filter="/api/")"""
    return await browser.get_network(limit=limit, url_filter=url_filter)

  # -------------------------------------------------------------------------
  # PDF (1)
  # -------------------------------------------------------------------------

  async def browser_print_to_pdf(name: str = "page") -> str:
    """Save the current page as a PDF and return the file path."""
    return await browser.print_to_pdf(name)

  # -------------------------------------------------------------------------
  # Build Function list
  # -------------------------------------------------------------------------
  fns = [
    # Navigation (4)
    browser_navigate,
    browser_go_back,
    browser_go_forward,
    browser_refresh,
    # Page state (7)
    browser_get_url,
    browser_get_title,
    browser_get_text,
    browser_get_source,
    browser_get_attribute,
    browser_is_visible,
    browser_get_page_info,
    # Perception (2)
    browser_snapshot,
    browser_screenshot,
    # Interaction (15)
    browser_click,
    browser_click_if_visible,
    browser_click_by_text,
    browser_type,
    browser_type_slowly,
    browser_press_keys,
    browser_press_key,
    browser_clear_input,
    browser_execute_js,
    browser_hover,
    browser_drag,
    browser_select_option,
    browser_fill_form,
    browser_set_value,
    browser_set_input_files,
    # Checkboxes (3)
    browser_is_checked,
    browser_check,
    browser_uncheck,
    # Scrolling (3)
    browser_scroll_down,
    browser_scroll_up,
    browser_scroll_to,
    # Waiting (4)
    browser_wait,
    browser_wait_for_element,
    browser_wait_for_text,
    browser_wait_for,
    # Tabs (4)
    browser_open_tab,
    browser_close_tab,
    browser_get_tabs,
    browser_switch_to_tab,
    # Cookies (3)
    browser_get_cookies,
    browser_set_cookie,
    browser_clear_cookies,
    # Dialogs (1)
    browser_handle_dialog,
    # Storage (2)
    browser_get_storage,
    browser_set_storage,
    # DOM mutation (2)
    browser_highlight,
    browser_remove_elements,
    # Emulation (1)
    browser_set_geolocation,
    # Diagnostics (3)
    browser_get_console,
    browser_get_errors,
    browser_get_network,
    # PDF (1)
    browser_print_to_pdf,
  ]
  tools = [Function(name=fn.__name__, entrypoint=fn) for fn in fns]  # type: ignore[arg-type]
  for t in tools:
    t.process_entrypoint()
  return tools


class BrowserToolkit(Toolkit):
  """Agent toolkit that provides full browser automation via Playwright CDP.

  Playwright connects to Chrome directly via Chrome DevTools Protocol — native
  async, role-based element refs, AI-friendly errors, and self-healing connections.

  Usage::

      from definable.browser import BrowserConfig, BrowserToolkit
      from definable.agent import Agent

      config = BrowserConfig(headless=False)
      async with BrowserToolkit(config=config) as toolkit:
          agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
          result = await agent.arun("Go to example.com and tell me the title")
          print(result.content)

  To attach to YOUR running Chrome::

      # 1. Launch Chrome with: --remote-debugging-port=9222 --no-first-run
      # 2. Use:
      config = BrowserConfig(cdp_url="ws://127.0.0.1:9222")
  """

  def __init__(
    self,
    config: Optional[BrowserConfig] = None,
    browser: Optional["PlaywrightBrowser"] = None,
    on_action: Optional[Callable[["BrowserActionEvent"], Any]] = None,
  ) -> None:
    """
    Args:
        config: Browser configuration. Defaults to ``BrowserConfig()``.
        browser: Inject a pre-built browser instance (useful for testing).
                 When provided, the toolkit does NOT call ``start()``/``stop()``.
        on_action: Optional callback invoked on every browser action (click, type,
                   navigate, etc.). Receives a ``BrowserActionEvent``. Can be sync
                   or async. Use to pipe browser events into EventStream, logging, or UI.
    """
    super().__init__()
    self._config = config or BrowserConfig()
    self._browser = browser
    self._on_action = on_action
    self._owned = browser is None
    self._initialized = False
    self._tools: list[Function] = []

  @property
  def tools(self) -> list[Function]:
    return self._tools

  async def initialize(self) -> None:
    if self._initialized:
      return
    if self._owned:
      if self._browser is None:
        from definable.browser.playwright_browser import PlaywrightBrowser

        self._browser = PlaywrightBrowser(self._config)
      await self._browser.start()

    assert self._browser is not None
    if self._on_action:
      self._browser.on_action = self._on_action
    self._tools = _make_tools(self._browser)
    self._initialized = True
    log_info(f"BrowserToolkit: initialized ({len(self._tools)} tools)")

  async def shutdown(self) -> None:
    self._tools = []
    if self._owned and self._browser is not None:
      await self._browser.stop()
      self._browser = None
    self._initialized = False
    log_debug("BrowserToolkit: shutdown complete")

  async def __aenter__(self) -> "BrowserToolkit":
    await self.initialize()
    return self

  async def __aexit__(self, *_: object) -> None:
    await self.shutdown()

  def __repr__(self) -> str:
    state = "ready" if self._initialized else "not initialized"
    return f"BrowserToolkit({state}, tools={len(self._tools)})"
