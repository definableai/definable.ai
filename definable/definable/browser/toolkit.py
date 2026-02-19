"""BrowserToolkit — exposes SeleniumBase CDP browser as agent tools.

Follows the same AsyncLifecycleToolkit pattern as MCPToolkit:
- Call ``await toolkit.initialize()`` (or ``async with toolkit:``) before
  passing the toolkit to an agent.
- The toolkit manages the browser lifecycle automatically.

Tools exposed (all use CSS selectors):
  Navigation  : browser_navigate, browser_go_back, browser_go_forward, browser_refresh
  Page state  : browser_get_url, browser_get_title, browser_get_text,
                browser_get_source, browser_get_attribute, browser_is_visible,
                browser_get_page_info, browser_snapshot
  Interaction : browser_click, browser_click_if_visible, browser_type,
                browser_press_keys, browser_press_key, browser_clear_input,
                browser_execute_js
  Scrolling   : browser_scroll_down, browser_scroll_up, browser_scroll_to
  Waiting     : browser_wait, browser_wait_for_element, browser_wait_for_text
  Screenshot  : browser_screenshot
  Tabs        : browser_open_tab, browser_close_tab
  CAPTCHA     : browser_solve_captcha
"""

from __future__ import annotations

from typing import Any, Optional

from definable.agent.toolkit import Toolkit
from definable.browser.config import BrowserConfig
from definable.tool.function import Function
from definable.utils.log import log_debug, log_info


def _make_tools(browser: Any) -> list[Function]:
  """Build all browser tool Function objects as closures over ``browser``."""

  # -------------------------------------------------------------------------
  # Navigation
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
  # Page state
  # -------------------------------------------------------------------------

  async def browser_get_url() -> str:
    """Return the current page URL."""
    return await browser.get_url()

  async def browser_get_title() -> str:
    """Return the current page title."""
    return await browser.get_title()

  async def browser_get_text(selector: str = "body") -> str:
    """Return the visible text content of an element.
    Use a CSS selector (e.g. "h1", "#main", "body").
    Defaults to the entire page body."""
    return await browser.get_text(selector)

  async def browser_get_source() -> str:
    """Return the full page HTML source (capped at 20 000 chars)."""
    return await browser.get_page_source()

  async def browser_get_attribute(selector: str, attribute: str) -> str:
    """Return the value of an HTML attribute on an element.
    Example: browser_get_attribute("a.logo", "href")"""
    return await browser.get_attribute(selector, attribute)

  async def browser_is_visible(selector: str) -> str:
    """Check if an element is currently visible on the page.
    Returns "true" or "false"."""
    return await browser.is_element_visible(selector)

  # -------------------------------------------------------------------------
  # Interaction
  # -------------------------------------------------------------------------

  async def browser_click(selector: str) -> str:
    """Click an element using a CSS selector.
    CSS selectors only. Do NOT use :has-text(), :contains(), or :text() — these
    are Playwright-specific and will fail. Use browser_click_by_text() instead.
    Example: browser_click("button#submit"), browser_click('a[href="/login"]')"""
    return await browser.click(selector)

  async def browser_click_if_visible(selector: str) -> str:
    """Click an element only if it is currently visible. Safe to call on
    conditionally-shown elements like popups, cookie banners, etc."""
    return await browser.click_if_visible(selector)

  async def browser_type(selector: str, text: str) -> str:
    """Clear an input field and type text into it.
    Example: browser_type("#search", "OpenAI")"""
    return await browser.type_text(selector, text)

  async def browser_press_keys(selector: str, keys: str) -> str:
    """Send keystrokes to a specific element (requires CSS selector).
    Use "\\n" for Enter, "\\t" for Tab, or any keyboard key name.
    Example: browser_press_keys("#search", "\\n")
    If you don't have a selector, use browser_press_key(key) instead."""
    return await browser.press_keys(selector, keys)

  async def browser_press_key(key: str) -> str:
    """Press a keyboard key on the currently focused element (no selector needed).
    Use for Enter, Tab, Escape, Backspace, ArrowDown, ArrowUp, etc.
    Example: browser_press_key("Enter"), browser_press_key("Escape")"""
    return await browser.press_key(key)

  async def browser_clear_input(selector: str) -> str:
    """Clear the contents of an input field or textarea."""
    return await browser.clear_input(selector)

  async def browser_execute_js(code: str) -> str:
    """Execute JavaScript in the page context and return the result.
    Example: browser_execute_js("return document.title")"""
    return await browser.execute_js(code)

  # -------------------------------------------------------------------------
  # Scrolling
  # -------------------------------------------------------------------------

  async def browser_scroll_down(amount: int = 3) -> str:
    """Scroll down by 'amount' screen-heights (default 3)."""
    return await browser.scroll_down(amount)

  async def browser_scroll_up(amount: int = 3) -> str:
    """Scroll up by 'amount' screen-heights (default 3)."""
    return await browser.scroll_up(amount)

  async def browser_scroll_to(selector: str) -> str:
    """Scroll the page until the element at 'selector' is in view."""
    return await browser.scroll_to_element(selector)

  # -------------------------------------------------------------------------
  # Waiting
  # -------------------------------------------------------------------------

  async def browser_wait(seconds: float = 2.0) -> str:
    """Pause execution for the given number of seconds. Use after actions that trigger page loads.
    PARAMETER: seconds (float) — number of seconds to wait, e.g. browser_wait(seconds=2.0).
    Do NOT use 'ms', 'timeout', or 'delay' — the only valid parameter name is 'seconds'."""
    return await browser.wait(seconds)

  async def browser_wait_for_element(selector: str, timeout: float = 10.0) -> str:
    """Wait up to 'timeout' seconds for an element to appear on the page."""
    return await browser.wait_for_element(selector, timeout)

  async def browser_wait_for_text(text: str, selector: str = "body", timeout: float = 10.0) -> str:
    """Wait up to 'timeout' seconds for text to appear inside a selector."""
    return await browser.wait_for_text(text, selector, timeout)

  # -------------------------------------------------------------------------
  # Screenshot
  # -------------------------------------------------------------------------

  async def browser_screenshot(name: str = "screenshot") -> str:
    """Take a screenshot of the current page and save it to a file.
    Returns the file path. Use this to visually inspect the page."""
    return await browser.screenshot(name)

  # -------------------------------------------------------------------------
  # Tabs
  # -------------------------------------------------------------------------

  async def browser_open_tab(url: str = "") -> str:
    """Open a new browser tab, optionally navigating to 'url'."""
    return await browser.open_tab(url)

  async def browser_close_tab() -> str:
    """Close the currently active browser tab."""
    return await browser.close_tab()

  # -------------------------------------------------------------------------
  # Advanced interaction
  # -------------------------------------------------------------------------

  async def browser_hover(selector: str) -> str:
    """Hover the mouse over an element (use CSS selector).
    Reveals dropdown menus, tooltips, and hover-triggered content."""
    return await browser.hover(selector)

  async def browser_drag(from_selector: str, to_selector: str) -> str:
    """Drag an element to another element using HTML5 drag-and-drop events.
    Use for reordering lists, sliders, Kanban boards, and canvas apps."""
    return await browser.drag(from_selector, to_selector)

  async def browser_type_slowly(selector: str, text: str) -> str:
    """Type text with human-like 75 ms delays between keystrokes.
    Use on sensitive form fields to avoid bot-detection triggers."""
    return await browser.type_slowly(selector, text)

  async def browser_select_option(selector: str, text: str) -> str:
    """Select an option from a <select> dropdown by its visible text.
    Example: browser_select_option(\"#country\", \"United States\")"""
    return await browser.select_option(selector, text)

  # -------------------------------------------------------------------------
  # Cookies
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
  # Dialogs
  # -------------------------------------------------------------------------

  async def browser_handle_dialog(accept: bool = True, prompt_text: str = "") -> str:
    """Accept or dismiss a browser dialog (alert / confirm / prompt).
    Set accept=false to dismiss. Use prompt_text to fill a prompt() dialog."""
    return await browser.handle_dialog(accept, prompt_text)

  # -------------------------------------------------------------------------
  # Storage
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
  # Browser state
  # -------------------------------------------------------------------------

  async def browser_set_geolocation(latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    """Override the browser's GPS coordinates via Chrome DevTools Protocol.
    Example: browser_set_geolocation(37.7749, -122.4194) sets San Francisco."""
    return await browser.set_geolocation(latitude, longitude, accuracy)

  async def browser_highlight(selector: str) -> str:
    """Flash a gold border around an element for 2 seconds.
    Use to visually confirm the correct element before acting on it."""
    return await browser.highlight(selector)

  async def browser_get_page_info() -> str:
    """Return situational snapshot: URL, title, scroll position, interactive element counts.
    Call this to understand page context before deciding your next action."""
    return await browser.get_page_info()

  async def browser_snapshot() -> str:
    """Return an accessibility-tree-style view of the page showing all interactive
    and landmark elements with labels, roles, and CSS selectors.
    Use this BEFORE interacting to discover element selectors, form fields, and
    navigation links. Much more useful than browser_get_source for understanding
    page structure. Each element has a [N] index, tag, label, and a sel="..." hint."""
    return await browser.get_page_snapshot()

  # -------------------------------------------------------------------------
  # Text-based interaction & DOM mutation
  # -------------------------------------------------------------------------

  async def browser_click_by_text(text: str, tag_name: str = "") -> str:
    """Click the first element whose visible text contains 'text'.
    More reliable than CSS selectors on dynamic sites.
    Optionally restrict to a tag: browser_click_by_text(\"Sign in\", \"button\")"""
    return await browser.click_by_text(text, tag_name)

  async def browser_remove_elements(selector: str) -> str:
    """Remove ALL elements matching selector from the DOM.
    Use to dismiss cookie banners, overlays, and popups before interacting.
    Example: browser_remove_elements(\".cookie-notice\")"""
    return await browser.remove_elements(selector)

  # -------------------------------------------------------------------------
  # Checkboxes
  # -------------------------------------------------------------------------

  async def browser_is_checked(selector: str) -> str:
    """Return 'true' or 'false' indicating whether a checkbox or radio is checked."""
    return await browser.is_checked(selector)

  async def browser_check(selector: str) -> str:
    """Check a checkbox or radio button if not already checked. Safe to call repeatedly."""
    return await browser.check_element(selector)

  async def browser_uncheck(selector: str) -> str:
    """Uncheck a checkbox if currently checked. Safe to call repeatedly."""
    return await browser.uncheck_element(selector)

  # -------------------------------------------------------------------------
  # Value & tab control
  # -------------------------------------------------------------------------

  async def browser_set_value(selector: str, value: str) -> str:
    """Set an element's value directly — works for sliders and range inputs.
    Example: browser_set_value(\"#volume\", \"75\")"""
    return await browser.set_value(selector, value)

  async def browser_get_tabs() -> str:
    """Return the number of open browser tabs.
    Use before browser_switch_to_tab to know valid indices (0 to N-1)."""
    return await browser.get_tabs()

  async def browser_switch_to_tab(index: int) -> str:
    """Switch focus to the tab at zero-based index.
    Example: browser_switch_to_tab(0) goes to the first tab."""
    return await browser.switch_to_tab(index)

  async def browser_print_to_pdf(name: str = "page") -> str:
    """Save the current page as a PDF and return the file path."""
    return await browser.print_to_pdf(name)

  # -------------------------------------------------------------------------
  # CAPTCHA
  # -------------------------------------------------------------------------

  async def browser_solve_captcha() -> str:
    """Attempt to automatically solve a CAPTCHA on the current page.
    Handles Cloudflare Turnstile, reCAPTCHA, hCaptcha, and others."""
    return await browser.solve_captcha()

  # -------------------------------------------------------------------------
  # Build Function list — closures capture `browser` directly, no deps magic
  # -------------------------------------------------------------------------
  fns = [
    browser_navigate,
    browser_go_back,
    browser_go_forward,
    browser_refresh,
    browser_get_url,
    browser_get_title,
    browser_get_text,
    browser_get_source,
    browser_get_attribute,
    browser_is_visible,
    browser_click,
    browser_click_if_visible,
    browser_type,
    browser_press_keys,
    browser_press_key,
    browser_clear_input,
    browser_execute_js,
    browser_scroll_down,
    browser_scroll_up,
    browser_scroll_to,
    browser_wait,
    browser_wait_for_element,
    browser_wait_for_text,
    browser_screenshot,
    browser_open_tab,
    browser_close_tab,
    browser_hover,
    browser_drag,
    browser_type_slowly,
    browser_select_option,
    browser_get_cookies,
    browser_set_cookie,
    browser_clear_cookies,
    browser_handle_dialog,
    browser_get_storage,
    browser_set_storage,
    browser_set_geolocation,
    browser_highlight,
    browser_get_page_info,
    browser_snapshot,
    browser_click_by_text,
    browser_remove_elements,
    browser_is_checked,
    browser_check,
    browser_uncheck,
    browser_set_value,
    browser_get_tabs,
    browser_switch_to_tab,
    browser_print_to_pdf,
    browser_solve_captcha,
  ]
  return [Function(name=fn.__name__, entrypoint=fn) for fn in fns]  # type: ignore[arg-type]


class BrowserToolkit(Toolkit):
  """Agent toolkit that provides full browser automation via SeleniumBase CDP.

  SeleniumBase CDP mode connects Chrome directly via Chrome DevTools Protocol
  — no Selenium WebDriver, no automation fingerprints. Websites cannot detect
  it with standard bot-detection checks.

  Usage::

      from definable.browser import BrowserConfig, BrowserToolkit
      from definable.agent import Agent
      from definable.model.openai import OpenAIChat

      config = BrowserConfig(headless=False)
      async with BrowserToolkit(config=config) as toolkit:
          agent = Agent(model=OpenAIChat(id="gpt-4o"), toolkits=[toolkit])
          result = await agent.arun("Go to example.com and tell me the title")
          print(result.content)

  To attach to YOUR running Chrome::

      # 1. Launch Chrome with:
      #    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
      #      --remote-debugging-port=9222 --no-first-run
      # 2. Use:
      config = BrowserConfig(host="127.0.0.1", port=9222)
  """

  def __init__(
    self,
    config: Optional[BrowserConfig] = None,
    browser: Optional[Any] = None,
  ) -> None:
    """
    Args:
        config: Browser configuration. Defaults to ``BrowserConfig()``.
        browser: Inject a pre-built browser instance (useful for testing).
                 When provided, the toolkit does NOT call ``start()``/``stop()``.
    """
    super().__init__()
    self._config = config or BrowserConfig()
    self._browser = browser
    self._owned = browser is None  # True → we manage lifecycle
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
        from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

        self._browser = SeleniumBaseBrowser(self._config)
      await self._browser.start()

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

  async def __aexit__(self, *_: Any) -> None:
    await self.shutdown()

  def __repr__(self) -> str:
    state = "ready" if self._initialized else "not initialized"
    return f"BrowserToolkit({state}, tools={len(self._tools)})"
