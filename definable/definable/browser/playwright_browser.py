"""PlaywrightBrowser — full browser automation via Playwright CDP.

Native async, role-based refs, AI-friendly errors, self-healing connections.
Ported from openclaw pw-session.ts and pw-tools-core.*.ts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import TYPE_CHECKING, Any, Callable

from definable.browser.chrome_launcher import get_chrome_ws_url, launch_chrome, stop_chrome
from definable.browser.config import BrowserConfig
from definable.browser.element_refs import (
  RefResolver,
  RoleSnapshotOptions,
  build_role_snapshot,
)
from definable.browser.errors import normalize_timeout_ms, to_ai_friendly_error
from definable.browser.events import BrowserActionEvent
from definable.browser.page_state import PageState
from definable.browser.base import BaseBrowser
from definable.browser.url_validator import (
  NavigationBlockedError,
  assert_navigation_allowed,
  assert_navigation_result_allowed,
)
from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from playwright.async_api import Browser, BrowserContext, Page, Playwright

  from definable.browser.chrome_launcher import RunningChrome


class PlaywrightBrowser(BaseBrowser):
  """Full browser automation via Playwright CDP.

  Features:
  - Native async (no ThreadPoolExecutor hack)
  - Role-based refs (e1, e2, e3) from ariaSnapshot
  - Unified ref/selector API: browser_click("e1") and browser_click("button.submit") both work
  - AI-friendly error messages
  - Console/error/network ring buffers
  - 3-attempt connection retry with force-disconnect recovery
  - SSRF navigation protection

  Usage::

      async with PlaywrightBrowser(BrowserConfig(headless=True)) as browser:
          await browser.navigate("https://example.com")
          snap = await browser.snapshot()
          await browser.click("e1")
  """

  def __init__(self, config: BrowserConfig | None = None) -> None:
    self._config = config or BrowserConfig()
    self._pw: Playwright | None = None
    self._browser: Browser | None = None
    self._context: BrowserContext | None = None
    self._page: Page | None = None
    self._running_chrome: RunningChrome | None = None
    self._page_state: PageState | None = None
    self._refs = RefResolver()
    self.on_action: Callable[[BrowserActionEvent], Any] | None = None
    self._screenshot_dir: str = ""
    if self._config.downloads_dir:
      self._screenshot_dir = self._config.downloads_dir
    else:
      from definable.utils.workspace import workspace_path

      self._screenshot_dir = str(workspace_path("browser", "screenshots"))

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def start(self) -> None:
    """Launch or connect to Chrome and initialize Playwright."""
    from playwright.async_api import async_playwright

    self._pw = await async_playwright().start()

    if self._config.cdp_url:
      self._browser = await self._connect_with_retry(self._config.cdp_url)
    else:
      self._running_chrome = await launch_chrome(self._config)
      self._browser = await self._connect_with_retry(self._running_chrome.cdp_url)

    # Get or create context and page
    self._context = (
      self._browser.contexts[0]
      if self._browser.contexts
      else await self._browser.new_context(
        viewport={"width": self._config.viewport_width, "height": self._config.viewport_height},
        locale=self._config.locale,
        timezone_id=self._config.timezone,
        user_agent=self._config.user_agent,
      )
    )

    # For self-launched Chrome with persistent profiles, session restore may
    # create extra pages/tabs. Close all but one to avoid multiple windows.
    if self._running_chrome and len(self._context.pages) > 1:
      for extra_page in self._context.pages[1:]:
        with contextlib.suppress(Exception):
          await extra_page.close()

    self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    # Ensure viewport is set (CDP-connected pages often have None viewport)
    if not self._page.viewport_size:
      await self._page.set_viewport_size({"width": self._config.viewport_width, "height": self._config.viewport_height})

    # Attach page state tracking
    self._page_state = PageState(self._config)
    self._page_state.attach(self._page)

    log_info("PlaywrightBrowser started")

  async def _connect_with_retry(self, cdp_url: str) -> "Browser":
    """3-attempt retry: timeout 5s/7s/9s, delay 250ms/500ms/750ms."""
    assert self._pw is not None
    last_err: Exception | None = None
    for attempt in range(3):
      try:
        timeout = 5000 + attempt * 2000
        ws_url = await get_chrome_ws_url(cdp_url, timeout=timeout / 1000) or cdp_url
        return await self._pw.chromium.connect_over_cdp(ws_url, timeout=timeout, slow_mo=self._config.slow_mo)
      except Exception as exc:
        last_err = exc
        delay = 0.25 + attempt * 0.25
        log_debug(f"CDP connect attempt {attempt + 1} failed: {exc}, retrying in {delay}s")
        await asyncio.sleep(delay)
    raise ConnectionError(f"Failed to connect after 3 attempts: {last_err}")

  async def _force_disconnect(self, reason: str = "") -> None:
    """Null cached state, fire-and-forget close. Next call reconnects fresh."""
    old_browser = self._browser
    self._browser = None
    self._page = None
    self._context = None
    if self._page_state:
      self._page_state.detach()
      self._page_state = None
    if old_browser:
      asyncio.create_task(self._safe_close_browser(old_browser))
    if reason:
      log_warning(f"Force disconnected: {reason}")

  async def _safe_close_browser(self, browser: "Browser") -> None:
    with contextlib.suppress(Exception):
      await asyncio.wait_for(browser.close(), timeout=2.0)

  async def stop(self) -> None:
    """Close browser, Playwright, and Chrome process."""
    if self._page_state:
      self._page_state.detach()
      self._page_state = None
    if self._browser:
      with contextlib.suppress(Exception):
        await self._browser.close()
      self._browser = None
    if self._pw:
      with contextlib.suppress(Exception):
        await self._pw.stop()
      self._pw = None
    if self._running_chrome:
      await stop_chrome(self._running_chrome)
      self._running_chrome = None
    self._page = None
    self._context = None
    log_debug("PlaywrightBrowser stopped")

  async def _ensure_page(self) -> "Page":
    if not self._page:
      raise RuntimeError("Browser not started. Call start() first.")
    return self._page

  async def __aenter__(self) -> "PlaywrightBrowser":
    await self.start()
    return self

  async def __aexit__(self, *_: object) -> None:
    await self.stop()

  # ---------------------------------------------------------------------------
  # Event emission
  # ---------------------------------------------------------------------------

  async def _emit(self, action: str, **kwargs: Any) -> None:
    """Emit a BrowserActionEvent to the on_action callback (if set)."""
    if not self.on_action:
      return
    url = ""
    try:
      if self._page:
        url = self._page.url
    except Exception:
      pass
    event = BrowserActionEvent(action=action, url=url, **kwargs)
    try:
      result = self.on_action(event)
      if asyncio.iscoroutine(result):
        await result
    except Exception:
      pass

  # ---------------------------------------------------------------------------
  # Perception — Snapshot & Screenshot
  # ---------------------------------------------------------------------------

  async def snapshot(
    self,
    options: RoleSnapshotOptions | None = None,
    selector: str | None = None,
    frame_selector: str | None = None,
  ) -> str:
    """Get role-based accessibility snapshot with element refs.

    Returns structured text with [ref=e1] labels on interactive elements.
    """
    page = await self._ensure_page()
    try:
      if frame_selector:
        locator = page.frame_locator(frame_selector).locator(selector or ":root")
      elif selector:
        locator = page.locator(selector)
      else:
        locator = page.locator(":root")

      aria_text = await locator.aria_snapshot()
      result = build_role_snapshot(aria_text, options)
      self._refs.store(result.refs, frame_selector)
      stats = f"[{result.stats.refs} refs, {result.stats.interactive} interactive]"
      output = f"{result.snapshot}\n\n{stats}"
      await self._emit("snapshot", result=stats)
      return output
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("snapshot", error=error)
      return f"Error: {error}"

  async def screenshot(
    self,
    name: str = "screenshot",
    ref: str | None = None,
    full_page: bool = False,
  ) -> str:
    """Take screenshot. If ref provided, screenshot that element."""
    page = await self._ensure_page()
    path = os.path.join(self._screenshot_dir, f"{name}.png")
    try:
      if ref:
        locator = self._refs.resolve(page, ref)
        await locator.screenshot(path=path)
      else:
        await page.screenshot(path=path, full_page=full_page)
      result = f"Screenshot saved: {path}"
      await self._emit("screenshot", selector=ref or "", value=path, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref or "")
      await self._emit("screenshot", selector=ref or "", error=error)
      return f"Error: {error}"

  async def get_page_info(self) -> str:
    """URL, title, scroll position, viewport."""
    page = await self._ensure_page()
    try:
      url = page.url
      title = await page.title()
      info = await page.evaluate("""() => ({
        scrollX: Math.round(window.scrollX),
        scrollY: Math.round(window.scrollY),
        scrollHeight: document.documentElement.scrollHeight,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      })""")
      scroll_pct = 0
      if info["scrollHeight"] > info["viewportHeight"]:
        scroll_pct = round(info["scrollY"] / (info["scrollHeight"] - info["viewportHeight"]) * 100)
      return f"URL: {url}\nTitle: {title}\nViewport: {info['viewportWidth']}x{info['viewportHeight']}\nScroll: {info['scrollY']}px ({scroll_pct}%)"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # Navigation
  # ---------------------------------------------------------------------------

  async def navigate(self, url: str) -> str:
    """Navigate to a URL with SSRF protection."""
    try:
      page = await self._ensure_page()
      await assert_navigation_allowed(url)
      timeout = int(self._config.timeout * 1000)
      await page.goto(url, timeout=timeout)
      final_url = page.url
      await assert_navigation_result_allowed(final_url)
      title = await page.title()
      result = f"Navigated to: {final_url}\nTitle: {title}"
      await self._emit("navigate", value=url, result=result)
      return result
    except NavigationBlockedError as exc:
      error = str(exc)
      await self._emit("navigate", value=url, error=error)
      return f"Error: {error}"
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("navigate", value=url, error=error)
      return f"Error: {error}"

  async def go_back(self) -> str:
    page = await self._ensure_page()
    with contextlib.suppress(Exception):
      await page.go_back(timeout=int(self._config.timeout * 1000), wait_until="domcontentloaded")
    result = f"Navigated back to: {page.url}"
    await self._emit("go_back", result=result)
    return result

  async def go_forward(self) -> str:
    page = await self._ensure_page()
    with contextlib.suppress(Exception):
      await page.go_forward(timeout=int(self._config.timeout * 1000), wait_until="domcontentloaded")
    result = f"Navigated forward to: {page.url}"
    await self._emit("go_forward", result=result)
    return result

  async def refresh(self) -> str:
    page = await self._ensure_page()
    try:
      await page.reload(timeout=int(self._config.timeout * 1000))
      result = f"Refreshed: {page.url}"
      await self._emit("refresh", result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("refresh", error=error)
      return f"Error: {error}"

  # ---------------------------------------------------------------------------
  # Page state
  # ---------------------------------------------------------------------------

  async def get_url(self) -> str:
    page = await self._ensure_page()
    return page.url

  async def get_title(self) -> str:
    page = await self._ensure_page()
    try:
      return await page.title()
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def get_page_source(self, max_chars: int = 20000) -> str:
    page = await self._ensure_page()
    try:
      content = await page.content()
      if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[...TRUNCATED at {max_chars} chars]"
      return content
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def get_text(self, ref_or_selector: str = "body") -> str:
    """Return visible text content of an element (ref or CSS selector)."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      return await locator.inner_text()
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc, ref_or_selector)}"

  async def get_attribute(self, ref_or_selector: str, attribute: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      value = await locator.get_attribute(attribute)
      return value or ""
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc, ref_or_selector)}"

  async def is_element_visible(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      visible = await locator.is_visible()
      return "true" if visible else "false"
    except Exception:
      return "false"

  # ---------------------------------------------------------------------------
  # Interaction
  # ---------------------------------------------------------------------------

  async def click(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      timeout = int(self._config.timeout * 1000)
      await locator.click(timeout=timeout)
      result = f"Clicked: {ref_or_selector}"
      await self._emit("click", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("click", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def click_if_visible(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      if await locator.is_visible():
        await locator.click(timeout=int(self._config.timeout * 1000))
        result = f"Clicked: {ref_or_selector}"
        await self._emit("click", selector=ref_or_selector, result=result)
        return result
      return f"Not visible: {ref_or_selector}"
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("click", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    page = await self._ensure_page()
    try:
      if tag_name:
        locator = page.locator(tag_name, has_text=text).first
      else:
        locator = page.get_by_text(text).first
      await locator.click(timeout=int(self._config.timeout * 1000))
      result = f'Clicked element with text: "{text}"'
      await self._emit("click", value=text, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, text)
      await self._emit("click", value=text, error=error)
      return f"Error: {error}"

  async def hover(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.hover(timeout=int(self._config.timeout * 1000))
      result = f"Hovered: {ref_or_selector}"
      await self._emit("hover", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("hover", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def drag(self, from_ref: str, to_ref: str) -> str:
    """Drag from one element to another using Playwright native drag_to."""
    page = await self._ensure_page()
    try:
      from_locator = self._refs.resolve(page, from_ref)
      to_locator = self._refs.resolve(page, to_ref)
      await from_locator.drag_to(to_locator, timeout=int(self._config.timeout * 1000))
      result = f"Dragged: {from_ref} -> {to_ref}"
      await self._emit("drag", selector=from_ref, value=to_ref, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, f"{from_ref} -> {to_ref}")
      await self._emit("drag", selector=from_ref, value=to_ref, error=error)
      return f"Error: {error}"

  async def type_text(self, ref_or_selector: str, text: str, submit: bool = False) -> str:
    """Clear and fill text into an input. Optionally press Enter to submit."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      timeout = int(self._config.timeout * 1000)
      await locator.fill(text, timeout=timeout)
      if submit:
        await locator.press("Enter", timeout=timeout)
      result = f"Typed into {ref_or_selector}: {text}"
      await self._emit("type", selector=ref_or_selector, value=text, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("type", selector=ref_or_selector, value=text, error=error)
      return f"Error: {error}"

  async def type_slowly(self, ref_or_selector: str, text: str, delay: float = 75.0) -> str:
    """Type text with human-like delays between keystrokes."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      timeout = int(self._config.timeout * 1000)
      await locator.click(timeout=timeout)
      await locator.press_sequentially(text, delay=delay, timeout=timeout)
      result = f"Typed slowly into {ref_or_selector}: {text}"
      await self._emit("type", selector=ref_or_selector, value=text, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("type", selector=ref_or_selector, value=text, error=error)
      return f"Error: {error}"

  async def press_key(self, key: str) -> str:
    """Press a keyboard key on the currently focused element."""
    page = await self._ensure_page()
    try:
      await page.keyboard.press(key)
      result = f"Pressed: {key}"
      await self._emit("key_press", value=key, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("key_press", value=key, error=error)
      return f"Error: {error}"

  async def press_keys(self, ref_or_selector: str, keys: str) -> str:
    """Send keystrokes to a specific element."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.press(keys, timeout=int(self._config.timeout * 1000))
      result = f"Pressed {keys} on {ref_or_selector}"
      await self._emit("key_press", selector=ref_or_selector, value=keys, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("key_press", selector=ref_or_selector, value=keys, error=error)
      return f"Error: {error}"

  async def clear_input(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.fill("", timeout=int(self._config.timeout * 1000))
      result = f"Cleared: {ref_or_selector}"
      await self._emit("clear", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("clear", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def select_option(self, ref_or_selector: str, text: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.select_option(label=text, timeout=int(self._config.timeout * 1000))
      result = f'Selected "{text}" in {ref_or_selector}'
      await self._emit("select", selector=ref_or_selector, value=text, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("select", selector=ref_or_selector, value=text, error=error)
      return f"Error: {error}"

  async def check_element(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.check(timeout=int(self._config.timeout * 1000))
      result = f"Checked: {ref_or_selector}"
      await self._emit("check", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("check", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def uncheck_element(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.uncheck(timeout=int(self._config.timeout * 1000))
      result = f"Unchecked: {ref_or_selector}"
      await self._emit("uncheck", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("uncheck", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def is_checked(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      checked = await locator.is_checked()
      return "true" if checked else "false"
    except Exception:
      return "false"

  async def set_value(self, ref_or_selector: str, value: str) -> str:
    """Set an element's value directly (works for sliders, range inputs)."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.evaluate(f"el => el.value = {json.dumps(value)}")
      await locator.dispatch_event("input")
      await locator.dispatch_event("change")
      result = f"Set value of {ref_or_selector} to: {value}"
      await self._emit("set_value", selector=ref_or_selector, value=value, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("set_value", selector=ref_or_selector, value=value, error=error)
      return f"Error: {error}"

  async def set_input_files(self, ref_or_selector: str, paths: list[str]) -> str:
    """Set files on a file input element."""
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.set_input_files(paths)
      result = f"Set {len(paths)} file(s) on {ref_or_selector}"
      await self._emit("set_files", selector=ref_or_selector, value=str(paths), result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("set_files", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def fill_form(self, fields: list[dict[str, Any]]) -> str:
    """Batch form fill. Each field: {ref, type, value}.

    Checkboxes/radios use setChecked(), others use fill().
    """
    page = await self._ensure_page()
    timeout = int(self._config.timeout * 1000)
    filled = 0
    errors: list[str] = []

    for field in fields:
      ref = str(field.get("ref", "")).strip()
      ftype = str(field.get("type", "")).strip()
      raw_value = field.get("value", "")
      if not ref or not ftype:
        continue

      try:
        locator = self._refs.resolve(page, ref)
        if ftype in ("checkbox", "radio"):
          checked = raw_value in (True, 1, "1", "true")
          await locator.set_checked(checked, timeout=timeout)
        else:
          value = str(raw_value) if raw_value is not None else ""
          await locator.fill(value, timeout=timeout)
        filled += 1
      except Exception as exc:
        errors.append(f"{ref}: {to_ai_friendly_error(exc, ref)}")

    result = f"Filled {filled}/{len(fields)} fields"
    if errors:
      result += "\nErrors:\n" + "\n".join(errors)
    await self._emit("fill_form", value=f"{filled}/{len(fields)} fields", result=result)
    return result

  async def highlight(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.highlight()
      result = f"Highlighted: {ref_or_selector}"
      await self._emit("highlight", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("highlight", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  async def remove_elements(self, selector: str) -> str:
    page = await self._ensure_page()
    try:
      count = await page.evaluate(
        """(selector) => {
          const els = document.querySelectorAll(selector);
          els.forEach(el => el.remove());
          return els.length;
        }""",
        selector,
      )
      result = f"Removed {count} element(s) matching: {selector}"
      await self._emit("remove_elements", selector=selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, selector)
      await self._emit("remove_elements", selector=selector, error=error)
      return f"Error: {error}"

  # ---------------------------------------------------------------------------
  # Scrolling
  # ---------------------------------------------------------------------------

  async def scroll_down(self, amount: int = 3) -> str:
    page = await self._ensure_page()
    try:
      await page.evaluate(f"window.scrollBy(0, window.innerHeight * {amount})")
      result = f"Scrolled down {amount} screen(s)"
      await self._emit("scroll_down", value=str(amount), result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("scroll_down", value=str(amount), error=error)
      return f"Error: {error}"

  async def scroll_up(self, amount: int = 3) -> str:
    page = await self._ensure_page()
    try:
      await page.evaluate(f"window.scrollBy(0, -window.innerHeight * {amount})")
      result = f"Scrolled up {amount} screen(s)"
      await self._emit("scroll_up", value=str(amount), result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("scroll_up", value=str(amount), error=error)
      return f"Error: {error}"

  async def scroll_to_element(self, ref_or_selector: str) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.scroll_into_view_if_needed()
      result = f"Scrolled to: {ref_or_selector}"
      await self._emit("scroll_to", selector=ref_or_selector, result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref_or_selector)
      await self._emit("scroll_to", selector=ref_or_selector, error=error)
      return f"Error: {error}"

  # ---------------------------------------------------------------------------
  # Waiting
  # ---------------------------------------------------------------------------

  async def wait(self, seconds: float = 2.0) -> str:
    await asyncio.sleep(seconds)
    return f"Waited {seconds}s"

  async def wait_for_element(self, ref_or_selector: str, timeout: float = 10.0) -> str:
    page = await self._ensure_page()
    try:
      locator = self._refs.resolve(page, ref_or_selector)
      await locator.wait_for(state="visible", timeout=int(timeout * 1000))
      return f"Element appeared: {ref_or_selector}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc, ref_or_selector)}"

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    page = await self._ensure_page()
    try:
      await page.get_by_text(text).first.wait_for(state="visible", timeout=int(timeout * 1000))
      return f'Text appeared: "{text}"'
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc, text)}"

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
    """Unified wait with 6 condition types (all optional, sequential)."""
    page = await self._ensure_page()
    timeout_ms = normalize_timeout_ms(int((timeout or 20) * 1000), 20000)
    conditions = []

    try:
      if text:
        await page.get_by_text(text).first.wait_for(state="visible", timeout=timeout_ms)
        conditions.append(f'text "{text}" appeared')

      if text_gone:
        await page.get_by_text(text_gone).first.wait_for(state="hidden", timeout=timeout_ms)
        conditions.append(f'text "{text_gone}" disappeared')

      if selector:
        await page.locator(selector).first.wait_for(state="visible", timeout=timeout_ms)
        conditions.append(f'selector "{selector}" appeared')

      if url:
        await page.wait_for_url(url, timeout=timeout_ms)
        conditions.append(f'URL matches "{url}"')

      if load_state:
        await page.wait_for_load_state(load_state, timeout=timeout_ms)  # type: ignore[arg-type]
        conditions.append(f'load state "{load_state}"')

      if fn:
        await page.wait_for_function(fn, timeout=timeout_ms)
        conditions.append("JS function returned truthy")

      if not conditions:
        return "No wait conditions specified."
      return "Wait complete: " + ", ".join(conditions)
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # JavaScript execution
  # ---------------------------------------------------------------------------

  async def execute_js(self, code: str, ref: str | None = None, timeout: float | None = None) -> str:
    """Execute JS with dual timeout protection.

    Inner: Promise.race timeout in browser context.
    Outer: asyncio.wait_for on the Playwright call.
    On outer timeout: force_disconnect to unblock stuck operations.
    """
    page = await self._ensure_page()
    timeout_ms = normalize_timeout_ms(int((timeout or 20) * 1000), 20000)
    # Inner timeout leaves headroom for Playwright overhead
    inner_timeout_ms = max(1000, timeout_ms - 500)

    try:
      # Wrap user code with Promise.race timeout in the browser context
      wrapped = """(args) => {
        "use strict";
        var fnBody = args.fnBody, timeoutMs = args.timeoutMs;
        try {
          var candidate = eval("(" + fnBody + ")");
          var result = typeof candidate === "function" ? candidate() : candidate;
          if (result && typeof result.then === "function") {
            return Promise.race([
              result,
              new Promise(function(_, reject) {
                setTimeout(function() { reject(new Error("evaluate timed out after " + timeoutMs + "ms")); }, timeoutMs);
              })
            ]);
          }
          return result;
        } catch (err) {
          throw new Error("JS eval error: " + (err && err.message ? err.message : String(err)));
        }
      }"""

      if ref:
        locator = self._refs.resolve(page, ref)
        raw = await asyncio.wait_for(
          locator.evaluate(wrapped, {"fnBody": code, "timeoutMs": inner_timeout_ms}),
          timeout=timeout_ms / 1000 + 1,
        )
      else:
        raw = await asyncio.wait_for(
          page.evaluate(wrapped, {"fnBody": code, "timeoutMs": inner_timeout_ms}),
          timeout=timeout_ms / 1000 + 1,
        )

      if raw is None:
        output = "undefined"
      elif isinstance(raw, str):
        output = raw
      else:
        output = json.dumps(raw, indent=2, default=str)
      await self._emit("execute_js", value=code[:200], result=output[:200])
      return output
    except asyncio.TimeoutError:
      await self._force_disconnect("JS evaluate timed out")
      error = f"JavaScript execution timed out after {timeout_ms}ms. Connection reset."
      await self._emit("execute_js", value=code[:200], error=error)
      return f"Error: {error}"
    except Exception as exc:
      error = to_ai_friendly_error(exc, ref or "")
      await self._emit("execute_js", value=code[:200], error=error)
      return f"Error: {error}"

  # ---------------------------------------------------------------------------
  # Tabs
  # ---------------------------------------------------------------------------

  async def get_tabs(self) -> str:
    page = await self._ensure_page()
    try:
      pages = page.context.pages
      count = len(pages)
      lines = [f"{count} tab(s):"]
      for i, p in enumerate(pages):
        active = " (active)" if p == self._page else ""
        lines.append(f"  [{i}] {p.url}{active}")
      return "\n".join(lines)
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def open_tab(self, url: str = "") -> str:
    page = await self._ensure_page()
    try:
      new_page = await page.context.new_page()
      if url:
        await assert_navigation_allowed(url)
        await new_page.goto(url, timeout=int(self._config.timeout * 1000))
      self._page = new_page
      if self._page_state:
        self._page_state.detach()
        self._page_state = PageState(self._config)
        self._page_state.attach(new_page)
      result = f"Opened new tab: {new_page.url}"
      await self._emit("tab_open", value=url, result=result)
      return result
    except NavigationBlockedError as exc:
      error = str(exc)
      await self._emit("tab_open", value=url, error=error)
      return f"Error: {error}"
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("tab_open", value=url, error=error)
      return f"Error: {error}"

  async def close_tab(self) -> str:
    page = await self._ensure_page()
    try:
      pages = page.context.pages
      await page.close()
      # Switch to last remaining tab
      if pages:
        remaining = [p for p in pages if p != page]
        if remaining:
          self._page = remaining[-1]
          if self._page_state:
            self._page_state.detach()
            self._page_state = PageState(self._config)
            self._page_state.attach(self._page)
          result = f"Closed tab. Active: {self._page.url}"
          await self._emit("tab_close", result=result)
          return result
      self._page = None
      result = "Closed last tab."
      await self._emit("tab_close", result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("tab_close", error=error)
      return f"Error: {error}"

  async def switch_to_tab(self, index: int) -> str:
    page = await self._ensure_page()
    try:
      pages = page.context.pages
      if index < 0 or index >= len(pages):
        return f"Error: Tab index {index} out of range (0-{len(pages) - 1})"
      self._page = pages[index]
      if self._page_state:
        self._page_state.detach()
        self._page_state = PageState(self._config)
        self._page_state.attach(self._page)
      await self._page.bring_to_front()
      result = f"Switched to tab [{index}]: {self._page.url}"
      await self._emit("tab_switch", value=str(index), result=result)
      return result
    except Exception as exc:
      error = to_ai_friendly_error(exc)
      await self._emit("tab_switch", value=str(index), error=error)
      return f"Error: {error}"

  # ---------------------------------------------------------------------------
  # Cookies
  # ---------------------------------------------------------------------------

  async def get_cookies(self) -> str:
    page = await self._ensure_page()
    try:
      cookies = await page.context.cookies()
      return json.dumps(cookies, indent=2, default=str)
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def set_cookie(self, name: str, value: str) -> str:
    page = await self._ensure_page()
    try:
      url = page.url
      await page.context.add_cookies([{"name": name, "value": value, "url": url}])
      return f"Cookie set: {name}={value}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def clear_cookies(self) -> str:
    page = await self._ensure_page()
    try:
      await page.context.clear_cookies()
      return "Cookies cleared."
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------

  async def get_storage(self, key: str | None = None, kind: str = "local") -> str:
    page = await self._ensure_page()
    try:
      values = await page.evaluate(
        """({kind, key}) => {
          const store = kind === "session" ? window.sessionStorage : window.localStorage;
          if (key) {
            const value = store.getItem(key);
            return value === null ? {} : {[key]: value};
          }
          const out = {};
          for (let i = 0; i < store.length; i++) {
            const k = store.key(i);
            if (k) { out[k] = store.getItem(k); }
          }
          return out;
        }""",
        {"kind": kind, "key": key},
      )
      return json.dumps(values, indent=2)
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def set_storage(self, key: str, value: str, kind: str = "local") -> str:
    page = await self._ensure_page()
    try:
      await page.evaluate(
        """({kind, key, value}) => {
          const store = kind === "session" ? window.sessionStorage : window.localStorage;
          store.setItem(key, value);
        }""",
        {"kind": kind, "key": key, "value": value},
      )
      return f"Storage set: {key}={value} ({kind})"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def clear_storage(self, kind: str = "local") -> str:
    page = await self._ensure_page()
    try:
      await page.evaluate(
        """({kind}) => {
          const store = kind === "session" ? window.sessionStorage : window.localStorage;
          store.clear();
        }""",
        {"kind": kind},
      )
      return f"{kind.capitalize()} storage cleared."
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # Dialogs
  # ---------------------------------------------------------------------------

  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    """Arm a dialog handler for the next dialog event."""
    page = await self._ensure_page()

    async def _handler(dialog: Any) -> None:
      try:
        if accept:
          await dialog.accept(prompt_text)
        else:
          await dialog.dismiss()
      except Exception:
        pass

    page.once("dialog", _handler)
    action = "accept" if accept else "dismiss"
    return f"Dialog handler armed: will {action} next dialog."

  # ---------------------------------------------------------------------------
  # Emulation
  # ---------------------------------------------------------------------------

  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    page = await self._ensure_page()
    try:
      ctx = page.context
      await ctx.set_geolocation({"latitude": latitude, "longitude": longitude, "accuracy": accuracy})
      # Grant geolocation permission for current origin
      try:
        origin = page.url.split("/")[0] + "//" + page.url.split("/")[2] if "://" in page.url else ""
        if origin:
          await ctx.grant_permissions(["geolocation"], origin=origin)
      except Exception:
        pass
      return f"Geolocation set: {latitude}, {longitude}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def set_offline(self, offline: bool = True) -> str:
    page = await self._ensure_page()
    try:
      await page.context.set_offline(offline)
      return f"Offline mode: {'on' if offline else 'off'}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def set_extra_headers(self, headers: dict[str, str]) -> str:
    page = await self._ensure_page()
    try:
      await page.context.set_extra_http_headers(headers)
      return f"Extra headers set: {list(headers.keys())}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def resize_viewport(self, width: int, height: int) -> str:
    page = await self._ensure_page()
    try:
      await page.set_viewport_size({"width": width, "height": height})
      return f"Viewport resized to: {width}x{height}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  async def emulate_media(self, color_scheme: str | None = None) -> str:
    page = await self._ensure_page()
    try:
      await page.emulate_media(color_scheme=color_scheme)  # type: ignore[arg-type]
      return f"Media emulation: color_scheme={color_scheme}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # PDF
  # ---------------------------------------------------------------------------

  async def print_to_pdf(self, name: str = "page") -> str:
    page = await self._ensure_page()
    path = os.path.join(self._screenshot_dir, f"{name}.pdf")
    try:
      await page.pdf(path=path, print_background=True)
      return f"PDF saved: {path}"
    except Exception as exc:
      return f"Error: {to_ai_friendly_error(exc)}"

  # ---------------------------------------------------------------------------
  # Page state queries (console, errors, network)
  # ---------------------------------------------------------------------------

  async def get_console(self, limit: int = 50, level: str | None = None) -> str:
    if not self._page_state:
      return "No page state available. Start browser first."
    return self._page_state.format_console(limit=limit, level=level)

  async def get_errors(self, limit: int = 20) -> str:
    if not self._page_state:
      return "No page state available. Start browser first."
    return self._page_state.format_errors(limit=limit)

  async def get_network(self, limit: int = 50, url_filter: str | None = None) -> str:
    if not self._page_state:
      return "No page state available. Start browser first."
    return self._page_state.format_network(limit=limit, url_filter=url_filter)
