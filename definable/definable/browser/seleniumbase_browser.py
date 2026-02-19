"""SeleniumBaseBrowser — CDP-backed browser using SeleniumBase (no WebDriver).

SeleniumBase CDP mode drives Chrome directly via Chrome DevTools Protocol.
It produces zero automation fingerprints — no webdriver flag, no blink
automation signals, no info banners.

Key design decisions:
- SeleniumBase's ``sb_cdp.Chrome`` API is **synchronous**. All calls are
  dispatched to a single dedicated ``ThreadPoolExecutor`` thread so the
  agent's asyncio event loop is never blocked.
- A single-thread executor guarantees that SeleniumBase's internal asyncio
  event loop (created in its own thread at startup) is always called from
  the correct thread.
- All public methods return plain ``str`` results so the agent can read them.
  Errors are returned as ``"Error: <message>"`` strings (never raised).

Connection modes (set via BrowserConfig):
  A. Fresh Chrome (default):
       ``BrowserConfig(headless=False)``
  B. Persistent profile (retains cookies/sessions):
       ``BrowserConfig(user_data_dir="/tmp/my-profile")``
  C. Attach to YOUR running Chrome:
       Launch Chrome with ``--remote-debugging-port=9222 --no-first-run``,
       then use ``BrowserConfig(host="127.0.0.1", port=9222)``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from definable.browser.base import BaseBrowser
from definable.browser.config import BrowserConfig
from definable.utils.log import log_debug, log_error, log_info, log_warning


class SeleniumBaseBrowser(BaseBrowser):
  """Browser automation via SeleniumBase CDP mode.

  Example::

      config = BrowserConfig(headless=False)
      browser = SeleniumBaseBrowser(config)
      await browser.start()
      print(await browser.navigate("https://example.com"))
      print(await browser.get_text("h1"))
      await browser.stop()
  """

  def __init__(self, config: Optional[BrowserConfig] = None) -> None:
    self._config = config or BrowserConfig()
    self._sb: Any = None
    # Single-thread executor — all SeleniumBase calls run in this one thread.
    # SeleniumBase's internal asyncio loop is bound to the thread it was
    # created in, so we must always dispatch from the same thread.
    self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
      max_workers=1,
      thread_name_prefix="definable_browser",
    )
    self._screenshot_dir = tempfile.mkdtemp(prefix="definable_browser_")

  # ---------------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------------

  async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous callable in the dedicated browser thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

  def _assert_started(self) -> None:
    if self._sb is None:
      raise RuntimeError("Browser not started — call start() first")

  def _start_sync(self) -> Any:
    """Synchronous Chrome launch — runs inside the executor thread."""
    try:
      from seleniumbase import sb_cdp
    except ImportError as exc:
      raise ImportError("seleniumbase is required for BrowserToolkit. Install: pip install 'definable[browser]'") from exc

    kwargs: dict[str, Any] = {
      "lang": self._config.lang,
      "sandbox": self._config.sandbox,
      "headless": self._config.headless,
    }

    # Connection mode C: attach to existing Chrome
    if self._config.host and self._config.port:
      kwargs["host"] = self._config.host
      kwargs["port"] = self._config.port
      log_debug(f"SeleniumBaseBrowser: Attaching via CDP to {self._config.host}:{self._config.port}")

    # Connection mode B: persistent profile
    elif self._config.user_data_dir:
      os.makedirs(self._config.user_data_dir, exist_ok=True)
      kwargs["user_data_dir"] = self._config.user_data_dir
      log_debug(f"SeleniumBaseBrowser: Launching with profile at {self._config.user_data_dir}")

    # Optional extras
    if self._config.proxy:
      kwargs["proxy"] = self._config.proxy
    if self._config.user_agent:
      kwargs["agent"] = self._config.user_agent
    if self._config.browser_executable_path:
      kwargs["browser_executable_path"] = self._config.browser_executable_path
    if self._config.browser_args:
      kwargs["browser_args"] = list(self._config.browser_args)
    if self._config.incognito:
      kwargs["incognito"] = True
    if self._config.mobile:
      kwargs["mobile"] = True
    if self._config.ad_block:
      kwargs["ad_block"] = True

    log_info("SeleniumBaseBrowser: Starting Chrome (CDP mode)...")
    sb = sb_cdp.Chrome(**kwargs)
    log_info("SeleniumBaseBrowser: Ready")
    return sb

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def start(self) -> None:
    """Launch (or attach to) Chrome using SeleniumBase CDP mode."""
    self._sb = await self._run(self._start_sync)

  async def stop(self) -> None:
    """Close the browser and release all resources."""
    if self._sb is not None:
      try:
        await self._run(lambda: self._sb.quit())
      except Exception as exc:
        log_warning(f"SeleniumBaseBrowser: Error stopping browser: {exc}")
      finally:
        self._sb = None
    self._executor.shutdown(wait=False)
    log_info("SeleniumBaseBrowser: Stopped")

  # ---------------------------------------------------------------------------
  # Navigation
  # ---------------------------------------------------------------------------

  async def navigate(self, url: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _nav() -> str:
        self._sb.open(url)
        current = self._sb.get_current_url()
        title = self._sb.get_title()
        return f"Navigated to {current} | Title: {title}"

      return await self._run(_nav)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.navigate({url!r}): {exc}")
      return f"Error: {exc}"

  async def go_back(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _back() -> str:
        self._sb.go_back()
        return f"Navigated back → {self._sb.get_current_url()}"

      return await self._run(_back)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.go_back: {exc}")
      return f"Error: {exc}"

  async def go_forward(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _fwd() -> str:
        self._sb.go_forward()
        return f"Navigated forward → {self._sb.get_current_url()}"

      return await self._run(_fwd)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.go_forward: {exc}")
      return f"Error: {exc}"

  async def refresh(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _ref() -> str:
        self._sb.reload()
        return f"Refreshed → {self._sb.get_current_url()}"

      return await self._run(_ref)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.refresh: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Page state
  # ---------------------------------------------------------------------------

  async def get_url(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      return await self._run(self._sb.get_current_url)
    except Exception as exc:
      return f"Error: {exc}"

  async def get_title(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      return await self._run(self._sb.get_title)
    except Exception as exc:
      return f"Error: {exc}"

  async def get_page_source(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _src() -> str:
        html = self._sb.get_page_source()
        if len(html) > 20000:
          html = html[:20000] + "\n... [truncated]"
        return html

      return await self._run(_src)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_page_source: {exc}")
      return f"Error: {exc}"

  async def get_text(self, selector: str = "body") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      return await self._run(lambda: self._sb.get_text(selector))
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_text({selector!r}): {exc}")
      return self._enrich_css_error(str(exc), selector)

  async def get_attribute(self, selector: str, attribute: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      val = await self._run(lambda: self._sb.get_attribute(selector, attribute))
      return str(val) if val is not None else f"Attribute '{attribute}' not found on {selector!r}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_attribute({selector!r}, {attribute!r}): {exc}")
      return f"Error: {exc}"

  async def is_element_visible(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      result = await self._run(lambda: self._sb.is_element_visible(selector))
      return "true" if result else "false"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.is_element_visible({selector!r}): {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Interaction
  # ---------------------------------------------------------------------------

  async def click(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.click(selector))
      return f"Clicked: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.click({selector!r}): {exc}")
      return self._enrich_css_error(str(exc), selector)

  async def click_if_visible(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.click_if_visible(selector))
      return f"Clicked (if visible): {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.click_if_visible({selector!r}): {exc}")
      return f"Error: {exc}"

  async def type_text(self, selector: str, text: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    preview = text[:50] + ("..." if len(text) > 50 else "")
    try:
      await self._run(lambda: self._sb.type(selector, text))
      return f"Typed into {selector!r}: {preview}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.type_text({selector!r}): {exc}")
      return f"Error: {exc}"

  async def press_keys(self, selector: str, keys: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.press_keys(selector, keys))
      return f"Pressed {keys!r} on {selector!r}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.press_keys({selector!r}, {keys!r}): {exc}")
      return f"Error: {exc}"

  async def press_key(self, key: str) -> str:
    """Send a keypress to the currently focused element (no selector needed)."""
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.press_keys("body", key))
      return f"Pressed key: {key!r}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.press_key({key!r}): {exc}")
      return f"Error: {exc}"

  async def clear_input(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.clear_input(selector))
      return f"Cleared input: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.clear_input({selector!r}): {exc}")
      return f"Error: {exc}"

  async def execute_js(self, code: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      result = await self._run(lambda: self._sb.execute_script(code))
      return str(result) if result is not None else "null"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.execute_js: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Scrolling
  # ---------------------------------------------------------------------------

  async def scroll_down(self, amount: int = 3) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      pixels = amount * 500
      await self._run(lambda: self._sb.scroll_by_y(pixels))
      return f"Scrolled down {amount} units ({pixels}px)"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.scroll_down: {exc}")
      return f"Error: {exc}"

  async def scroll_up(self, amount: int = 3) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      pixels = amount * 500
      await self._run(lambda: self._sb.scroll_by_y(-pixels))
      return f"Scrolled up {amount} units ({pixels}px)"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.scroll_up: {exc}")
      return f"Error: {exc}"

  async def scroll_to_element(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.scroll_into_view(selector))
      return f"Scrolled to: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.scroll_to_element({selector!r}): {exc}")
      return self._enrich_css_error(str(exc), selector)

  # ---------------------------------------------------------------------------
  # Waiting
  # ---------------------------------------------------------------------------

  async def wait(self, seconds: float = 2.0) -> str:
    secs = max(0.1, min(120.0, float(seconds)))
    await asyncio.sleep(secs)
    return f"Waited {secs:.1f}s"

  async def wait_for_element(self, selector: str, timeout: float = 10.0) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.wait_for_element(selector, timeout=timeout))
      return f"Element appeared: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.wait_for_element({selector!r}): {exc}")
      enriched = self._enrich_css_error(str(exc), selector)
      if enriched.startswith("Error: Invalid CSS"):
        return enriched
      return f"Error: element not found within {timeout}s: {selector}"

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.wait_for_text(text, selector=selector, timeout=timeout))
      return f"Text appeared: {text!r}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.wait_for_text({text!r}): {exc}")
      return f"Error: text not found within {timeout}s: {text!r}"

  # ---------------------------------------------------------------------------
  # Screenshot
  # ---------------------------------------------------------------------------

  async def screenshot(self, name: str = "screenshot") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      if not name.endswith(".png"):
        name = name + ".png"
      path = os.path.join(self._screenshot_dir, name)

      def _shot() -> str:
        self._sb.save_screenshot(name, folder=self._screenshot_dir)
        return path

      saved = await self._run(_shot)
      return f"Screenshot saved: {saved}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.screenshot: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Tabs
  # ---------------------------------------------------------------------------

  async def open_tab(self, url: str = "") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _open() -> str:
        if url:
          self._sb.open_new_tab(url=url)
          return f"Opened new tab → {self._sb.get_current_url()}"
        else:
          self._sb.open_new_tab()
          return "Opened new tab"

      return await self._run(_open)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.open_tab: {exc}")
      return f"Error: {exc}"

  async def close_tab(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.close_active_tab())
      return "Closed active tab"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.close_tab: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Advanced interaction
  # ---------------------------------------------------------------------------

  async def hover(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.hover(selector))
      return f"Hovered over: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.hover({selector!r}): {exc}")
      return f"Error: {exc}"

  async def drag(self, from_selector: str, to_selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _drag_sync() -> str:
        import json

        src_q = json.dumps(from_selector)
        dst_q = json.dumps(to_selector)
        js = f"""(function() {{
var src = document.querySelector({src_q});
var dst = document.querySelector({dst_q});
if (!src) return 'Error: source element not found: ' + {src_q};
if (!dst) return 'Error: target element not found: ' + {dst_q};
var dt = new DataTransfer();
src.dispatchEvent(new DragEvent('dragstart', {{dataTransfer: dt, bubbles: true}}));
dst.dispatchEvent(new DragEvent('dragover',  {{dataTransfer: dt, bubbles: true}}));
dst.dispatchEvent(new DragEvent('drop',      {{dataTransfer: dt, bubbles: true}}));
src.dispatchEvent(new DragEvent('dragend',   {{dataTransfer: dt, bubbles: true}}));
return 'ok';
}})()"""
        result = self._sb.execute_script(js)
        if isinstance(result, str) and result.startswith("Error"):
          return result
        return f"Dragged {from_selector!r} → {to_selector!r}"

      return await self._run(_drag_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.drag: {exc}")
      return f"Error: {exc}"

  async def type_slowly(self, selector: str, text: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    preview = text[:50] + ("..." if len(text) > 50 else "")
    try:

      def _type_slowly_sync() -> str:
        import time

        # Focus and clear the target field
        self._sb.click(selector)
        self._sb.clear_input(selector)
        # Send one keystroke at a time with a 75 ms human-paced delay
        for char in text:
          self._sb.press_keys(selector, char)
          time.sleep(0.075)
        return f"Slowly typed into {selector!r}: {preview}"

      return await self._run(_type_slowly_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.type_slowly({selector!r}): {exc}")
      return f"Error: {exc}"

  async def select_option(self, selector: str, text: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _select_sync() -> str:
        self._sb.select_option_by_text(selector, text)
        return f"Selected '{text}' in {selector!r}"

      return await self._run(_select_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.select_option({selector!r}, {text!r}): {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Cookies
  # ---------------------------------------------------------------------------

  async def get_cookies(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _get_cookies_sync() -> str:
        import json

        cookies = self._sb.get_cookies()
        return json.dumps(cookies, indent=2)

      return await self._run(_get_cookies_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_cookies: {exc}")
      return f"Error: {exc}"

  async def set_cookie(self, name: str, value: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _set_sync() -> str:
        import json

        cookie_str = json.dumps(f"{name}={value}")
        self._sb.execute_script(f"document.cookie = {cookie_str};")
        return f"Set cookie: {name}={value}"

      return await self._run(_set_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.set_cookie({name!r}): {exc}")
      return f"Error: {exc}"

  async def clear_cookies(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _clear_sync() -> str:
        js = (
          "document.cookie.split(';').forEach(function(c) {"
          "  var n = c.split('=')[0].trim();"
          "  document.cookie = n + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';"
          "});"
        )
        self._sb.execute_script(js)
        return "All cookies cleared"

      return await self._run(_clear_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.clear_cookies: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Dialogs
  # ---------------------------------------------------------------------------

  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _handle_sync() -> str:
        try:
          if accept:
            self._sb.accept_alert()
            return "Accepted dialog"
          else:
            self._sb.dismiss_alert()
            return "Dismissed dialog"
        except Exception:
          return "No dialog present"

      return await self._run(_handle_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.handle_dialog: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------

  async def get_storage(self, key: str, storage_type: str = "local") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    store = "localStorage" if storage_type.lower() == "local" else "sessionStorage"
    try:

      def _get_sync() -> str:
        import json

        js = f"(function() {{ return window[{json.dumps(store)}].getItem({json.dumps(key)}); }})()"
        val = self._sb.execute_script(js)
        return str(val) if val is not None else f"null (key {key!r} not found in {store})"

      return await self._run(_get_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_storage({key!r}): {exc}")
      return f"Error: {exc}"

  async def set_storage(self, key: str, value: str, storage_type: str = "local") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    store = "localStorage" if storage_type.lower() == "local" else "sessionStorage"
    try:

      def _set_sync() -> str:
        import json

        js = f"window[{json.dumps(store)}].setItem({json.dumps(key)}, {json.dumps(value)});"
        self._sb.execute_script(js)
        return f"Set {store}[{key!r}] = {value!r}"

      return await self._run(_set_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.set_storage({key!r}): {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Browser state
  # ---------------------------------------------------------------------------

  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _geo_sync() -> str:
        # Try SeleniumBase CDP's native execute_cdp_cmd first, then JS fallback
        try:
          self._sb.execute_cdp_cmd(
            "Emulation.setGeolocationOverride",
            {"latitude": latitude, "longitude": longitude, "accuracy": accuracy},
          )
        except AttributeError:
          js = (
            f"Object.defineProperty(navigator, 'geolocation', {{"
            f"  configurable: true,"
            f"  get: function() {{"
            f"    return {{"
            f"      getCurrentPosition: function(ok) {{"
            f"        ok({{coords: {{latitude: {latitude}, longitude: {longitude}, accuracy: {accuracy}}}, timestamp: Date.now()}});"
            f"      }},"
            f"      watchPosition: function(ok) {{"
            f"        ok({{coords: {{latitude: {latitude}, longitude: {longitude}, accuracy: {accuracy}}}, timestamp: Date.now()}});"
            f"        return 0;"
            f"      }},"
            f"      clearWatch: function() {{}}"
            f"    }};"
            f"  }}"
            f"}});"
          )
          self._sb.execute_script(js)
        return f"Geolocation set: lat={latitude}, lon={longitude}, accuracy={accuracy}m"

      return await self._run(_geo_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.set_geolocation: {exc}")
      return f"Error: {exc}"

  async def highlight(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _highlight_sync() -> str:
        import json

        sel_q = json.dumps(selector)
        js = (
          "(function() {"
          f"var el = document.querySelector({sel_q});"
          f"if (!el) return 'Error: element not found: ' + {sel_q};"
          "var prev = el.style.outline;"
          "var prevOff = el.style.outlineOffset;"
          "el.style.outline = '3px solid #FFD700';"
          "el.style.outlineOffset = '2px';"
          "setTimeout(function() {"
          "  el.style.outline = prev;"
          "  el.style.outlineOffset = prevOff;"
          "}, 2000);"
          "return 'highlighted';"
          "})()"
        )
        result = self._sb.execute_script(js)
        if isinstance(result, str) and result.startswith("Error"):
          return result
        return f"Highlighted: {selector} (2 s)"

      return await self._run(_highlight_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.highlight({selector!r}): {exc}")
      return f"Error: {exc}"

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _click_by_text_sync() -> str:
        import json

        tag = tag_name or "*"
        tag_q = json.dumps(tag)
        text_q = json.dumps(text)
        text_lower_q = json.dumps(text.lower())
        js = (
          "(function() {"
          f"var elems = document.querySelectorAll({tag_q});"
          f"var target = {text_lower_q};"
          "for (var i = 0; i < elems.length; i++) {"
          "  var t = (elems[i].textContent || '').trim().toLowerCase();"
          "  if (t === target || t.includes(target)) {"
          "    elems[i].click();"
          f"   return 'Clicked <' + elems[i].tagName.toLowerCase() + '> with text: ' + {text_q};"
          "  }"
          "}"
          "return null;"
          "})()"
        )
        result = self._sb.execute_script(js)
        return result or f"Error: no element with text {text!r}"

      return await self._run(_click_by_text_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.click_by_text({text!r}): {exc}")
      return f"Error: {exc}"

  async def remove_elements(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _remove_sync() -> str:
        import json

        sel_q = json.dumps(selector)
        js = (
          "(function() {"
          f"var elems = document.querySelectorAll({sel_q});"
          "var count = elems.length;"
          "for (var i = count - 1; i >= 0; i--) { elems[i].remove(); }"
          "return count;"
          "})()"
        )
        count = self._sb.execute_script(js)
        return f"Removed {count} element(s): {selector}"

      return await self._run(_remove_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.remove_elements({selector!r}): {exc}")
      return f"Error: {exc}"

  async def is_checked(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      result = await self._run(lambda: self._sb.is_checked(selector))
      return "true" if result else "false"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.is_checked({selector!r}): {exc}")
      return f"Error: {exc}"

  async def check_element(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.check_if_unchecked(selector))
      return f"Checked: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.check_element({selector!r}): {exc}")
      return f"Error: {exc}"

  async def uncheck_element(self, selector: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.uncheck_if_checked(selector))
      return f"Unchecked: {selector}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.uncheck_element({selector!r}): {exc}")
      return f"Error: {exc}"

  async def set_value(self, selector: str, value: str) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.set_value(selector, value))
      return f"Set value of {selector!r} to {value!r}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.set_value({selector!r}, {value!r}): {exc}")
      return f"Error: {exc}"

  async def get_tabs(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      tabs = await self._run(lambda: self._sb.get_tabs())
      return f"{len(tabs)} tab(s) open"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_tabs: {exc}")
      return f"Error: {exc}"

  async def switch_to_tab(self, index: int) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _switch_sync() -> str:
        tabs = self._sb.get_tabs()
        if index < 0 or index >= len(tabs):
          return f"Error: tab index {index} out of range (0–{len(tabs) - 1})"
        self._sb.switch_to_tab(tabs[index])
        return f"Switched to tab {index} of {len(tabs)}"

      return await self._run(_switch_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.switch_to_tab({index}): {exc}")
      return f"Error: {exc}"

  async def print_to_pdf(self, name: str = "page") -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      if not name.endswith(".pdf"):
        name = name + ".pdf"

      def _pdf_sync() -> str:
        self._sb.print_to_pdf(name, folder=self._screenshot_dir)
        return os.path.join(self._screenshot_dir, name)

      path = await self._run(_pdf_sync)
      return f"PDF saved: {path}"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.print_to_pdf: {exc}")
      return f"Error: {exc}"

  async def get_page_info(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _info_sync() -> str:
        url = self._sb.get_current_url()
        title = self._sb.get_title()
        metrics = self._sb.execute_script(
          "(function() {"
          "return {"
          "  scrollY: window.scrollY,"
          "  totalH: document.body.scrollHeight,"
          "  viewH: window.innerHeight,"
          "  links: document.querySelectorAll('a[href]').length,"
          "  buttons: document.querySelectorAll('button,input[type=submit],input[type=button]').length,"
          "  inputs: document.querySelectorAll('input:not([type=hidden]),textarea,select').length,"
          "  forms: document.querySelectorAll('form').length"
          "};"
          "})()"
        )
        scroll_y = metrics.get("scrollY", 0)
        total_h = metrics.get("totalH", 1)
        view_h = metrics.get("viewH", 1)
        denominator = max(1, total_h - view_h)
        pct = int(100 * scroll_y / denominator) if total_h > view_h else 100
        return "\n".join([
          f"URL: {url}",
          f"Title: {title}",
          f"Scroll: {pct}% ({scroll_y}px / {total_h}px total)",
          f"Interactive: {metrics.get('links', 0)} links, "
          f"{metrics.get('buttons', 0)} buttons, "
          f"{metrics.get('inputs', 0)} inputs, "
          f"{metrics.get('forms', 0)} forms",
        ])

      return await self._run(_info_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_page_info: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # Page snapshot (accessibility-tree-like DOM representation)
  # ---------------------------------------------------------------------------

  async def get_page_snapshot(self, max_chars: int = 8000) -> str:
    """Return a simplified, labeled DOM representation for LLM consumption.

    Produces an accessibility-tree-like text view showing interactive and
    landmark elements with labels, roles, and text content. This gives
    the LLM a structured understanding of the page without raw HTML.
    """
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:

      def _snapshot_sync() -> str:
        js = (
          """(function() {
var out = [];
var idx = 0;
var MAX = """
          + str(max_chars)
          + """;

function label(el) {
  return el.getAttribute('aria-label')
    || el.getAttribute('alt')
    || el.getAttribute('title')
    || el.getAttribute('placeholder')
    || el.getAttribute('name')
    || '';
}

function walk(node, depth) {
  if (idx > 9999) return;
  if (node.nodeType !== 1) return;
  var tag = node.tagName.toLowerCase();

  // Skip hidden, script, style, noscript, svg internals
  if (tag === 'script' || tag === 'style' || tag === 'noscript') return;
  var style = window.getComputedStyle(node);
  if (style.display === 'none' || style.visibility === 'hidden') return;

  var isInteractive = (
    tag === 'a' || tag === 'button' || tag === 'input' ||
    tag === 'textarea' || tag === 'select' || tag === 'details' ||
    tag === 'summary' || node.getAttribute('role') === 'button' ||
    node.getAttribute('role') === 'link' || node.getAttribute('tabindex') !== null ||
    node.onclick !== null
  );
  var isLandmark = (
    tag === 'nav' || tag === 'main' || tag === 'header' ||
    tag === 'footer' || tag === 'aside' || tag === 'section' ||
    tag === 'article' || tag === 'form' || tag === 'h1' ||
    tag === 'h2' || tag === 'h3' || tag === 'h4' || tag === 'h5' ||
    tag === 'h6' || tag === 'table' || tag === 'ul' || tag === 'ol' ||
    tag === 'dialog' || tag === 'img'
  );

  if (isInteractive || isLandmark) {
    idx++;
    var indent = '  '.repeat(Math.min(depth, 8));
    var line = indent + '[' + idx + '] ' + tag;

    var role = node.getAttribute('role');
    if (role) line += '[' + role + ']';

    var type = node.getAttribute('type');
    if (tag === 'input' && type) line += '[' + type + ']';

    var lbl = label(node);
    var txt = '';
    if (tag === 'a' || tag === 'button' || tag === 'summary' ||
        tag === 'h1' || tag === 'h2' || tag === 'h3' || tag === 'h4' ||
        tag === 'h5' || tag === 'h6' || tag === 'option' || tag === 'label') {
      txt = (node.textContent || '').trim().substring(0, 80);
    }

    if (lbl) line += ' "' + lbl + '"';
    else if (txt) line += ' "' + txt.replace(/\\s+/g, ' ') + '"';

    var href = node.getAttribute('href');
    if (href && href !== '#' && !href.startsWith('javascript:')) {
      line += ' href="' + href.substring(0, 80) + '"';
    }

    var val = node.value;
    if (val && (tag === 'input' || tag === 'textarea' || tag === 'select')) {
      line += ' value="' + String(val).substring(0, 40) + '"';
    }

    if (node.disabled) line += ' [disabled]';
    if (node.checked) line += ' [checked]';
    if (node.required) line += ' [required]';

    // CSS selector hint for actionable elements
    if (isInteractive) {
      var id = node.id;
      if (id) {
        line += ' sel="#' + id + '"';
      } else {
        var cls = Array.from(node.classList).slice(0, 2).join('.');
        if (cls) line += ' sel="' + tag + '.' + cls + '"';
      }
    }

    out.push(line);
  }

  var kids = node.children;
  for (var i = 0; i < kids.length; i++) {
    walk(kids[i], isInteractive || isLandmark ? depth + 1 : depth);
  }
}

walk(document.body, 0);

var result = out.join('\\n');
if (result.length > MAX) {
  result = result.substring(0, MAX) + '\\n... [truncated at ' + MAX + ' chars]';
}
return result || '(empty page)';
})()"""
        )
        return self._sb.execute_script(js)

      return await self._run(_snapshot_sync)
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.get_page_snapshot: {exc}")
      return f"Error: {exc}"

  # ---------------------------------------------------------------------------
  # CSS selector error enrichment
  # ---------------------------------------------------------------------------

  def _enrich_css_error(self, error_msg: str, selector: str) -> str:
    """Add actionable hints when the LLM uses invalid CSS pseudo-selectors."""
    msg = str(error_msg)
    invalid_pseudos = {
      ":has-text(": "':has-text()' is not valid CSS. Use browser_click_by_text() instead.",
      ":contains(": "':contains()' is not valid CSS. Use browser_click_by_text() instead.",
      ":text(": "':text()' is not valid CSS. Use browser_click_by_text() instead.",
      ":visible": "':visible' is not valid CSS. Use browser_is_visible() to check visibility.",
      ">>": "'>>' (deep combinator) is not valid CSS. Use standard CSS selectors.",
    }
    for pattern, hint in invalid_pseudos.items():
      if pattern in selector:
        return f"Error: Invalid CSS selector '{selector}'. {hint}"
    return f"Error: {msg}"

  # ---------------------------------------------------------------------------
  # CAPTCHA
  # ---------------------------------------------------------------------------

  async def solve_captcha(self) -> str:
    try:
      self._assert_started()
    except RuntimeError as exc:
      return f"Error: {exc}"
    try:
      await self._run(lambda: self._sb.solve_captcha())
      return "CAPTCHA solved"
    except Exception as exc:
      log_error(f"SeleniumBaseBrowser.solve_captcha: {exc}")
      return f"Error: {exc}"
