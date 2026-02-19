"""
Behavioral tests: BrowserToolkit + MockModel dispatch (SeleniumBase CDP version).

No real browser launched, no seleniumbase API calls.
Uses a mock browser to verify tool dispatch and argument passing.
"""

from __future__ import annotations

import pytest

from definable.browser.base import BaseBrowser
from definable.browser.config import BrowserConfig
from definable.browser.toolkit import BrowserToolkit

# ---------------------------------------------------------------------------
# Expected tool counts
# ---------------------------------------------------------------------------

_EXPECTED_TOOLS = 50  # See toolkit.py _make_tools for the full list


# ---------------------------------------------------------------------------
# Mock browser (no seleniumbase dependency)
# ---------------------------------------------------------------------------


class _MockBrowser(BaseBrowser):
  """In-memory mock browser for testing tool dispatch."""

  def __init__(self) -> None:
    self._started = False
    self._calls: list[tuple[str, tuple, dict]] = []

  def _record(self, name: str, *args, **kwargs) -> str:
    self._calls.append((name, args, kwargs))
    return f"ok:{name}"

  async def start(self) -> None:
    self._started = True

  async def stop(self) -> None:
    self._started = False

  async def navigate(self, url: str) -> str:
    return self._record("navigate", url)

  async def go_back(self) -> str:
    return self._record("go_back")

  async def go_forward(self) -> str:
    return self._record("go_forward")

  async def refresh(self) -> str:
    return self._record("refresh")

  async def get_url(self) -> str:
    return self._record("get_url")

  async def get_title(self) -> str:
    return self._record("get_title")

  async def get_page_source(self) -> str:
    return self._record("get_page_source")

  async def get_text(self, selector: str = "body") -> str:
    return self._record("get_text", selector)

  async def get_attribute(self, selector: str, attribute: str) -> str:
    return self._record("get_attribute", selector, attribute)

  async def is_element_visible(self, selector: str) -> str:
    return self._record("is_element_visible", selector)

  async def click(self, selector: str) -> str:
    return self._record("click", selector)

  async def click_if_visible(self, selector: str) -> str:
    return self._record("click_if_visible", selector)

  async def type_text(self, selector: str, text: str) -> str:
    return self._record("type_text", selector, text)

  async def press_keys(self, selector: str, keys: str) -> str:
    return self._record("press_keys", selector, keys)

  async def clear_input(self, selector: str) -> str:
    return self._record("clear_input", selector)

  async def execute_js(self, code: str) -> str:
    return self._record("execute_js", code)

  async def scroll_down(self, amount: int = 3) -> str:
    return self._record("scroll_down", amount)

  async def scroll_up(self, amount: int = 3) -> str:
    return self._record("scroll_up", amount)

  async def scroll_to_element(self, selector: str) -> str:
    return self._record("scroll_to_element", selector)

  async def wait(self, seconds: float = 2.0) -> str:
    return self._record("wait", seconds)

  async def wait_for_element(self, selector: str, timeout: float = 10.0) -> str:
    return self._record("wait_for_element", selector, timeout)

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    return self._record("wait_for_text", text, selector, timeout)

  async def screenshot(self, name: str = "screenshot") -> str:
    return self._record("screenshot", name)

  async def open_tab(self, url: str = "") -> str:
    return self._record("open_tab", url)

  async def close_tab(self) -> str:
    return self._record("close_tab")

  async def hover(self, selector: str) -> str:
    return self._record("hover", selector)

  async def drag(self, from_selector: str, to_selector: str) -> str:
    return self._record("drag", from_selector, to_selector)

  async def type_slowly(self, selector: str, text: str) -> str:
    return self._record("type_slowly", selector, text)

  async def select_option(self, selector: str, text: str) -> str:
    return self._record("select_option", selector, text)

  async def get_cookies(self) -> str:
    return self._record("get_cookies")

  async def set_cookie(self, name: str, value: str) -> str:
    return self._record("set_cookie", name, value)

  async def clear_cookies(self) -> str:
    return self._record("clear_cookies")

  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    return self._record("handle_dialog", accept, prompt_text)

  async def get_storage(self, key: str, storage_type: str = "local") -> str:
    return self._record("get_storage", key, storage_type)

  async def set_storage(self, key: str, value: str, storage_type: str = "local") -> str:
    return self._record("set_storage", key, value, storage_type)

  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    return self._record("set_geolocation", latitude, longitude, accuracy)

  async def highlight(self, selector: str) -> str:
    return self._record("highlight", selector)

  async def get_page_info(self) -> str:
    return self._record("get_page_info")

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    return self._record("click_by_text", text, tag_name)

  async def remove_elements(self, selector: str) -> str:
    return self._record("remove_elements", selector)

  async def is_checked(self, selector: str) -> str:
    return self._record("is_checked", selector)

  async def check_element(self, selector: str) -> str:
    return self._record("check_element", selector)

  async def uncheck_element(self, selector: str) -> str:
    return self._record("uncheck_element", selector)

  async def set_value(self, selector: str, value: str) -> str:
    return self._record("set_value", selector, value)

  async def get_tabs(self) -> str:
    return self._record("get_tabs")

  async def switch_to_tab(self, index: int) -> str:
    return self._record("switch_to_tab", index)

  async def print_to_pdf(self, name: str = "page") -> str:
    return self._record("print_to_pdf", name)

  async def solve_captcha(self) -> str:
    return self._record("solve_captcha")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_browser():
  return _MockBrowser()


@pytest.fixture
async def toolkit(mock_browser):
  tk = BrowserToolkit(browser=mock_browser)
  await tk.initialize()
  yield tk
  await tk.shutdown()


# ---------------------------------------------------------------------------
# Tool count and names
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestBrowserToolkitTools:
  async def test_tools_count_after_init(self, toolkit):
    assert len(toolkit.tools) == _EXPECTED_TOOLS

  async def test_tool_names_are_unique(self, toolkit):
    names = [t.name for t in toolkit.tools]
    assert len(names) == len(set(names))

  async def test_expected_tool_names_present(self, toolkit):
    names = {t.name for t in toolkit.tools}
    expected = {
      "browser_navigate",
      "browser_go_back",
      "browser_go_forward",
      "browser_refresh",
      "browser_get_url",
      "browser_get_title",
      "browser_get_text",
      "browser_get_source",
      "browser_get_attribute",
      "browser_is_visible",
      "browser_click",
      "browser_click_if_visible",
      "browser_type",
      "browser_press_keys",
      "browser_press_key",
      "browser_clear_input",
      "browser_execute_js",
      "browser_scroll_down",
      "browser_scroll_up",
      "browser_scroll_to",
      "browser_wait",
      "browser_wait_for_element",
      "browser_wait_for_text",
      "browser_screenshot",
      "browser_open_tab",
      "browser_close_tab",
      "browser_hover",
      "browser_drag",
      "browser_type_slowly",
      "browser_select_option",
      "browser_get_cookies",
      "browser_set_cookie",
      "browser_clear_cookies",
      "browser_handle_dialog",
      "browser_get_storage",
      "browser_set_storage",
      "browser_set_geolocation",
      "browser_highlight",
      "browser_get_page_info",
      "browser_snapshot",
      "browser_click_by_text",
      "browser_remove_elements",
      "browser_is_checked",
      "browser_check",
      "browser_uncheck",
      "browser_set_value",
      "browser_get_tabs",
      "browser_switch_to_tab",
      "browser_print_to_pdf",
      "browser_solve_captcha",
    }
    assert expected == names

  async def test_tools_empty_before_init(self):
    tk = BrowserToolkit()
    assert tk.tools == []

  async def test_tools_empty_after_shutdown(self, mock_browser):
    tk = BrowserToolkit(browser=mock_browser)
    await tk.initialize()
    await tk.shutdown()
    assert tk.tools == []


# ---------------------------------------------------------------------------
# Tool dispatch via entrypoints
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestBrowserToolDispatch:
  async def test_navigate_dispatched(self, toolkit, mock_browser):
    nav = next(t for t in toolkit.tools if t.name == "browser_navigate")
    result = await nav.entrypoint(url="https://example.com")
    assert result == "ok:navigate"
    assert mock_browser._calls[-1] == ("navigate", ("https://example.com",), {})

  async def test_click_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_click")
    result = await fn.entrypoint(selector="#btn")
    assert result == "ok:click"
    assert mock_browser._calls[-1] == ("click", ("#btn",), {})

  async def test_type_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_type")
    result = await fn.entrypoint(selector="#q", text="hello")
    assert result == "ok:type_text"
    assert mock_browser._calls[-1] == ("type_text", ("#q", "hello"), {})

  async def test_get_text_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_text")
    result = await fn.entrypoint(selector="h1")
    assert result == "ok:get_text"
    assert mock_browser._calls[-1] == ("get_text", ("h1",), {})

  async def test_scroll_down_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_scroll_down")
    result = await fn.entrypoint(amount=5)
    assert result == "ok:scroll_down"
    assert mock_browser._calls[-1] == ("scroll_down", (5,), {})

  async def test_screenshot_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_screenshot")
    result = await fn.entrypoint(name="test_shot")
    assert result == "ok:screenshot"
    assert mock_browser._calls[-1] == ("screenshot", ("test_shot",), {})

  async def test_execute_js_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_execute_js")
    result = await fn.entrypoint(code="return 1+1")
    assert result == "ok:execute_js"
    assert mock_browser._calls[-1] == ("execute_js", ("return 1+1",), {})

  async def test_solve_captcha_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_solve_captcha")
    result = await fn.entrypoint()
    assert result == "ok:solve_captcha"
    assert mock_browser._calls[-1][0] == "solve_captcha"

  async def test_wait_for_text_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_wait_for_text")
    result = await fn.entrypoint(text="Welcome", selector="#msg", timeout=5.0)
    assert result == "ok:wait_for_text"
    assert mock_browser._calls[-1] == ("wait_for_text", ("Welcome", "#msg", 5.0), {})

  async def test_open_tab_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_open_tab")
    result = await fn.entrypoint(url="https://example.com")
    assert result == "ok:open_tab"

  async def test_get_attribute_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_attribute")
    result = await fn.entrypoint(selector="a", attribute="href")
    assert result == "ok:get_attribute"
    assert mock_browser._calls[-1] == ("get_attribute", ("a", "href"), {})

  async def test_press_keys_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_press_keys")
    result = await fn.entrypoint(selector="#q", keys="\n")
    assert result == "ok:press_keys"

  async def test_hover_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_hover")
    result = await fn.entrypoint(selector="#menu")
    assert result == "ok:hover"
    assert mock_browser._calls[-1] == ("hover", ("#menu",), {})

  async def test_drag_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_drag")
    result = await fn.entrypoint(from_selector="#item", to_selector="#target")
    assert result == "ok:drag"
    assert mock_browser._calls[-1] == ("drag", ("#item", "#target"), {})

  async def test_type_slowly_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_type_slowly")
    result = await fn.entrypoint(selector="#q", text="hello")
    assert result == "ok:type_slowly"
    assert mock_browser._calls[-1] == ("type_slowly", ("#q", "hello"), {})

  async def test_select_option_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_select_option")
    result = await fn.entrypoint(selector="#country", text="Canada")
    assert result == "ok:select_option"
    assert mock_browser._calls[-1] == ("select_option", ("#country", "Canada"), {})

  async def test_get_cookies_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_cookies")
    result = await fn.entrypoint()
    assert result == "ok:get_cookies"

  async def test_set_cookie_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_set_cookie")
    result = await fn.entrypoint(name="session", value="abc123")
    assert result == "ok:set_cookie"
    assert mock_browser._calls[-1] == ("set_cookie", ("session", "abc123"), {})

  async def test_clear_cookies_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_clear_cookies")
    result = await fn.entrypoint()
    assert result == "ok:clear_cookies"

  async def test_handle_dialog_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_handle_dialog")
    result = await fn.entrypoint(accept=True, prompt_text="")
    assert result == "ok:handle_dialog"

  async def test_get_storage_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_storage")
    result = await fn.entrypoint(key="token", storage_type="local")
    assert result == "ok:get_storage"
    assert mock_browser._calls[-1] == ("get_storage", ("token", "local"), {})

  async def test_set_storage_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_set_storage")
    result = await fn.entrypoint(key="token", value="xyz", storage_type="session")
    assert result == "ok:set_storage"
    assert mock_browser._calls[-1] == ("set_storage", ("token", "xyz", "session"), {})

  async def test_set_geolocation_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_set_geolocation")
    result = await fn.entrypoint(latitude=37.7749, longitude=-122.4194, accuracy=5.0)
    assert result == "ok:set_geolocation"
    assert mock_browser._calls[-1] == ("set_geolocation", (37.7749, -122.4194, 5.0), {})

  async def test_highlight_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_highlight")
    result = await fn.entrypoint(selector="h1")
    assert result == "ok:highlight"
    assert mock_browser._calls[-1] == ("highlight", ("h1",), {})

  async def test_get_page_info_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_page_info")
    result = await fn.entrypoint()
    assert result == "ok:get_page_info"

  async def test_click_by_text_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_click_by_text")
    result = await fn.entrypoint(text="Sign in", tag_name="button")
    assert result == "ok:click_by_text"
    assert mock_browser._calls[-1] == ("click_by_text", ("Sign in", "button"), {})

  async def test_remove_elements_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_remove_elements")
    result = await fn.entrypoint(selector=".cookie-banner")
    assert result == "ok:remove_elements"
    assert mock_browser._calls[-1] == ("remove_elements", (".cookie-banner",), {})

  async def test_is_checked_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_is_checked")
    result = await fn.entrypoint(selector="#agree")
    assert result == "ok:is_checked"

  async def test_check_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_check")
    result = await fn.entrypoint(selector="#agree")
    assert result == "ok:check_element"

  async def test_uncheck_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_uncheck")
    result = await fn.entrypoint(selector="#newsletter")
    assert result == "ok:uncheck_element"

  async def test_set_value_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_set_value")
    result = await fn.entrypoint(selector="#volume", value="75")
    assert result == "ok:set_value"
    assert mock_browser._calls[-1] == ("set_value", ("#volume", "75"), {})

  async def test_get_tabs_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_get_tabs")
    result = await fn.entrypoint()
    assert result == "ok:get_tabs"

  async def test_switch_to_tab_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_switch_to_tab")
    result = await fn.entrypoint(index=1)
    assert result == "ok:switch_to_tab"
    assert mock_browser._calls[-1] == ("switch_to_tab", (1,), {})

  async def test_print_to_pdf_dispatched(self, toolkit, mock_browser):
    fn = next(t for t in toolkit.tools if t.name == "browser_print_to_pdf")
    result = await fn.entrypoint(name="report")
    assert result == "ok:print_to_pdf"
    assert mock_browser._calls[-1] == ("print_to_pdf", ("report",), {})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestBrowserToolkitLifecycle:
  async def test_initialize_starts_owned_browser(self):
    mb = _MockBrowser()
    tk = BrowserToolkit()
    tk._browser = mb
    tk._owned = True
    await tk.initialize()
    assert mb._started is True
    await tk.shutdown()

  async def test_context_manager(self, mock_browser):
    async with BrowserToolkit(browser=mock_browser) as tk:
      assert len(tk.tools) == _EXPECTED_TOOLS
    assert len(tk.tools) == 0

  async def test_initialize_is_idempotent(self, mock_browser):
    tk = BrowserToolkit(browser=mock_browser)
    await tk.initialize()
    count1 = len(tk.tools)
    await tk.initialize()
    count2 = len(tk.tools)
    assert count1 == count2
    await tk.shutdown()

  async def test_repr_contains_tool_count(self, toolkit):
    r = repr(toolkit)
    assert str(_EXPECTED_TOOLS) in r


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestBrowserToolkitConfig:
  def test_default_config_is_browser_config(self):
    tk = BrowserToolkit()
    assert isinstance(tk._config, BrowserConfig)

  def test_custom_config_stored(self):
    cfg = BrowserConfig(headless=True, lang="de")
    tk = BrowserToolkit(config=cfg)
    assert tk._config is cfg

  def test_injected_browser_not_owned(self, mock_browser):
    tk = BrowserToolkit(browser=mock_browser)
    assert tk._owned is False

  def test_no_injected_browser_is_owned(self):
    tk = BrowserToolkit()
    assert tk._owned is True
