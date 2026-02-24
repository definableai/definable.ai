"""
Contract tests: BaseBrowser ABC compliance (Playwright CDP version).

No browser is launched, no Playwright import, no API calls.

Covers:
  - A concrete subclass of BaseBrowser can be instantiated (no TypeError)
  - All abstract method signatures match the declared parameter names
  - PlaywrightBrowser is a subclass of BaseBrowser (import check only)
  - BrowserToolkit is a subclass of Toolkit
"""

import inspect
from typing import Any

import pytest

from definable.browser.base import BaseBrowser


# ---------------------------------------------------------------------------
# Minimal concrete implementation (no Playwright dependency)
# ---------------------------------------------------------------------------


class _ConcreteTestBrowser(BaseBrowser):
  """Minimal BaseBrowser implementation for contract checking."""

  async def start(self) -> None:
    pass

  async def stop(self) -> None:
    pass

  # -- Perception --
  async def snapshot(self, **kwargs: Any) -> str:
    return "(empty)"

  async def screenshot(self, name: str = "screenshot", **kwargs: Any) -> str:
    return "/tmp/screenshot.png"

  async def get_page_info(self) -> str:
    return "URL: https://example.com\nTitle: Test"

  # -- Navigation --
  async def navigate(self, url: str) -> str:
    return f"navigated to {url}"

  async def go_back(self) -> str:
    return "went back"

  async def go_forward(self) -> str:
    return "went forward"

  async def refresh(self) -> str:
    return "refreshed"

  # -- Page state --
  async def get_url(self) -> str:
    return "https://example.com"

  async def get_title(self) -> str:
    return "Test Page"

  async def get_page_source(self, max_chars: int = 20000) -> str:
    return "<html></html>"

  async def get_text(self, ref_or_selector: str = "body") -> str:
    return "page text"

  async def get_attribute(self, ref_or_selector: str, attribute: str) -> str:
    return "value"

  async def is_element_visible(self, ref_or_selector: str) -> str:
    return "true"

  # -- Interaction --
  async def click(self, ref_or_selector: str) -> str:
    return f"clicked {ref_or_selector}"

  async def click_if_visible(self, ref_or_selector: str) -> str:
    return f"clicked if visible {ref_or_selector}"

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    return f"clicked text: {text}"

  async def hover(self, ref_or_selector: str) -> str:
    return f"hovered {ref_or_selector}"

  async def drag(self, from_ref: str, to_ref: str) -> str:
    return f"dragged {from_ref} to {to_ref}"

  async def type_text(self, ref_or_selector: str, text: str, submit: bool = False) -> str:
    return f"typed {text!r} into {ref_or_selector}"

  async def type_slowly(self, ref_or_selector: str, text: str, delay: float = 75.0) -> str:
    return f"slowly typed {text!r} into {ref_or_selector}"

  async def press_key(self, key: str) -> str:
    return f"pressed {key}"

  async def press_keys(self, ref_or_selector: str, keys: str) -> str:
    return f"pressed {keys!r} on {ref_or_selector}"

  async def clear_input(self, ref_or_selector: str) -> str:
    return f"cleared {ref_or_selector}"

  async def select_option(self, ref_or_selector: str, text: str) -> str:
    return f"selected '{text}' in {ref_or_selector}"

  async def check_element(self, ref_or_selector: str) -> str:
    return f"checked {ref_or_selector}"

  async def uncheck_element(self, ref_or_selector: str) -> str:
    return f"unchecked {ref_or_selector}"

  async def is_checked(self, ref_or_selector: str) -> str:
    return "false"

  async def set_value(self, ref_or_selector: str, value: str) -> str:
    return f"set {ref_or_selector}={value}"

  async def set_input_files(self, ref_or_selector: str, paths: list[str]) -> str:
    return f"set files on {ref_or_selector}"

  async def fill_form(self, fields: list[dict[str, Any]]) -> str:
    return f"filled {len(fields)} fields"

  async def execute_js(self, code: str, **kwargs: Any) -> str:
    return "null"

  async def highlight(self, ref_or_selector: str) -> str:
    return f"highlighted {ref_or_selector}"

  async def remove_elements(self, selector: str) -> str:
    return f"removed {selector}"

  # -- Scrolling --
  async def scroll_down(self, amount: int = 3) -> str:
    return f"scrolled down {amount}"

  async def scroll_up(self, amount: int = 3) -> str:
    return f"scrolled up {amount}"

  async def scroll_to_element(self, ref_or_selector: str) -> str:
    return f"scrolled to {ref_or_selector}"

  # -- Waiting --
  async def wait(self, seconds: float = 2.0) -> str:
    return f"waited {seconds}s"

  async def wait_for_element(self, ref_or_selector: str, timeout: float = 10.0) -> str:
    return f"waited for {ref_or_selector}"

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    return f"waited for {text!r}"

  async def wait_for(self, **kwargs: Any) -> str:
    return "waited"

  # -- Tabs --
  async def open_tab(self, url: str = "") -> str:
    return f"opened tab {url}"

  async def close_tab(self) -> str:
    return "closed tab"

  async def get_tabs(self) -> str:
    return "1 tab(s) open"

  async def switch_to_tab(self, index: int) -> str:
    return f"switched to tab {index}"

  # -- Cookies --
  async def get_cookies(self) -> str:
    return "[]"

  async def set_cookie(self, name: str, value: str) -> str:
    return f"set {name}={value}"

  async def clear_cookies(self) -> str:
    return "cleared"

  # -- Storage --
  async def get_storage(self, key: str | None = None, kind: str = "local") -> str:
    return "null"

  async def set_storage(self, key: str, value: str, kind: str = "local") -> str:
    return f"set {key}={value}"

  # -- Dialogs --
  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    return "accepted" if accept else "dismissed"

  # -- Emulation --
  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    return f"geo set: {latitude},{longitude}"

  # -- PDF --
  async def print_to_pdf(self, name: str = "page") -> str:
    return f"/tmp/{name}.pdf"

  # -- Diagnostics --
  async def get_console(self, limit: int = 50, level: str | None = None) -> str:
    return "No console messages."

  async def get_errors(self, limit: int = 20) -> str:
    return "No page errors."

  async def get_network(self, limit: int = 50, url_filter: str | None = None) -> str:
    return "No network requests."


# ---------------------------------------------------------------------------
# Contract: concrete subclass instantiation
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestBaseBrowserContractCompliance:
  """Every BaseBrowser implementation must pass these checks."""

  def test_concrete_subclass_instantiates_without_error(self):
    browser = _ConcreteTestBrowser()
    assert browser is not None

  def test_concrete_subclass_is_instance_of_base_browser(self):
    browser = _ConcreteTestBrowser()
    assert isinstance(browser, BaseBrowser)

  def test_incomplete_subclass_raises_type_error(self):
    """An abstract-method-missing subclass raises TypeError on instantiation."""

    class _IncompleteImpl(BaseBrowser):
      async def start(self) -> None:
        pass

      # Missing all other abstract methods

    with pytest.raises(TypeError):
      _IncompleteImpl()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Contract: abstract method signatures
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestBaseBrowserSignatures:
  """All abstract methods must have the expected parameter names."""

  def _params(self, method_name: str) -> list[str]:
    sig = inspect.signature(getattr(BaseBrowser, method_name))
    return [p for p in sig.parameters if p not in ("self", "kwargs")]

  # Lifecycle
  def test_start_has_no_required_params(self):
    assert self._params("start") == []

  def test_stop_has_no_required_params(self):
    assert self._params("stop") == []

  # Perception
  def test_snapshot_is_abstract(self):
    assert "snapshot" in [m for m in dir(BaseBrowser) if not m.startswith("_")]

  def test_screenshot_has_optional_name(self):
    params = self._params("screenshot")
    assert "name" in params
    sig = inspect.signature(BaseBrowser.screenshot)
    assert sig.parameters["name"].default is not inspect.Parameter.empty

  def test_get_page_info_has_no_required_params(self):
    assert self._params("get_page_info") == []

  # Navigation
  def test_navigate_has_url_param(self):
    assert "url" in self._params("navigate")

  def test_go_back_has_no_required_params(self):
    assert self._params("go_back") == []

  def test_go_forward_has_no_required_params(self):
    assert self._params("go_forward") == []

  def test_refresh_has_no_required_params(self):
    assert self._params("refresh") == []

  # Page state
  def test_get_url_has_no_required_params(self):
    assert self._params("get_url") == []

  def test_get_title_has_no_required_params(self):
    assert self._params("get_title") == []

  def test_get_page_source_has_optional_max_chars(self):
    params = self._params("get_page_source")
    assert "max_chars" in params

  def test_get_text_has_optional_ref_or_selector(self):
    params = self._params("get_text")
    assert "ref_or_selector" in params
    sig = inspect.signature(BaseBrowser.get_text)
    assert sig.parameters["ref_or_selector"].default is not inspect.Parameter.empty

  def test_get_attribute_has_ref_or_selector_and_attribute(self):
    params = self._params("get_attribute")
    assert "ref_or_selector" in params
    assert "attribute" in params

  def test_is_element_visible_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("is_element_visible")

  # Interaction
  def test_click_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("click")

  def test_click_if_visible_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("click_if_visible")

  def test_type_text_has_ref_or_selector_and_text(self):
    params = self._params("type_text")
    assert "ref_or_selector" in params
    assert "text" in params

  def test_press_key_has_key(self):
    assert "key" in self._params("press_key")

  def test_press_keys_has_ref_or_selector_and_keys(self):
    params = self._params("press_keys")
    assert "ref_or_selector" in params
    assert "keys" in params

  def test_clear_input_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("clear_input")

  def test_execute_js_has_code(self):
    assert "code" in self._params("execute_js")

  def test_fill_form_has_fields(self):
    assert "fields" in self._params("fill_form")

  def test_set_input_files_has_ref_or_selector_and_paths(self):
    params = self._params("set_input_files")
    assert "ref_or_selector" in params
    assert "paths" in params

  # Scrolling
  def test_scroll_down_has_optional_amount(self):
    params = self._params("scroll_down")
    assert "amount" in params
    sig = inspect.signature(BaseBrowser.scroll_down)
    assert sig.parameters["amount"].default is not inspect.Parameter.empty

  def test_scroll_up_has_optional_amount(self):
    params = self._params("scroll_up")
    assert "amount" in params
    sig = inspect.signature(BaseBrowser.scroll_up)
    assert sig.parameters["amount"].default is not inspect.Parameter.empty

  def test_scroll_to_element_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("scroll_to_element")

  # Waiting
  def test_wait_has_optional_seconds(self):
    params = self._params("wait")
    assert "seconds" in params
    sig = inspect.signature(BaseBrowser.wait)
    assert sig.parameters["seconds"].default is not inspect.Parameter.empty

  def test_wait_for_element_has_ref_or_selector_and_optional_timeout(self):
    params = self._params("wait_for_element")
    assert "ref_or_selector" in params
    assert "timeout" in params
    sig = inspect.signature(BaseBrowser.wait_for_element)
    assert sig.parameters["timeout"].default is not inspect.Parameter.empty

  def test_wait_for_text_has_text_and_optional_selector_timeout(self):
    params = self._params("wait_for_text")
    assert "text" in params
    assert "selector" in params
    assert "timeout" in params
    sig = inspect.signature(BaseBrowser.wait_for_text)
    assert sig.parameters["selector"].default is not inspect.Parameter.empty
    assert sig.parameters["timeout"].default is not inspect.Parameter.empty

  # Tabs
  def test_open_tab_has_optional_url(self):
    params = self._params("open_tab")
    assert "url" in params
    sig = inspect.signature(BaseBrowser.open_tab)
    assert sig.parameters["url"].default is not inspect.Parameter.empty

  def test_close_tab_has_no_required_params(self):
    assert self._params("close_tab") == []

  # Advanced interaction
  def test_hover_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("hover")

  def test_drag_has_from_and_to(self):
    params = self._params("drag")
    assert "from_ref" in params
    assert "to_ref" in params

  def test_type_slowly_has_ref_or_selector_and_text(self):
    params = self._params("type_slowly")
    assert "ref_or_selector" in params
    assert "text" in params

  def test_select_option_has_ref_or_selector_and_text(self):
    params = self._params("select_option")
    assert "ref_or_selector" in params
    assert "text" in params

  # Cookies
  def test_get_cookies_has_no_required_params(self):
    assert self._params("get_cookies") == []

  def test_set_cookie_has_name_and_value(self):
    params = self._params("set_cookie")
    assert "name" in params
    assert "value" in params

  def test_clear_cookies_has_no_required_params(self):
    assert self._params("clear_cookies") == []

  # Dialogs
  def test_handle_dialog_has_optional_accept_and_prompt(self):
    params = self._params("handle_dialog")
    assert "accept" in params
    assert "prompt_text" in params
    sig = inspect.signature(BaseBrowser.handle_dialog)
    assert sig.parameters["accept"].default is not inspect.Parameter.empty
    assert sig.parameters["prompt_text"].default is not inspect.Parameter.empty

  # Storage
  def test_get_storage_has_key_and_optional_kind(self):
    params = self._params("get_storage")
    assert "key" in params
    assert "kind" in params
    sig = inspect.signature(BaseBrowser.get_storage)
    assert sig.parameters["kind"].default is not inspect.Parameter.empty

  def test_set_storage_has_key_value_and_optional_kind(self):
    params = self._params("set_storage")
    assert "key" in params
    assert "value" in params
    assert "kind" in params

  # Browser state
  def test_set_geolocation_has_lat_lon_and_optional_accuracy(self):
    params = self._params("set_geolocation")
    assert "latitude" in params
    assert "longitude" in params
    assert "accuracy" in params
    sig = inspect.signature(BaseBrowser.set_geolocation)
    assert sig.parameters["accuracy"].default is not inspect.Parameter.empty

  def test_highlight_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("highlight")

  # Text interaction & DOM mutation
  def test_click_by_text_has_text_and_optional_tag(self):
    params = self._params("click_by_text")
    assert "text" in params
    assert "tag_name" in params
    sig = inspect.signature(BaseBrowser.click_by_text)
    assert sig.parameters["tag_name"].default is not inspect.Parameter.empty

  def test_remove_elements_has_selector(self):
    assert "selector" in self._params("remove_elements")

  # Checkboxes
  def test_is_checked_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("is_checked")

  def test_check_element_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("check_element")

  def test_uncheck_element_has_ref_or_selector(self):
    assert "ref_or_selector" in self._params("uncheck_element")

  # Value & tab control
  def test_set_value_has_ref_or_selector_and_value(self):
    params = self._params("set_value")
    assert "ref_or_selector" in params
    assert "value" in params

  def test_get_tabs_has_no_required_params(self):
    assert self._params("get_tabs") == []

  def test_switch_to_tab_has_index(self):
    assert "index" in self._params("switch_to_tab")

  def test_print_to_pdf_has_optional_name(self):
    params = self._params("print_to_pdf")
    assert "name" in params
    sig = inspect.signature(BaseBrowser.print_to_pdf)
    assert sig.parameters["name"].default is not inspect.Parameter.empty

  # Diagnostics
  def test_get_console_has_optional_limit_and_level(self):
    params = self._params("get_console")
    assert "limit" in params
    assert "level" in params

  def test_get_errors_has_optional_limit(self):
    assert "limit" in self._params("get_errors")

  def test_get_network_has_optional_limit_and_url_filter(self):
    params = self._params("get_network")
    assert "limit" in params
    assert "url_filter" in params


# ---------------------------------------------------------------------------
# Contract: PlaywrightBrowser is a BaseBrowser subclass (import check only)
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestPlaywrightBrowserSubclass:
  """PlaywrightBrowser must be a subclass of BaseBrowser."""

  def test_playwright_browser_is_subclass_of_base_browser(self):
    from definable.browser.playwright_browser import PlaywrightBrowser

    assert issubclass(PlaywrightBrowser, BaseBrowser)

  def test_playwright_browser_can_be_instantiated_with_default_config(self):
    from definable.browser.playwright_browser import PlaywrightBrowser

    browser = PlaywrightBrowser()
    assert isinstance(browser, BaseBrowser)


# ---------------------------------------------------------------------------
# Contract: BrowserToolkit is a Toolkit subclass
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestBrowserToolkitSubclass:
  """BrowserToolkit must be a subclass of Toolkit."""

  def test_browser_toolkit_is_subclass_of_toolkit(self):
    from definable.agent.toolkit import Toolkit
    from definable.browser.toolkit import BrowserToolkit

    assert issubclass(BrowserToolkit, Toolkit)

  def test_browser_toolkit_can_be_instantiated(self):
    from definable.browser.toolkit import BrowserToolkit

    toolkit = BrowserToolkit()
    assert toolkit is not None

  def test_browser_toolkit_tools_empty_before_init(self):
    from definable.browser.toolkit import BrowserToolkit

    toolkit = BrowserToolkit()
    assert toolkit.tools == []

  def test_browser_toolkit_initialized_false_before_init(self):
    from definable.browser.toolkit import BrowserToolkit

    toolkit = BrowserToolkit()
    assert toolkit._initialized is False
