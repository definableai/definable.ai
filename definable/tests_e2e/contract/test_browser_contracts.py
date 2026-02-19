"""
Contract tests: BaseBrowser ABC compliance (SeleniumBase CDP version).

No browser is launched, no seleniumbase import, no API calls.

Covers:
  - A concrete subclass of BaseBrowser can be instantiated (no TypeError)
  - All abstract method signatures match the declared parameter names
  - SeleniumBaseBrowser is a subclass of BaseBrowser (import check only)
  - BrowserToolkit is a subclass of Toolkit
"""

import inspect

import pytest

from definable.browser.base import BaseBrowser
from definable.browser.config import BrowserConfig


# ---------------------------------------------------------------------------
# Minimal concrete implementation (no seleniumbase dependency)
# ---------------------------------------------------------------------------


class _ConcreteTestBrowser(BaseBrowser):
  """Minimal BaseBrowser implementation for contract checking."""

  async def start(self) -> None:
    pass

  async def stop(self) -> None:
    pass

  async def navigate(self, url: str) -> str:
    return f"navigated to {url}"

  async def go_back(self) -> str:
    return "went back"

  async def go_forward(self) -> str:
    return "went forward"

  async def refresh(self) -> str:
    return "refreshed"

  async def get_url(self) -> str:
    return "https://example.com"

  async def get_title(self) -> str:
    return "Test Page"

  async def get_page_source(self) -> str:
    return "<html></html>"

  async def get_text(self, selector: str = "body") -> str:
    return "page text"

  async def get_attribute(self, selector: str, attribute: str) -> str:
    return "value"

  async def is_element_visible(self, selector: str) -> str:
    return "true"

  async def click(self, selector: str) -> str:
    return f"clicked {selector}"

  async def click_if_visible(self, selector: str) -> str:
    return f"clicked if visible {selector}"

  async def type_text(self, selector: str, text: str) -> str:
    return f"typed {text!r} into {selector}"

  async def press_keys(self, selector: str, keys: str) -> str:
    return f"pressed {keys!r} on {selector}"

  async def clear_input(self, selector: str) -> str:
    return f"cleared {selector}"

  async def execute_js(self, code: str) -> str:
    return "null"

  async def scroll_down(self, amount: int = 3) -> str:
    return f"scrolled down {amount}"

  async def scroll_up(self, amount: int = 3) -> str:
    return f"scrolled up {amount}"

  async def scroll_to_element(self, selector: str) -> str:
    return f"scrolled to {selector}"

  async def wait(self, seconds: float = 2.0) -> str:
    return f"waited {seconds}s"

  async def wait_for_element(self, selector: str, timeout: float = 10.0) -> str:
    return f"waited for {selector}"

  async def wait_for_text(self, text: str, selector: str = "body", timeout: float = 10.0) -> str:
    return f"waited for {text!r}"

  async def screenshot(self, name: str = "screenshot") -> str:
    return "/tmp/screenshot.png"

  async def open_tab(self, url: str = "") -> str:
    return f"opened tab {url}"

  async def close_tab(self) -> str:
    return "closed tab"

  async def hover(self, selector: str) -> str:
    return f"hovered {selector}"

  async def drag(self, from_selector: str, to_selector: str) -> str:
    return f"dragged {from_selector} to {to_selector}"

  async def type_slowly(self, selector: str, text: str) -> str:
    return f"slowly typed {text!r} into {selector}"

  async def select_option(self, selector: str, text: str) -> str:
    return f"selected '{text}' in {selector}"

  async def get_cookies(self) -> str:
    return "[]"

  async def set_cookie(self, name: str, value: str) -> str:
    return f"set {name}={value}"

  async def clear_cookies(self) -> str:
    return "cleared"

  async def handle_dialog(self, accept: bool = True, prompt_text: str = "") -> str:
    return "accepted" if accept else "dismissed"

  async def get_storage(self, key: str, storage_type: str = "local") -> str:
    return "null"

  async def set_storage(self, key: str, value: str, storage_type: str = "local") -> str:
    return f"set {key}={value}"

  async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 10.0) -> str:
    return f"geo set: {latitude},{longitude}"

  async def highlight(self, selector: str) -> str:
    return f"highlighted {selector}"

  async def get_page_info(self) -> str:
    return "URL: https://example.com\nTitle: Test"

  async def click_by_text(self, text: str, tag_name: str = "") -> str:
    return f"clicked text: {text}"

  async def remove_elements(self, selector: str) -> str:
    return f"removed {selector}"

  async def is_checked(self, selector: str) -> str:
    return "false"

  async def check_element(self, selector: str) -> str:
    return f"checked {selector}"

  async def uncheck_element(self, selector: str) -> str:
    return f"unchecked {selector}"

  async def set_value(self, selector: str, value: str) -> str:
    return f"set {selector}={value}"

  async def get_tabs(self) -> str:
    return "1 tab(s) open"

  async def switch_to_tab(self, index: int) -> str:
    return f"switched to tab {index}"

  async def print_to_pdf(self, name: str = "page") -> str:
    return f"/tmp/{name}.pdf"

  async def solve_captcha(self) -> str:
    return "CAPTCHA solved"


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
    return [p for p in sig.parameters if p != "self"]

  # Lifecycle
  def test_start_has_no_required_params(self):
    assert self._params("start") == []

  def test_stop_has_no_required_params(self):
    assert self._params("stop") == []

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

  def test_get_page_source_has_no_required_params(self):
    assert self._params("get_page_source") == []

  def test_get_text_has_optional_selector(self):
    params = self._params("get_text")
    assert "selector" in params
    sig = inspect.signature(BaseBrowser.get_text)
    assert sig.parameters["selector"].default is not inspect.Parameter.empty

  def test_get_attribute_has_selector_and_attribute(self):
    params = self._params("get_attribute")
    assert "selector" in params
    assert "attribute" in params

  def test_is_element_visible_has_selector(self):
    assert "selector" in self._params("is_element_visible")

  # Interaction
  def test_click_has_selector(self):
    assert "selector" in self._params("click")

  def test_click_if_visible_has_selector(self):
    assert "selector" in self._params("click_if_visible")

  def test_type_text_has_selector_and_text(self):
    params = self._params("type_text")
    assert "selector" in params
    assert "text" in params

  def test_press_keys_has_selector_and_keys(self):
    params = self._params("press_keys")
    assert "selector" in params
    assert "keys" in params

  def test_clear_input_has_selector(self):
    assert "selector" in self._params("clear_input")

  def test_execute_js_has_code(self):
    assert "code" in self._params("execute_js")

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

  def test_scroll_to_element_has_selector(self):
    assert "selector" in self._params("scroll_to_element")

  # Waiting
  def test_wait_has_optional_seconds(self):
    params = self._params("wait")
    assert "seconds" in params
    sig = inspect.signature(BaseBrowser.wait)
    assert sig.parameters["seconds"].default is not inspect.Parameter.empty

  def test_wait_for_element_has_selector_and_optional_timeout(self):
    params = self._params("wait_for_element")
    assert "selector" in params
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

  # Screenshot
  def test_screenshot_has_optional_name(self):
    params = self._params("screenshot")
    assert "name" in params
    sig = inspect.signature(BaseBrowser.screenshot)
    assert sig.parameters["name"].default is not inspect.Parameter.empty

  # Tabs
  def test_open_tab_has_optional_url(self):
    params = self._params("open_tab")
    assert "url" in params
    sig = inspect.signature(BaseBrowser.open_tab)
    assert sig.parameters["url"].default is not inspect.Parameter.empty

  def test_close_tab_has_no_required_params(self):
    assert self._params("close_tab") == []

  # CAPTCHA
  def test_solve_captcha_has_no_required_params(self):
    assert self._params("solve_captcha") == []

  # Advanced interaction
  def test_hover_has_selector(self):
    assert "selector" in self._params("hover")

  def test_drag_has_from_and_to(self):
    params = self._params("drag")
    assert "from_selector" in params
    assert "to_selector" in params

  def test_type_slowly_has_selector_and_text(self):
    params = self._params("type_slowly")
    assert "selector" in params
    assert "text" in params

  def test_select_option_has_selector_and_text(self):
    params = self._params("select_option")
    assert "selector" in params
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
  def test_get_storage_has_key_and_optional_storage_type(self):
    params = self._params("get_storage")
    assert "key" in params
    assert "storage_type" in params
    sig = inspect.signature(BaseBrowser.get_storage)
    assert sig.parameters["storage_type"].default is not inspect.Parameter.empty

  def test_set_storage_has_key_value_and_optional_storage_type(self):
    params = self._params("set_storage")
    assert "key" in params
    assert "value" in params
    assert "storage_type" in params

  # Browser state
  def test_set_geolocation_has_lat_lon_and_optional_accuracy(self):
    params = self._params("set_geolocation")
    assert "latitude" in params
    assert "longitude" in params
    assert "accuracy" in params
    sig = inspect.signature(BaseBrowser.set_geolocation)
    assert sig.parameters["accuracy"].default is not inspect.Parameter.empty

  def test_highlight_has_selector(self):
    assert "selector" in self._params("highlight")

  def test_get_page_info_has_no_required_params(self):
    assert self._params("get_page_info") == []

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
  def test_is_checked_has_selector(self):
    assert "selector" in self._params("is_checked")

  def test_check_element_has_selector(self):
    assert "selector" in self._params("check_element")

  def test_uncheck_element_has_selector(self):
    assert "selector" in self._params("uncheck_element")

  # Value & tab control
  def test_set_value_has_selector_and_value(self):
    params = self._params("set_value")
    assert "selector" in params
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


# ---------------------------------------------------------------------------
# Contract: SeleniumBaseBrowser is a BaseBrowser subclass (import check only)
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestSeleniumBaseBrowserSubclass:
  """SeleniumBaseBrowser must be a subclass of BaseBrowser."""

  def test_seleniumbase_browser_is_subclass_of_base_browser(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    assert issubclass(SeleniumBaseBrowser, BaseBrowser)

  def test_seleniumbase_browser_can_be_instantiated_with_default_config(self):
    """Instantiation must NOT launch the browser or import seleniumbase."""
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser()
    assert isinstance(browser, BaseBrowser)

  def test_seleniumbase_browser_accepts_config(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    cfg = BrowserConfig(headless=True, lang="fr")
    browser = SeleniumBaseBrowser(config=cfg)
    assert browser._config is cfg

  def test_seleniumbase_browser_has_no_sb_on_init(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser()
    assert browser._sb is None

  def test_seleniumbase_browser_has_executor_on_init(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser()
    assert browser._executor is not None
    # Cleanup
    browser._executor.shutdown(wait=False)


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
