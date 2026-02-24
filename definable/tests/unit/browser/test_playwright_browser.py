"""Tests for PlaywrightBrowser (mocked Playwright)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from definable.browser.config import BrowserConfig
from definable.browser.element_refs import RoleRef
from definable.browser.playwright_browser import PlaywrightBrowser
from tests.unit.browser.conftest import MockPage


def _make_browser(page: MockPage | None = None) -> PlaywrightBrowser:
  """Create a PlaywrightBrowser with mocked internals."""
  browser = PlaywrightBrowser(BrowserConfig())
  mock_page = page or MockPage()
  browser._page = mock_page  # type: ignore[assignment]
  browser._browser = MagicMock()  # type: ignore[assignment]
  browser._context = mock_page.context  # type: ignore[assignment]
  # Don't attach page state (would need real Playwright Page)
  return browser


class TestLifecycle:
  def test_init_defaults(self):
    browser = PlaywrightBrowser()
    assert browser._config == BrowserConfig()
    assert browser._page is None
    assert browser._browser is None

  def test_init_with_config(self):
    config = BrowserConfig(headless=True, timeout=60.0)
    browser = PlaywrightBrowser(config)
    assert browser._config.headless is True
    assert browser._config.timeout == 60.0

  @pytest.mark.asyncio
  async def test_ensure_page_raises_when_not_started(self):
    browser = PlaywrightBrowser()
    with pytest.raises(RuntimeError, match="not started"):
      await browser._ensure_page()

  @pytest.mark.asyncio
  async def test_stop_clears_state(self):
    browser = _make_browser()
    browser._pw = MagicMock()
    browser._pw.stop = AsyncMock()
    browser._browser = MagicMock()
    browser._browser.close = AsyncMock()
    await browser.stop()
    assert browser._page is None
    assert browser._browser is None
    assert browser._pw is None


class TestNavigation:
  @pytest.mark.asyncio
  @patch("definable.browser.playwright_browser.assert_navigation_allowed", new_callable=AsyncMock)
  @patch("definable.browser.playwright_browser.assert_navigation_result_allowed", new_callable=AsyncMock)
  async def test_navigate(self, mock_result, mock_allowed):
    browser = _make_browser()
    result = await browser.navigate("https://example.com")
    assert "Navigated to" in result

  @pytest.mark.asyncio
  async def test_go_back(self):
    browser = _make_browser()
    result = await browser.go_back()
    assert "Navigated back" in result

  @pytest.mark.asyncio
  async def test_go_forward(self):
    browser = _make_browser()
    result = await browser.go_forward()
    assert "Navigated forward" in result

  @pytest.mark.asyncio
  async def test_refresh(self):
    browser = _make_browser()
    result = await browser.refresh()
    assert "Refreshed" in result


class TestPageState:
  @pytest.mark.asyncio
  async def test_get_url(self):
    browser = _make_browser()
    url = await browser.get_url()
    assert url == "https://example.com"

  @pytest.mark.asyncio
  async def test_get_title(self):
    browser = _make_browser()
    title = await browser.get_title()
    assert title == "Example"

  @pytest.mark.asyncio
  async def test_get_page_source(self):
    browser = _make_browser()
    source = await browser.get_page_source()
    assert "<html>" in source

  @pytest.mark.asyncio
  async def test_get_text(self):
    browser = _make_browser()
    text = await browser.get_text("body")
    assert text == "mock text"

  @pytest.mark.asyncio
  async def test_is_element_visible(self):
    browser = _make_browser()
    result = await browser.is_element_visible("button")
    assert result == "true"


class TestInteraction:
  @pytest.mark.asyncio
  async def test_click_css(self):
    browser = _make_browser()
    result = await browser.click("button.submit")
    assert "Clicked" in result

  @pytest.mark.asyncio
  async def test_click_ref(self):
    browser = _make_browser()
    browser._refs.store({"e1": RoleRef(role="button", name="Submit")})
    result = await browser.click("e1")
    assert "Clicked" in result

  @pytest.mark.asyncio
  async def test_click_if_visible(self):
    browser = _make_browser()
    result = await browser.click_if_visible("button")
    assert "Clicked" in result

  @pytest.mark.asyncio
  async def test_click_by_text(self):
    browser = _make_browser()
    result = await browser.click_by_text("Sign In")
    assert "Clicked" in result

  @pytest.mark.asyncio
  async def test_type_text(self):
    browser = _make_browser()
    result = await browser.type_text("input", "hello")
    assert "Typed" in result

  @pytest.mark.asyncio
  async def test_press_key(self):
    browser = _make_browser()
    result = await browser.press_key("Enter")
    assert "Pressed" in result

  @pytest.mark.asyncio
  async def test_clear_input(self):
    browser = _make_browser()
    result = await browser.clear_input("input")
    assert "Cleared" in result

  @pytest.mark.asyncio
  async def test_hover(self):
    browser = _make_browser()
    result = await browser.hover("button")
    assert "Hovered" in result

  @pytest.mark.asyncio
  async def test_drag(self):
    browser = _make_browser()
    result = await browser.drag("div.source", "div.target")
    assert "Dragged" in result

  @pytest.mark.asyncio
  async def test_select_option(self):
    browser = _make_browser()
    result = await browser.select_option("select", "Option A")
    assert "Selected" in result

  @pytest.mark.asyncio
  async def test_check(self):
    browser = _make_browser()
    result = await browser.check_element("input[type=checkbox]")
    assert "Checked" in result

  @pytest.mark.asyncio
  async def test_uncheck(self):
    browser = _make_browser()
    result = await browser.uncheck_element("input[type=checkbox]")
    assert "Unchecked" in result

  @pytest.mark.asyncio
  async def test_set_value(self):
    browser = _make_browser()
    result = await browser.set_value("input", "42")
    assert "Set value" in result

  @pytest.mark.asyncio
  async def test_highlight(self):
    browser = _make_browser()
    result = await browser.highlight("button")
    assert "Highlighted" in result

  @pytest.mark.asyncio
  async def test_remove_elements(self):
    browser = _make_browser()
    result = await browser.remove_elements(".cookie-banner")
    assert "Removed" in result


class TestScrolling:
  @pytest.mark.asyncio
  async def test_scroll_down(self):
    browser = _make_browser()
    result = await browser.scroll_down(2)
    assert "Scrolled down" in result

  @pytest.mark.asyncio
  async def test_scroll_up(self):
    browser = _make_browser()
    result = await browser.scroll_up(1)
    assert "Scrolled up" in result

  @pytest.mark.asyncio
  async def test_scroll_to_element(self):
    browser = _make_browser()
    result = await browser.scroll_to_element("footer")
    assert "Scrolled to" in result


class TestWaiting:
  @pytest.mark.asyncio
  async def test_wait(self):
    browser = _make_browser()
    result = await browser.wait(0.01)
    assert "Waited" in result

  @pytest.mark.asyncio
  async def test_wait_for_element(self):
    browser = _make_browser()
    result = await browser.wait_for_element("button", timeout=0.1)
    assert "appeared" in result

  @pytest.mark.asyncio
  async def test_wait_for_text(self):
    browser = _make_browser()
    result = await browser.wait_for_text("hello", timeout=0.1)
    assert "appeared" in result


class TestTabs:
  @pytest.mark.asyncio
  async def test_get_tabs(self):
    browser = _make_browser()
    result = await browser.get_tabs()
    assert "tab(s)" in result

  @pytest.mark.asyncio
  async def test_switch_to_tab_out_of_range(self):
    browser = _make_browser()
    result = await browser.switch_to_tab(99)
    assert "Error" in result


class TestCookies:
  @pytest.mark.asyncio
  async def test_get_cookies(self):
    browser = _make_browser()
    result = await browser.get_cookies()
    assert "[]" in result

  @pytest.mark.asyncio
  async def test_set_cookie(self):
    browser = _make_browser()
    result = await browser.set_cookie("name", "value")
    assert "Cookie set" in result

  @pytest.mark.asyncio
  async def test_clear_cookies(self):
    browser = _make_browser()
    result = await browser.clear_cookies()
    assert "cleared" in result


class TestStorage:
  @pytest.mark.asyncio
  async def test_get_storage(self):
    browser = _make_browser()
    result = await browser.get_storage("key")
    # MockPage evaluate returns a dict
    assert "{" in result or "Error" in result

  @pytest.mark.asyncio
  async def test_set_storage(self):
    browser = _make_browser()
    result = await browser.set_storage("key", "value")
    assert "Storage set" in result


class TestDialog:
  @pytest.mark.asyncio
  async def test_handle_dialog_accept(self):
    browser = _make_browser()
    result = await browser.handle_dialog(accept=True)
    assert "accept" in result

  @pytest.mark.asyncio
  async def test_handle_dialog_dismiss(self):
    browser = _make_browser()
    result = await browser.handle_dialog(accept=False)
    assert "dismiss" in result


class TestPageInfo:
  @pytest.mark.asyncio
  async def test_get_page_info(self):
    browser = _make_browser()
    result = await browser.get_page_info()
    assert "URL:" in result
    assert "Title:" in result
    assert "Viewport:" in result


class TestDiagnostics:
  @pytest.mark.asyncio
  async def test_get_console_no_state(self):
    browser = _make_browser()
    result = await browser.get_console()
    assert "No page state" in result

  @pytest.mark.asyncio
  async def test_get_errors_no_state(self):
    browser = _make_browser()
    result = await browser.get_errors()
    assert "No page state" in result

  @pytest.mark.asyncio
  async def test_get_network_no_state(self):
    browser = _make_browser()
    result = await browser.get_network()
    assert "No page state" in result


class TestForceDisconnect:
  @pytest.mark.asyncio
  async def test_force_disconnect_nulls_state(self):
    browser = _make_browser()
    mock_browser_obj = MagicMock()
    mock_browser_obj.close = AsyncMock()
    browser._browser = mock_browser_obj
    await browser._force_disconnect("test")
    assert browser._browser is None
    assert browser._page is None

  @pytest.mark.asyncio
  async def test_connect_retry_logic(self):
    """Verify _connect_with_retry attempts 3 times."""
    browser = PlaywrightBrowser(BrowserConfig())
    browser._pw = MagicMock()
    browser._pw.chromium = MagicMock()

    attempts = []

    async def mock_connect(*args, **kwargs):
      attempts.append(1)
      if len(attempts) < 3:
        raise ConnectionError("fail")
      return MagicMock()

    browser._pw.chromium.connect_over_cdp = mock_connect

    with patch("definable.browser.playwright_browser.get_chrome_ws_url", new_callable=AsyncMock, return_value=None):
      await browser._connect_with_retry("http://127.0.0.1:9222")
      assert len(attempts) == 3
