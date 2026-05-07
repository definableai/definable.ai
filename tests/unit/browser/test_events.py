"""Tests for browser action events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from definable.browser.config import BrowserConfig
from definable.browser.events import BrowserActionEvent
from definable.browser.playwright_browser import PlaywrightBrowser
from tests.unit.browser.conftest import MockPage


def _make_browser(page: MockPage | None = None) -> PlaywrightBrowser:
  """Create a PlaywrightBrowser with mocked internals."""
  browser = PlaywrightBrowser(BrowserConfig())
  mock_page = page or MockPage()
  browser._page = mock_page  # type: ignore[assignment]
  browser._browser = MagicMock()  # type: ignore[assignment]
  browser._context = mock_page.context  # type: ignore[assignment]
  return browser


# ---------------------------------------------------------------------------
# BrowserActionEvent dataclass
# ---------------------------------------------------------------------------


class TestBrowserActionEvent:
  def test_defaults(self):
    event = BrowserActionEvent()
    assert event.event == "BrowserAction"
    assert event.action == ""
    assert event.selector == ""
    assert event.value == ""
    assert event.result == ""
    assert event.url == ""
    assert event.error == ""
    assert event.timestamp > 0

  def test_with_fields(self):
    event = BrowserActionEvent(action="click", selector="e1", result="Clicked: e1", url="https://example.com")
    assert event.action == "click"
    assert event.selector == "e1"
    assert event.result == "Clicked: e1"
    assert event.url == "https://example.com"

  def test_to_dict(self):
    event = BrowserActionEvent(action="navigate", value="https://example.com", result="OK")
    d = event.to_dict()
    assert d["event"] == "BrowserAction"
    assert d["action"] == "navigate"
    assert d["value"] == "https://example.com"
    assert d["result"] == "OK"

  def test_inherits_base(self):
    from definable.run.base import BaseRunOutputEvent

    assert issubclass(BrowserActionEvent, BaseRunOutputEvent)


# ---------------------------------------------------------------------------
# _emit helper
# ---------------------------------------------------------------------------


class TestEmitHelper:
  @pytest.mark.asyncio
  async def test_emit_calls_sync_callback(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser._emit("click", selector="e1", result="Clicked: e1")
    assert len(events) == 1
    assert events[0].action == "click"
    assert events[0].selector == "e1"
    assert events[0].url == "https://example.com"

  @pytest.mark.asyncio
  async def test_emit_calls_async_callback(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []

    async def handler(e: BrowserActionEvent) -> None:
      events.append(e)

    browser.on_action = handler

    await browser._emit("navigate", value="https://test.com")
    assert len(events) == 1
    assert events[0].action == "navigate"

  @pytest.mark.asyncio
  async def test_emit_noop_when_no_callback(self):
    browser = _make_browser()
    assert browser.on_action is None
    # Should not raise
    await browser._emit("click", selector="e1")

  @pytest.mark.asyncio
  async def test_emit_captures_page_url(self):
    page = MockPage("https://my-app.com/dashboard")
    browser = _make_browser(page)
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser._emit("snapshot")
    assert events[0].url == "https://my-app.com/dashboard"

  @pytest.mark.asyncio
  async def test_emit_swallows_callback_errors(self):
    browser = _make_browser()

    def bad_handler(e: BrowserActionEvent) -> None:
      raise ValueError("handler broke")

    browser.on_action = bad_handler
    # Should not raise
    await browser._emit("click", selector="e1")

  @pytest.mark.asyncio
  async def test_emit_url_empty_when_no_page(self):
    browser = PlaywrightBrowser(BrowserConfig())
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser._emit("navigate", value="https://example.com")
    assert events[0].url == ""


# ---------------------------------------------------------------------------
# Event emission from browser actions
# ---------------------------------------------------------------------------


class TestNavigationEvents:
  @pytest.mark.asyncio
  async def test_navigate_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.navigate("https://example.com")
    assert len(events) == 1
    assert events[0].action == "navigate"
    assert events[0].value == "https://example.com"
    assert "Navigated to" in events[0].result
    assert events[0].error == ""

  @pytest.mark.asyncio
  async def test_go_back_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.go_back()
    assert len(events) == 1
    assert events[0].action == "go_back"

  @pytest.mark.asyncio
  async def test_go_forward_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.go_forward()
    assert events[0].action == "go_forward"

  @pytest.mark.asyncio
  async def test_refresh_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.refresh()
    assert events[0].action == "refresh"


class TestInteractionEvents:
  @pytest.mark.asyncio
  async def test_click_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    # Use CSS selector (not ref) to avoid ref lookup failure
    await browser.click("button.submit")
    assert len(events) == 1
    assert events[0].action == "click"
    assert events[0].selector == "button.submit"
    assert "Clicked" in events[0].result

  @pytest.mark.asyncio
  async def test_click_ref_emits_event(self):
    browser = _make_browser()
    from definable.browser.element_refs import RoleRef

    browser._refs.store({"e1": RoleRef(role="button", name="Submit")})
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.click("e1")
    assert events[0].action == "click"
    assert events[0].selector == "e1"

  @pytest.mark.asyncio
  async def test_click_error_emits_event(self):
    page = MockPage()

    # Make locator.click raise
    async def bad_click(**kwargs):
      raise TimeoutError("Timeout 30000ms exceeded")

    page._locator.click = bad_click  # type: ignore[assignment]
    browser = _make_browser(page)
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    result = await browser.click("button.gone")
    assert len(events) == 1
    assert events[0].action == "click"
    assert events[0].error != ""
    assert "Error" in result

  @pytest.mark.asyncio
  async def test_click_if_visible_emits_when_clicked(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.click_if_visible("button.submit")
    assert len(events) == 1
    assert events[0].action == "click"

  @pytest.mark.asyncio
  async def test_click_if_visible_no_event_when_not_visible(self):
    page = MockPage()
    page._locator._visible = False
    browser = _make_browser(page)
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.click_if_visible("button.submit")
    assert len(events) == 0

  @pytest.mark.asyncio
  async def test_click_by_text_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.click_by_text("Sign In")
    assert events[0].action == "click"
    assert events[0].value == "Sign In"

  @pytest.mark.asyncio
  async def test_type_text_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.type_text("input.email", "test@example.com")
    assert events[0].action == "type"
    assert events[0].selector == "input.email"
    assert events[0].value == "test@example.com"

  @pytest.mark.asyncio
  async def test_type_slowly_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.type_slowly("input.search", "hello")
    assert events[0].action == "type"
    assert events[0].value == "hello"

  @pytest.mark.asyncio
  async def test_press_key_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.press_key("Enter")
    assert events[0].action == "key_press"
    assert events[0].value == "Enter"

  @pytest.mark.asyncio
  async def test_press_keys_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.press_keys("input.search", "Tab")
    assert events[0].action == "key_press"
    assert events[0].selector == "input.search"
    assert events[0].value == "Tab"

  @pytest.mark.asyncio
  async def test_hover_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.hover("button.menu")
    assert events[0].action == "hover"
    assert events[0].selector == "button.menu"

  @pytest.mark.asyncio
  async def test_drag_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.drag("div.source", "div.target")
    assert events[0].action == "drag"
    assert events[0].selector == "div.source"
    assert events[0].value == "div.target"

  @pytest.mark.asyncio
  async def test_select_option_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.select_option("select.country", "United States")
    assert events[0].action == "select"
    assert events[0].value == "United States"

  @pytest.mark.asyncio
  async def test_check_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.check_element("input.agree")
    assert events[0].action == "check"

  @pytest.mark.asyncio
  async def test_uncheck_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.uncheck_element("input.agree")
    assert events[0].action == "uncheck"

  @pytest.mark.asyncio
  async def test_clear_input_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.clear_input("input.search")
    assert events[0].action == "clear"

  @pytest.mark.asyncio
  async def test_set_value_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.set_value("input.slider", "75")
    assert events[0].action == "set_value"
    assert events[0].value == "75"

  @pytest.mark.asyncio
  async def test_set_input_files_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.set_input_files("input.upload", ["/tmp/file.pdf"])
    assert events[0].action == "set_files"

  @pytest.mark.asyncio
  async def test_fill_form_emits_event(self):
    browser = _make_browser()
    # Store refs for form fill
    from definable.browser.element_refs import RoleRef

    browser._refs.store({
      "e1": RoleRef(role="textbox", name="Name"),
      "e2": RoleRef(role="checkbox", name="Agree"),
    })
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.fill_form([
      {"ref": "e1", "type": "text", "value": "John"},
      {"ref": "e2", "type": "checkbox", "value": True},
    ])
    assert events[0].action == "fill_form"
    assert "2/2" in events[0].value


class TestScrollEvents:
  @pytest.mark.asyncio
  async def test_scroll_down_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.scroll_down(2)
    assert events[0].action == "scroll_down"
    assert events[0].value == "2"

  @pytest.mark.asyncio
  async def test_scroll_up_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.scroll_up(1)
    assert events[0].action == "scroll_up"
    assert events[0].value == "1"

  @pytest.mark.asyncio
  async def test_scroll_to_element_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.scroll_to_element("div.footer")
    assert events[0].action == "scroll_to"
    assert events[0].selector == "div.footer"


class TestPerceptionEvents:
  @pytest.mark.asyncio
  async def test_snapshot_emits_event(self):
    page = MockPage()

    async def mock_aria_snapshot() -> str:
      return '- button "Submit"\n- textbox "Email"'

    page._locator.aria_snapshot = mock_aria_snapshot  # type: ignore[attr-defined]
    browser = _make_browser(page)
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.snapshot()
    assert len(events) == 1
    assert events[0].action == "snapshot"
    assert "refs" in events[0].result

  @pytest.mark.asyncio
  async def test_screenshot_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.screenshot("test_shot")
    assert events[0].action == "screenshot"
    assert "test_shot.png" in events[0].value


class TestDOMMutationEvents:
  @pytest.mark.asyncio
  async def test_highlight_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.highlight("button.submit")
    assert events[0].action == "highlight"

  @pytest.mark.asyncio
  async def test_remove_elements_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.remove_elements(".cookie-banner")
    assert events[0].action == "remove_elements"
    assert events[0].selector == ".cookie-banner"


class TestTabEvents:
  @pytest.mark.asyncio
  async def test_open_tab_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.open_tab()
    assert events[0].action == "tab_open"

  @pytest.mark.asyncio
  async def test_close_tab_emits_event(self):
    # Setup: two tabs so close doesn't null out _page
    page1 = MockPage("https://page1.com")
    page2 = MockPage("https://page2.com")
    browser = _make_browser(page1)
    browser._context._pages.append(page2)  # type: ignore[union-attr]
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.close_tab()
    assert events[0].action == "tab_close"

  @pytest.mark.asyncio
  async def test_switch_to_tab_emits_event(self):
    page1 = MockPage("https://page1.com")
    page2 = MockPage("https://page2.com")
    browser = _make_browser(page1)
    browser._context._pages.append(page2)  # type: ignore[union-attr]
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.switch_to_tab(1)
    assert events[0].action == "tab_switch"
    assert events[0].value == "1"


class TestJSEvents:
  @pytest.mark.asyncio
  async def test_execute_js_emits_event(self):
    browser = _make_browser()
    events: list[BrowserActionEvent] = []
    browser.on_action = lambda e: events.append(e)

    await browser.execute_js("document.title")
    assert events[0].action == "execute_js"
    assert "document.title" in events[0].value


# ---------------------------------------------------------------------------
# Toolkit wiring
# ---------------------------------------------------------------------------


class TestToolkitWiring:
  @pytest.mark.asyncio
  async def test_on_action_passed_to_browser(self):
    """BrowserToolkit passes on_action to the browser on initialize."""
    from definable.browser.toolkit import BrowserToolkit

    events: list[BrowserActionEvent] = []

    def handler(e: BrowserActionEvent) -> None:
      events.append(e)

    # Create toolkit with injected browser (skip real Playwright)
    browser = _make_browser()
    toolkit = BrowserToolkit(browser=browser, on_action=handler)
    await toolkit.initialize()

    assert browser.on_action is handler

    await toolkit.shutdown()

  @pytest.mark.asyncio
  async def test_toolkit_without_on_action(self):
    """BrowserToolkit works without on_action (default)."""
    from definable.browser.toolkit import BrowserToolkit

    browser = _make_browser()
    toolkit = BrowserToolkit(browser=browser)
    await toolkit.initialize()

    assert browser.on_action is None
    await toolkit.shutdown()


# ---------------------------------------------------------------------------
# Event re-exports
# ---------------------------------------------------------------------------


class TestExports:
  def test_browser_module_exports(self):
    from definable.browser import BrowserActionEvent

    assert BrowserActionEvent is not None

  def test_agent_events_exports(self):
    from definable.agent.events import BrowserActionEvent

    assert BrowserActionEvent is not None
