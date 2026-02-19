"""
Unit tests for browser toolkit reliability fixes.

Tests cover:
- Phase 1: IIFE-wrapped JS execution (no bare return statements)
- Phase 3a: browser_press_key tool (no selector required)
- Phase 5: CSS selector error enrichment
- Phase 6: Page snapshot output format
- Toolkit tool count after adding new tools
- Phase 4: max_tool_rounds config

No browser launched, no seleniumbase import. Uses mock browser.

Migrated from tests_e2e/unit/test_browser_fixes.py -- all original tests preserved.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.browser.config import BrowserConfig


def _make_mock_browser():
  """Create a mock SeleniumBaseBrowser with all methods as AsyncMock."""
  mock = AsyncMock()
  mock._config = BrowserConfig()
  mock._sb = MagicMock()
  mock._assert_started = MagicMock()
  return mock


# ---------------------------------------------------------------------------
# Phase 1: IIFE-wrapped JS -- verify no bare `return` at top level
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIIFEWrappedJS:
  """All JS strings passed to execute_script must be wrapped in IIFEs
  so they work with SeleniumBase CDP's eval()-style execution."""

  def _get_real_browser(self):
    """Create a real SeleniumBaseBrowser with _sb mocked (no Chrome launch)."""
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser.__new__(SeleniumBaseBrowser)
    browser._config = BrowserConfig()
    browser._sb = MagicMock()
    browser._executor = MagicMock()
    browser._screenshot_dir = "/tmp/test"
    return browser

  def test_get_page_info_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.get_current_url.return_value = "https://example.com"
    browser._sb.get_title.return_value = "Example"
    browser._sb.execute_script.return_value = {
      "scrollY": 0,
      "totalH": 1000,
      "viewH": 800,
      "links": 5,
      "buttons": 2,
      "inputs": 1,
      "forms": 1,
    }

    # Call get_page_info and check what JS was passed to execute_script.
    loop = asyncio.new_event_loop()
    try:
      # Patch _run to execute the function directly
      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      result = loop.run_until_complete(browser.get_page_info())
    finally:
      loop.close()

    # Verify the JS passed to execute_script starts with IIFE
    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), f"get_page_info JS must be IIFE-wrapped, got: {js_call[:60]}"
    assert js_call.strip().endswith("})()") or js_call.strip().endswith("})()\n"), "get_page_info JS must end with })()"
    assert "URL:" in result

  def test_click_by_text_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.execute_script.return_value = "Clicked <button> with text: Submit"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      loop.run_until_complete(browser.click_by_text("Submit"))
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), f"click_by_text JS must be IIFE-wrapped, got: {js_call[:60]}"

  def test_remove_elements_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.execute_script.return_value = 3

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      result = loop.run_until_complete(browser.remove_elements(".cookie-banner"))
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), f"remove_elements JS must be IIFE-wrapped, got: {js_call[:60]}"
    assert "Removed 3" in result

  def test_highlight_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.execute_script.return_value = "highlighted"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      loop.run_until_complete(browser.highlight("#btn"))
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), f"highlight JS must be IIFE-wrapped, got: {js_call[:60]}"

  def test_get_storage_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.execute_script.return_value = "bar"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      result = loop.run_until_complete(browser.get_storage("foo"))
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), f"get_storage JS must be IIFE-wrapped, got: {js_call[:60]}"
    assert result == "bar"

  def test_drag_js_is_iife(self):
    browser = self._get_real_browser()
    browser._sb.execute_script.return_value = "ok"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run
      loop.run_until_complete(browser.drag("#src", "#dst"))
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert "(function()" in js_call, f"drag JS must be IIFE-wrapped, got: {js_call[:60]}"


# ---------------------------------------------------------------------------
# Phase 3a: browser_press_key -- no selector needed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPressKey:
  def test_press_key_method_exists(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    assert hasattr(SeleniumBaseBrowser, "press_key")

  def test_press_key_calls_body_selector(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser.__new__(SeleniumBaseBrowser)
    browser._config = BrowserConfig()
    browser._sb = MagicMock()
    browser._executor = MagicMock()
    browser._screenshot_dir = "/tmp/test"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run  # type: ignore[method-assign]
      result = loop.run_until_complete(browser.press_key("Enter"))
    finally:
      loop.close()

    browser._sb.press_keys.assert_called_once_with("body", "Enter")
    assert "Enter" in result

  def test_press_key_tool_in_toolkit(self):
    """browser_press_key tool must be present in toolkit tool list."""
    from definable.browser.toolkit import _make_tools

    mock_browser = AsyncMock()
    tools = _make_tools(mock_browser)
    tool_names = [t.name for t in tools]
    assert "browser_press_key" in tool_names


# ---------------------------------------------------------------------------
# Phase 5: CSS selector error enrichment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCSSErrorEnrichment:
  def _get_browser(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser.__new__(SeleniumBaseBrowser)
    browser._config = BrowserConfig()
    browser._sb = MagicMock()
    browser._executor = MagicMock()
    browser._screenshot_dir = "/tmp/test"
    return browser

  def test_has_text_pseudo_detected(self):
    browser = self._get_browser()
    result = browser._enrich_css_error("some error", 'button:has-text("Sign in")')
    assert "Invalid CSS selector" in result
    assert "browser_click_by_text()" in result

  def test_contains_pseudo_detected(self):
    browser = self._get_browser()
    result = browser._enrich_css_error("some error", 'div:contains("hello")')
    assert "Invalid CSS selector" in result
    assert "browser_click_by_text()" in result

  def test_text_pseudo_detected(self):
    browser = self._get_browser()
    result = browser._enrich_css_error("some error", 'a:text("Link")')
    assert "Invalid CSS selector" in result
    assert "browser_click_by_text()" in result

  def test_visible_pseudo_detected(self):
    browser = self._get_browser()
    result = browser._enrich_css_error("some error", "input:visible")
    assert "Invalid CSS selector" in result
    assert "browser_is_visible()" in result

  def test_valid_selector_passes_through(self):
    browser = self._get_browser()
    result = browser._enrich_css_error("Element not found", "#login-btn")
    assert result == "Error: Element not found"
    assert "Invalid CSS" not in result


# ---------------------------------------------------------------------------
# Phase 6: Page snapshot tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPageSnapshot:
  def test_snapshot_method_exists(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    assert hasattr(SeleniumBaseBrowser, "get_page_snapshot")

  def test_snapshot_tool_in_toolkit(self):
    from definable.browser.toolkit import _make_tools

    mock_browser = AsyncMock()
    tools = _make_tools(mock_browser)
    tool_names = [t.name for t in tools]
    assert "browser_snapshot" in tool_names

  def test_snapshot_js_is_iife(self):
    from definable.browser.seleniumbase_browser import SeleniumBaseBrowser

    browser = SeleniumBaseBrowser.__new__(SeleniumBaseBrowser)
    browser._config = BrowserConfig()
    browser._sb = MagicMock()
    browser._sb.execute_script.return_value = '[1] a "Home" href="/"'
    browser._executor = MagicMock()
    browser._screenshot_dir = "/tmp/test"

    loop = asyncio.new_event_loop()
    try:

      async def fake_run(fn, *args, **kwargs):
        return fn(*args, **kwargs)

      browser._run = fake_run  # type: ignore[method-assign]
      result = loop.run_until_complete(browser.get_page_snapshot())
    finally:
      loop.close()

    js_call = browser._sb.execute_script.call_args[0][0]
    assert js_call.strip().startswith("(function()"), "snapshot JS must be IIFE"
    assert result == '[1] a "Home" href="/"'


# ---------------------------------------------------------------------------
# Tool count
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolkitToolCount:
  def test_tool_count_is_50(self):
    """After adding browser_press_key and browser_snapshot, total should be 50."""
    from definable.browser.toolkit import _make_tools

    mock_browser = AsyncMock()
    tools = _make_tools(mock_browser)
    assert len(tools) == 50, f"Expected 50 tools, got {len(tools)}: {[t.name for t in tools]}"


# ---------------------------------------------------------------------------
# Phase 4: max_tool_rounds config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaxToolRoundsConfig:
  def test_default_max_tool_rounds_is_30(self):
    from definable.agent.config import AgentConfig

    assert AgentConfig().max_tool_rounds == 30

  def test_custom_max_tool_rounds(self):
    from definable.agent.config import AgentConfig

    config = AgentConfig(max_tool_rounds=50)
    assert config.max_tool_rounds == 50
