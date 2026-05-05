"""Tests for BrowserToolkit."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.browser.config import BrowserConfig
from definable.browser.toolkit import BrowserToolkit, _make_tools


class TestMakeTools:
  def test_tool_count(self):
    """Verify expected tool count."""
    mock_browser = MagicMock()
    tools = _make_tools(mock_browser)
    # Expected: 4+7+2+15+3+3+4+4+3+1+2+2+1+3+1 = 55
    assert len(tools) == 55, f"Expected 55 tools, got {len(tools)}"

  def test_tool_names_unique(self):
    mock_browser = MagicMock()
    tools = _make_tools(mock_browser)
    names = [t.name for t in tools]
    assert len(names) == len(set(names)), f"Duplicate tool names: {[n for n in names if names.count(n) > 1]}"

  def test_all_tools_have_browser_prefix(self):
    mock_browser = MagicMock()
    tools = _make_tools(mock_browser)
    for tool in tools:
      assert tool.name.startswith("browser_"), f"Tool {tool.name} missing browser_ prefix"

  def test_all_tools_have_docstrings(self):
    mock_browser = MagicMock()
    tools = _make_tools(mock_browser)
    for tool in tools:
      # Function.description is set from docstring
      assert tool.description or tool.entrypoint.__doc__, f"Tool {tool.name} has no docstring"

  def test_expected_tool_names(self):
    mock_browser = MagicMock()
    tools = _make_tools(mock_browser)
    names = {t.name for t in tools}

    # Navigation
    assert "browser_navigate" in names
    assert "browser_go_back" in names
    assert "browser_go_forward" in names
    assert "browser_refresh" in names

    # Perception
    assert "browser_snapshot" in names
    assert "browser_screenshot" in names

    # Interaction
    assert "browser_click" in names
    assert "browser_type" in names
    assert "browser_fill_form" in names
    assert "browser_set_input_files" in names

    # New tools (not in old toolkit)
    assert "browser_wait_for" in names
    assert "browser_get_console" in names
    assert "browser_get_errors" in names
    assert "browser_get_network" in names

    # Removed tools
    assert "browser_solve_captcha" not in names


class TestBrowserToolkit:
  def test_init_defaults(self):
    tk = BrowserToolkit()
    assert tk._config == BrowserConfig()
    assert tk._browser is None
    assert tk._owned is True
    assert tk._initialized is False
    assert tk._tools == []

  def test_init_with_config(self):
    config = BrowserConfig(headless=True)
    tk = BrowserToolkit(config=config)
    assert tk._config.headless is True

  def test_init_with_browser(self):
    mock_browser = MagicMock()
    tk = BrowserToolkit(browser=mock_browser)
    assert tk._browser is mock_browser
    assert tk._owned is False

  def test_repr_not_initialized(self):
    tk = BrowserToolkit()
    assert "not initialized" in repr(tk)

  def test_tools_empty_before_init(self):
    tk = BrowserToolkit()
    assert tk.tools == []

  @pytest.mark.asyncio
  async def test_initialize_with_injected_browser(self):
    mock_browser = MagicMock()
    mock_browser.start = AsyncMock()
    tk = BrowserToolkit(browser=mock_browser)
    await tk.initialize()
    assert tk._initialized is True
    assert len(tk.tools) == 55
    mock_browser.start.assert_not_called()  # not owned

  @pytest.mark.asyncio
  async def test_shutdown(self):
    mock_browser = MagicMock()
    mock_browser.start = AsyncMock()
    mock_browser.stop = AsyncMock()
    tk = BrowserToolkit(browser=mock_browser)
    await tk.initialize()
    assert len(tk.tools) == 55
    await tk.shutdown()
    assert tk._initialized is False
    assert tk.tools == []
    mock_browser.stop.assert_not_called()  # not owned

  @pytest.mark.asyncio
  async def test_initialize_idempotent(self):
    mock_browser = MagicMock()
    mock_browser.start = AsyncMock()
    tk = BrowserToolkit(browser=mock_browser)
    await tk.initialize()
    tools1 = tk.tools
    await tk.initialize()
    assert tk.tools is tools1


class TestBackwardCompat:
  def test_playwright_browser_import(self):
    from definable.browser import PlaywrightBrowser

    assert PlaywrightBrowser is not None

  def test_browser_toolkit_import(self):
    from definable.browser import BrowserToolkit

    assert BrowserToolkit is not None
