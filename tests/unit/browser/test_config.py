"""Tests for BrowserConfig."""

from __future__ import annotations

import pytest

from definable.browser.config import BrowserConfig


class TestDefaults:
  def test_default_values(self):
    c = BrowserConfig()
    assert c.cdp_url is None
    assert c.user_data_dir is None
    assert c.headless is False
    assert c.executable_path is None
    assert c.extra_args == ()
    assert c.no_sandbox is False
    assert c.proxy is None
    assert c.stealth is True
    assert c.timeout == 30.0
    assert c.viewport_width == 1280
    assert c.viewport_height == 720
    assert c.locale == "en-US"
    assert c.timezone is None
    assert c.user_agent is None
    assert c.max_console_messages == 500
    assert c.max_page_errors == 200
    assert c.max_network_requests == 500
    assert c.downloads_dir is None
    assert c.cdp_port == 9222
    assert c.slow_mo == 0.0

  def test_custom_values(self):
    c = BrowserConfig(
      cdp_url="ws://localhost:9222",
      headless=True,
      timeout=60.0,
      viewport_width=1920,
      viewport_height=1080,
      stealth=False,
      no_sandbox=True,
      proxy="proxy.example.com:8080",
      locale="fr-FR",
      timezone="Europe/Paris",
    )
    assert c.cdp_url == "ws://localhost:9222"
    assert c.headless is True
    assert c.timeout == 60.0
    assert c.viewport_width == 1920
    assert c.viewport_height == 1080
    assert c.stealth is False
    assert c.no_sandbox is True
    assert c.proxy == "proxy.example.com:8080"
    assert c.locale == "fr-FR"
    assert c.timezone == "Europe/Paris"

  def test_frozen(self):
    c = BrowserConfig()
    with pytest.raises(AttributeError):
      c.headless = True  # type: ignore[misc]

  def test_extra_args_tuple(self):
    c = BrowserConfig(extra_args=("--window-size=800,600", "--incognito"))
    assert c.extra_args == ("--window-size=800,600", "--incognito")


class TestConnectionModes:
  def test_cdp_url_priority(self):
    c = BrowserConfig(cdp_url="ws://localhost:9222")
    assert c.cdp_url == "ws://localhost:9222"

  def test_user_data_dir(self):
    c = BrowserConfig(user_data_dir="/tmp/profile")
    assert c.user_data_dir == "/tmp/profile"

  def test_fresh_mode_default(self):
    c = BrowserConfig()
    assert c.cdp_url is None
    assert c.user_data_dir is None


class TestRingBufferSizes:
  def test_custom_sizes(self):
    c = BrowserConfig(
      max_console_messages=100,
      max_page_errors=50,
      max_network_requests=200,
    )
    assert c.max_console_messages == 100
    assert c.max_page_errors == 50
    assert c.max_network_requests == 200
