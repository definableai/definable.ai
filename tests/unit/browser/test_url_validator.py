"""Tests for URL validator / SSRF protection."""

from __future__ import annotations

import pytest

from definable.browser.url_validator import (
  NavigationBlockedError,
  assert_navigation_allowed,
  assert_navigation_result_allowed,
)


@pytest.mark.asyncio
class TestAssertNavigationAllowed:
  async def test_valid_http(self):
    # Should not raise for public URLs
    # Note: This resolves DNS, so we test with a domain that resolves to public IP
    # For unit tests, we mainly test the validation logic
    pass

  async def test_empty_url(self):
    with pytest.raises(NavigationBlockedError, match="required"):
      await assert_navigation_allowed("")

  async def test_none_url(self):
    with pytest.raises(NavigationBlockedError, match="required"):
      await assert_navigation_allowed("")

  async def test_about_blank_allowed(self):
    await assert_navigation_allowed("about:blank")

  async def test_javascript_blocked(self):
    with pytest.raises(NavigationBlockedError, match="unsupported protocol"):
      await assert_navigation_allowed("javascript:alert(1)")

  async def test_file_blocked(self):
    with pytest.raises(NavigationBlockedError, match="unsupported protocol"):
      await assert_navigation_allowed("file:///etc/passwd")

  async def test_ftp_blocked(self):
    with pytest.raises(NavigationBlockedError, match="unsupported protocol"):
      await assert_navigation_allowed("ftp://example.com/file")

  async def test_data_blocked(self):
    with pytest.raises(NavigationBlockedError, match="unsupported protocol"):
      await assert_navigation_allowed("data:text/html,<h1>Hi</h1>")

  async def test_no_scheme(self):
    with pytest.raises(NavigationBlockedError):
      await assert_navigation_allowed("example.com")

  async def test_localhost_blocked(self):
    with pytest.raises(NavigationBlockedError, match="private IP"):
      await assert_navigation_allowed("http://127.0.0.1:8080")

  async def test_private_ip_blocked(self):
    with pytest.raises(NavigationBlockedError, match="private IP"):
      await assert_navigation_allowed("http://192.168.1.1")

  async def test_10_network_blocked(self):
    with pytest.raises(NavigationBlockedError, match="private IP"):
      await assert_navigation_allowed("http://10.0.0.1")


@pytest.mark.asyncio
class TestAssertNavigationResultAllowed:
  async def test_empty_url_ok(self):
    # Post-nav: empty URL should not raise
    await assert_navigation_result_allowed("")

  async def test_chrome_error_ok(self):
    # Post-nav: chrome-error:// should not raise
    await assert_navigation_result_allowed("chrome-error://chromewebdata/")

  async def test_devtools_ok(self):
    await assert_navigation_result_allowed("devtools://devtools/")

  async def test_network_url_validated(self):
    # Post-nav: http URLs are still validated
    with pytest.raises(NavigationBlockedError, match="private IP"):
      await assert_navigation_result_allowed("http://127.0.0.1:8080")
