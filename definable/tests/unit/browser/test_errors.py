"""Tests for browser error enrichment."""

from __future__ import annotations


from definable.browser.errors import normalize_timeout_ms, to_ai_friendly_error


class TestToAIFriendlyError:
  def test_strict_mode_violation(self):
    exc = Exception("strict mode violation: locator resolved to 5 elements")
    result = to_ai_friendly_error(exc, "button.submit")
    assert "5 elements" in result
    assert "browser_snapshot()" in result

  def test_strict_mode_no_count(self):
    exc = Exception("strict mode violation: something else")
    result = to_ai_friendly_error(exc, "btn")
    assert "multiple elements" in result

  def test_timeout_not_visible(self):
    exc = Exception("Timeout 30000ms exceeded waiting for locator to be visible")
    result = to_ai_friendly_error(exc, "e1")
    assert "not found or not visible" in result
    assert "browser_snapshot()" in result

  def test_pointer_events_intercepted(self):
    exc = Exception('Element "e1" intercepts pointer events')
    result = to_ai_friendly_error(exc, "e1")
    assert "covered by another element" in result
    assert "browser_remove_elements()" in result

  def test_not_receive_pointer_events(self):
    exc = Exception("Element did not receive pointer events")
    result = to_ai_friendly_error(exc, "e2")
    assert "covered" in result

  def test_context_destroyed(self):
    exc = Exception("Execution context was destroyed")
    result = to_ai_friendly_error(exc)
    assert "Page navigated" in result
    assert "browser_snapshot()" in result

  def test_detached(self):
    exc = Exception("Frame was detached")
    result = to_ai_friendly_error(exc)
    assert "Page navigated" in result

  def test_unknown_ref(self):
    exc = Exception('Unknown ref "e99".')
    result = to_ai_friendly_error(exc)
    assert "Unknown ref" in result
    assert "browser_snapshot()" in result

  def test_passthrough(self):
    exc = Exception("Some random error")
    result = to_ai_friendly_error(exc)
    assert result == "Some random error"


class TestNormalizeTimeoutMs:
  def test_within_range(self):
    assert normalize_timeout_ms(5000, 10000) == 5000

  def test_below_minimum(self):
    assert normalize_timeout_ms(100, 10000) == 500

  def test_above_maximum(self):
    assert normalize_timeout_ms(200000, 10000) == 120000

  def test_none_uses_fallback(self):
    assert normalize_timeout_ms(None, 15000) == 15000

  def test_fallback_clamped(self):
    assert normalize_timeout_ms(None, 300) == 500
