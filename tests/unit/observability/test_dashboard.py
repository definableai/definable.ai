"""Tests for the dashboard HTML generator — structure, theme, content validation."""

from definable.agent.observability.config import ObservabilityConfig
from definable.agent.observability.dashboard import get_dashboard_html


class TestDashboardHTML:
  """Tests for get_dashboard_html()."""

  def test_returns_html_string(self):
    """Should return a non-empty HTML string."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert isinstance(html, str)
    assert len(html) > 0

  def test_contains_doctype(self):
    """HTML should start with DOCTYPE declaration."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert html.strip().startswith("<!DOCTYPE html>")

  def test_dark_theme_default(self):
    """Default theme should be dark."""
    config = ObservabilityConfig(enabled=True, theme="dark")
    html = get_dashboard_html(config)
    assert 'data-theme="dark"' in html

  def test_light_theme(self):
    """Should respect light theme setting."""
    config = ObservabilityConfig(enabled=True, theme="light")
    html = get_dashboard_html(config)
    assert 'data-theme="light"' in html

  def test_contains_alpine_js(self):
    """Should include Alpine.js CDN script."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "alpinejs" in html

  def test_contains_css_variables(self):
    """Should include CSS custom properties for theming."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "--bg:" in html
    assert "--accent:" in html
    assert "--surface:" in html

  def test_contains_sidebar_nav(self):
    """Should include sidebar navigation items."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "overview" in html
    assert "live_events" in html
    assert "sessions" in html
    assert "compare" in html
    assert "tools" in html
    assert "models" in html

  def test_contains_app_init(self):
    """Should include Alpine.js app initialization."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "function obsApp()" in html
    assert "async init()" in html

  def test_contains_sse_connection(self):
    """Should include EventSource for SSE."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "EventSource" in html
    assert "/obs/api/events" in html

  def test_contains_stat_cards(self):
    """Overview page should have stat cards for key metrics."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "total_runs" in html
    assert "success_rate" in html
    assert "total_tokens" in html
    assert "total_cost" in html

  def test_contains_branding(self):
    """Should include Definable observability branding."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "DEFINABLE_OBS" in html
    assert "Definable Observability" in html

  def test_contains_responsive_css(self):
    """Should include responsive breakpoints."""
    config = ObservabilityConfig(enabled=True)
    html = get_dashboard_html(config)
    assert "@media" in html
    assert "768px" in html
