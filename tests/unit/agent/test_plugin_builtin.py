"""Tests for built-in plugins — Logging, Metrics, Caching."""

from definable.agent.plugin.builtin.logging_plugin import LoggingPlugin
from definable.agent.plugin.builtin.metrics_plugin import MetricsPlugin, RunMetrics
from definable.agent.plugin.builtin.caching_plugin import CachingPlugin, CacheEntry


class TestLoggingPlugin:
  def test_name(self):
    p = LoggingPlugin()
    assert p.name == "logging"

  def test_description(self):
    p = LoggingPlugin()
    assert "logging" in p.description.lower()

  def test_modifies_wildcard(self):
    p = LoggingPlugin()
    assert "*" in p.modifies

  def test_verbose_default(self):
    p = LoggingPlugin()
    assert p._verbose is False

  def test_verbose_true(self):
    p = LoggingPlugin(verbose=True)
    assert p._verbose is True

  def test_custom_log_fn(self):
    calls = []

    def my_log(msg):
      calls.append(msg)

    p = LoggingPlugin(log_fn=my_log)
    assert p._log is my_log


class TestMetricsPlugin:
  def test_name(self):
    p = MetricsPlugin()
    assert p.name == "metrics"

  def test_description(self):
    p = MetricsPlugin()
    assert "metrics" in p.description.lower()

  def test_empty_history(self):
    p = MetricsPlugin()
    assert p.history == []
    assert p.last is None

  def test_average_duration_empty(self):
    p = MetricsPlugin()
    assert p.average_duration_ms == 0.0

  def test_average_duration(self):
    p = MetricsPlugin()
    p.history = [
      RunMetrics(run_id="r1", total_duration_ms=100.0),
      RunMetrics(run_id="r2", total_duration_ms=200.0),
    ]
    assert p.average_duration_ms == 150.0

  def test_max_history(self):
    p = MetricsPlugin(max_history=3)
    assert p._max_history == 3


class TestRunMetrics:
  def test_defaults(self):
    m = RunMetrics()
    assert m.run_id == ""
    assert m.phase_durations == {}
    assert m.total_duration_ms == 0.0
    assert m.tool_call_count == 0
    assert m.message_count == 0

  def test_fields(self):
    m = RunMetrics(
      run_id="test",
      phase_durations={"prepare": 10.0},
      total_duration_ms=50.0,
    )
    assert m.run_id == "test"
    assert m.phase_durations["prepare"] == 10.0


class TestCachingPlugin:
  def test_name(self):
    p = CachingPlugin()
    assert p.name == "caching"

  def test_description(self):
    p = CachingPlugin()
    assert "cache" in p.description.lower()

  def test_modifies(self):
    p = CachingPlugin()
    assert "invoke_loop" in p.modifies

  def test_initial_state(self):
    p = CachingPlugin()
    assert p.hit_count == 0
    assert p.miss_count == 0
    assert p.size == 0

  def test_max_size(self):
    p = CachingPlugin(max_size=50)
    assert p._max_size == 50

  def test_clear(self):
    p = CachingPlugin()
    p.hit_count = 5
    p.miss_count = 10
    p._cache["key"] = CacheEntry(output_content="test")
    p.clear()
    assert p.hit_count == 0
    assert p.miss_count == 0
    assert p.size == 0


class TestCacheEntry:
  def test_defaults(self):
    e = CacheEntry(output_content="hello")
    assert e.output_content == "hello"
    assert e.hit_count == 0

  def test_hit_increment(self):
    e = CacheEntry(output_content="hello")
    e.hit_count += 1
    assert e.hit_count == 1


class TestPluginExports:
  def test_import_from_builtin(self):
    from definable.agent.plugin.builtin import LoggingPlugin, MetricsPlugin, CachingPlugin

    assert LoggingPlugin is not None
    assert MetricsPlugin is not None
    assert CachingPlugin is not None

  def test_import_from_plugin_package(self):
    from definable.agent.plugin import Plugin, PluginRegistry

    assert Plugin is not None
    assert PluginRegistry is not None

  def test_import_from_agent(self):
    from definable.agent import Plugin, PluginRegistry

    assert Plugin is not None
    assert PluginRegistry is not None
