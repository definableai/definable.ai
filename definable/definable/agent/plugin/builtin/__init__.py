"""Built-in plugins for common agent patterns."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.plugin.builtin.caching_plugin import CachingPlugin
  from definable.agent.plugin.builtin.logging_plugin import LoggingPlugin
  from definable.agent.plugin.builtin.metrics_plugin import MetricsPlugin

__all__ = [
  "LoggingPlugin",
  "MetricsPlugin",
  "CachingPlugin",
]


def __getattr__(name: str):
  if name == "LoggingPlugin":
    from definable.agent.plugin.builtin.logging_plugin import LoggingPlugin

    return LoggingPlugin
  if name == "MetricsPlugin":
    from definable.agent.plugin.builtin.metrics_plugin import MetricsPlugin

    return MetricsPlugin
  if name == "CachingPlugin":
    from definable.agent.plugin.builtin.caching_plugin import CachingPlugin

    return CachingPlugin
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
