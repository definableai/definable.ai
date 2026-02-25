"""Plugin system — extensibility architecture for agents."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.plugin.base import Plugin
  from definable.agent.plugin.registry import PluginRegistry

__all__ = [
  "Plugin",
  "PluginRegistry",
]


def __getattr__(name: str):
  if name == "Plugin":
    from definable.agent.plugin.base import Plugin

    return Plugin
  if name == "PluginRegistry":
    from definable.agent.plugin.registry import PluginRegistry

    return PluginRegistry
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
