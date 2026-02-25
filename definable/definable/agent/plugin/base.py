"""Plugin base — Protocol and ABC for agent plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, FrozenSet

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class Plugin(ABC):
  """Base class for agent plugins.

  Plugins are the highest-level extensibility unit in Definable.
  They can register hooks on the pipeline, subscribe to events,
  add tools, modify configuration, and compose with other plugins.

  Lifecycle:
    1. ``on_load(agent)`` — called when the plugin is loaded (async).
    2. Hooks fire during ``arun()`` calls.
    3. ``on_unload(agent)`` — called when the plugin is removed (async).

  Example::

    class MyPlugin(Plugin):
      name = "my-plugin"

      async def on_load(self, agent: Agent) -> None:
        agent.pipeline.hook("after:invoke_loop", self._log_output)

      async def _log_output(self, state):
        print(f"Output: {state.output_content}")
        return state
  """

  # --- Identity ---

  @property
  @abstractmethod
  def name(self) -> str:
    """Unique plugin name (used for dependency resolution and dedup)."""
    ...

  @property
  def version(self) -> str:
    """Semantic version string."""
    return "0.1.0"

  @property
  def description(self) -> str:
    """Human-readable description."""
    return ""

  # --- Dependencies ---

  @property
  def requires(self) -> FrozenSet[str]:
    """Plugin names this plugin depends on (loaded first)."""
    return frozenset()

  @property
  def conflicts(self) -> FrozenSet[str]:
    """Plugin names that conflict with this one (cannot coexist)."""
    return frozenset()

  @property
  def modifies(self) -> FrozenSet[str]:
    """Pipeline phases or agent features this plugin modifies.

    Used for conflict detection — two plugins modifying the same
    phase raise a warning (not error) so the user is aware.
    """
    return frozenset()

  # --- Lifecycle ---

  @abstractmethod
  async def on_load(self, agent: "Agent") -> None:
    """Called when the plugin is loaded onto an agent.

    Register hooks, subscribe to events, add tools, etc.

    Args:
      agent: The agent this plugin is being loaded onto.
    """
    ...

  async def on_unload(self, agent: "Agent") -> None:
    """Called when the plugin is removed from an agent.

    Clean up hooks, event subscriptions, tools, etc.
    Default implementation is a no-op.

    Args:
      agent: The agent this plugin is being removed from.
    """

  # --- Introspection ---

  def to_dict(self) -> Dict[str, Any]:
    """Serialize plugin metadata to a dict."""
    return {
      "name": self.name,
      "version": self.version,
      "description": self.description,
      "requires": sorted(self.requires),
      "conflicts": sorted(self.conflicts),
      "modifies": sorted(self.modifies),
    }

  def __repr__(self) -> str:
    return f"<{type(self).__name__} name={self.name!r} version={self.version!r}>"
