"""PluginRegistry — manages plugin lifecycle, ordering, and conflict detection."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set

from definable.agent.plugin.base import Plugin
from definable.utils.log import log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class PluginRegistry:
  """Registry that manages plugin loading, ordering, and lifecycle.

  Handles:
    - Topological sort for dependency ordering
    - Conflict detection (explicit conflicts + shared modifies)
    - Enable/disable without unloading
    - Clean teardown via unload_all

  Example::

    registry = PluginRegistry()
    registry.add(LoggingPlugin())
    registry.add(MetricsPlugin())
    await registry.load_all(agent)
  """

  def __init__(self) -> None:
    self._plugins: Dict[str, Plugin] = {}  # name → plugin
    self._loaded: Set[str] = set()  # names of loaded plugins
    self._disabled: Set[str] = set()  # names of disabled plugins
    self._load_order: List[str] = []  # topologically sorted names

  # --- Registration ---

  def add(self, plugin: Plugin) -> "PluginRegistry":
    """Register a plugin (does not load it yet).

    Args:
      plugin: Plugin instance to register.

    Returns:
      Self for chaining.

    Raises:
      ValueError: If a plugin with the same name is already registered.
    """
    if plugin.name in self._plugins:
      raise ValueError(f"Plugin '{plugin.name}' is already registered.")
    self._plugins[plugin.name] = plugin
    self._load_order = []  # invalidate cached order
    return self

  def remove(self, name: str) -> Optional[Plugin]:
    """Remove a plugin by name (must be unloaded first).

    Args:
      name: Plugin name to remove.

    Returns:
      The removed Plugin, or None if not found.

    Raises:
      RuntimeError: If the plugin is still loaded.
    """
    if name in self._loaded:
      raise RuntimeError(f"Plugin '{name}' is still loaded. Call unload() first.")
    plugin = self._plugins.pop(name, None)
    if plugin is not None:
      self._disabled.discard(name)
      self._load_order = []
    return plugin

  # --- Loading ---

  async def load_all(self, agent: "Agent") -> None:
    """Load all registered (non-disabled) plugins in dependency order.

    Args:
      agent: The agent to load plugins onto.

    Raises:
      ValueError: If dependencies are missing or conflicts exist.
    """
    self._validate()
    order = self._topological_sort()
    self._load_order = order

    for name in order:
      if name in self._disabled:
        continue
      if name in self._loaded:
        continue
      plugin = self._plugins[name]
      await plugin.on_load(agent)
      self._loaded.add(name)
      log_info(f"Plugin loaded: {name} v{plugin.version}")

  async def unload_all(self, agent: "Agent") -> None:
    """Unload all loaded plugins in reverse order.

    Args:
      agent: The agent to unload plugins from.
    """
    for name in reversed(self._load_order):
      if name not in self._loaded:
        continue
      plugin = self._plugins[name]
      await plugin.on_unload(agent)
      self._loaded.discard(name)
      log_info(f"Plugin unloaded: {name}")

  async def load_one(self, name: str, agent: "Agent") -> None:
    """Load a single plugin by name.

    Args:
      name: Plugin name.
      agent: The agent to load onto.

    Raises:
      KeyError: If plugin not found.
      ValueError: If dependencies not satisfied.
    """
    if name not in self._plugins:
      raise KeyError(f"Plugin '{name}' not registered.")
    if name in self._loaded:
      return

    plugin = self._plugins[name]
    # Check deps are loaded
    missing = plugin.requires - self._loaded
    if missing:
      raise ValueError(f"Plugin '{name}' requires {missing} which are not loaded.")

    await plugin.on_load(agent)
    self._loaded.add(name)
    log_info(f"Plugin loaded: {name} v{plugin.version}")

  async def unload_one(self, name: str, agent: "Agent") -> None:
    """Unload a single plugin by name.

    Args:
      name: Plugin name.
      agent: The agent to unload from.

    Raises:
      KeyError: If plugin not found.
      RuntimeError: If other loaded plugins depend on this one.
    """
    if name not in self._plugins:
      raise KeyError(f"Plugin '{name}' not registered.")
    if name not in self._loaded:
      return

    # Check no loaded plugin depends on this one
    dependents = [n for n, p in self._plugins.items() if n in self._loaded and name in p.requires]
    if dependents:
      raise RuntimeError(f"Cannot unload '{name}': plugins {dependents} depend on it.")

    plugin = self._plugins[name]
    await plugin.on_unload(agent)
    self._loaded.discard(name)
    log_info(f"Plugin unloaded: {name}")

  # --- Enable/Disable ---

  def disable(self, name: str) -> None:
    """Mark a plugin as disabled (skipped during load_all).

    Args:
      name: Plugin name to disable.
    """
    if name not in self._plugins:
      raise KeyError(f"Plugin '{name}' not registered.")
    self._disabled.add(name)

  def enable(self, name: str) -> None:
    """Re-enable a disabled plugin.

    Args:
      name: Plugin name to enable.
    """
    self._disabled.discard(name)

  # --- Validation ---

  def _validate(self) -> None:
    """Validate all plugins for missing deps and conflicts.

    Raises:
      ValueError: On missing dependencies or explicit conflicts.
    """
    names = set(self._plugins.keys())

    for name, plugin in self._plugins.items():
      if name in self._disabled:
        continue

      # Missing dependencies
      missing = plugin.requires - names
      if missing:
        raise ValueError(f"Plugin '{name}' requires {missing} which are not registered.")

      # Explicit conflicts
      active_conflicts = plugin.conflicts & (names - self._disabled - {name})
      if active_conflicts:
        raise ValueError(f"Plugin '{name}' conflicts with active plugins: {active_conflicts}")

    # Warn on shared modifies (not an error — just a heads-up)
    modifies_map: Dict[str, List[str]] = defaultdict(list)
    for name, plugin in self._plugins.items():
      if name in self._disabled:
        continue
      for mod in plugin.modifies:
        modifies_map[mod].append(name)

    for target, plugins in modifies_map.items():
      if len(plugins) > 1:
        log_warning(f"Multiple plugins modify '{target}': {plugins}. Check for unintended interactions.")

  def _topological_sort(self) -> List[str]:
    """Sort plugins by dependency order (Kahn's algorithm).

    Returns:
      Ordered list of plugin names (dependencies first).

    Raises:
      ValueError: If a dependency cycle is detected.
    """
    active = {n for n in self._plugins if n not in self._disabled}

    # Build adjacency: edges from dependency → dependent
    in_degree: Dict[str, int] = {n: 0 for n in active}
    adj: Dict[str, List[str]] = {n: [] for n in active}

    for name in active:
      for dep in self._plugins[name].requires:
        if dep in active:
          adj[dep].append(name)
          in_degree[name] += 1

    # Kahn's algorithm
    queue = sorted(n for n in active if in_degree[n] == 0)
    result: List[str] = []

    while queue:
      node = queue.pop(0)
      result.append(node)
      for neighbor in sorted(adj[node]):
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
          queue.append(neighbor)

    if len(result) != len(active):
      # Cycle detected
      remaining = active - set(result)
      raise ValueError(f"Dependency cycle detected among plugins: {remaining}")

    return result

  # --- Introspection ---

  @property
  def plugin_names(self) -> List[str]:
    """All registered plugin names."""
    return list(self._plugins.keys())

  @property
  def loaded_names(self) -> List[str]:
    """Names of currently loaded plugins."""
    return [n for n in self._load_order if n in self._loaded]

  @property
  def disabled_names(self) -> List[str]:
    """Names of disabled plugins."""
    return list(self._disabled)

  def get(self, name: str) -> Optional[Plugin]:
    """Get a plugin by name."""
    return self._plugins.get(name)

  def is_loaded(self, name: str) -> bool:
    """Check if a plugin is loaded."""
    return name in self._loaded

  def __len__(self) -> int:
    return len(self._plugins)

  def __iter__(self) -> Iterator[Plugin]:
    return iter(self._plugins.values())

  def __contains__(self, name: str) -> bool:
    return name in self._plugins

  def __repr__(self) -> str:
    return f"<PluginRegistry plugins={len(self._plugins)} loaded={len(self._loaded)}>"
