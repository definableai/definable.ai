# Plugin System

> Extensibility architecture for Definable AI agents -- composable, dependency-aware, conflict-safe.

Plugins are the highest-level extensibility unit in Definable. They can register hooks on the pipeline, subscribe to events, add tools, modify configuration, and compose with other plugins. The PluginRegistry handles dependency ordering (topological sort), conflict detection, enable/disable without unloading, and clean teardown.

## Quick Start

```python
import asyncio
from definable.agent.plugin import Plugin, PluginRegistry
from definable.agent.plugin.builtin import LoggingPlugin, MetricsPlugin, CachingPlugin

# Create a custom plugin
class HelloPlugin(Plugin):
  @property
  def name(self):
    return "hello"

  async def on_load(self, agent):
    agent.pipeline.hook("after:prepare", self._greet)

  async def _greet(self, state):
    print(f"Hello from plugin! Run ID: {state.run_id}")
    return state

# Register and load
registry = PluginRegistry()
registry.add(HelloPlugin())
registry.add(LoggingPlugin(verbose=True))
registry.add(MetricsPlugin())

print(len(registry))          # 3
print("hello" in registry)    # True

# Load all onto an agent (dependency-ordered)
await registry.load_all(agent)
print(registry.loaded_names)  # ["hello", "logging", "metrics"]

# Clean teardown (reverse order)
await registry.unload_all(agent)
```

## Architecture

```
PluginRegistry (manages lifecycle)
  |
  +-- _plugins: Dict[name, Plugin]    -- all registered plugins
  +-- _loaded: Set[name]              -- currently loaded plugins
  +-- _disabled: Set[name]            -- disabled plugins (skipped during load_all)
  +-- _load_order: List[name]         -- topologically sorted order
  |
  +-- Validation
  |     +-- dependency resolution (Kahn's algorithm)
  |     +-- conflict detection (explicit conflicts)
  |     +-- shared-modifies warnings
  |
  +-- Plugin (ABC)
        +-- name (abstract)           -- unique identifier
        +-- version                   -- "0.1.0"
        +-- requires                  -- frozenset of dependency names
        +-- conflicts                 -- frozenset of incompatible names
        +-- modifies                  -- frozenset of affected phases
        +-- on_load(agent)            -- register hooks, tools, events
        +-- on_unload(agent)          -- cleanup
```

### Module Structure

```
agent/plugin/
+-- __init__.py         # Public API: Plugin, PluginRegistry
+-- base.py             # Plugin ABC (identity, dependencies, lifecycle)
+-- registry.py         # PluginRegistry (ordering, conflicts, load/unload)
+-- builtin/
    +-- __init__.py     # Public API: LoggingPlugin, MetricsPlugin, CachingPlugin
    +-- logging_plugin.py   # Structured logging on all phases
    +-- metrics_plugin.py   # Per-run timing and usage metrics
    +-- caching_plugin.py   # LRU prompt cache to skip redundant model calls
```

### How It Connects

```
Agent
  +-- plugins: List[Plugin]         -- user-provided plugins
  +-- _plugin_registry: PluginRegistry
        |
        +-- load_all(agent) ------> topological sort by dependencies
        |                            for each plugin:
        |                              plugin.on_load(agent)
        |                                +-- agent.pipeline.hook(...)
        |                                +-- agent.pipeline.subscribe(...)
        |                                +-- modify agent config, tools, etc.
        |
        +-- unload_all(agent) ----> reverse order teardown
                                     for each plugin:
                                       plugin.on_unload(agent)
```

## API Reference

### Plugin (ABC)

Base class for all plugins. Subclasses must implement `name` and `on_load`.

```python
from definable.agent.plugin import Plugin

class MyPlugin(Plugin):
  @property
  def name(self) -> str:
    return "my-plugin"                      # Required: unique identifier

  @property
  def version(self) -> str:
    return "1.0.0"                          # Default: "0.1.0"

  @property
  def description(self) -> str:
    return "Does something useful."         # Default: ""

  @property
  def requires(self) -> frozenset[str]:
    return frozenset({"logging"})           # Default: frozenset()

  @property
  def conflicts(self) -> frozenset[str]:
    return frozenset({"old-plugin"})        # Default: frozenset()

  @property
  def modifies(self) -> frozenset[str]:
    return frozenset({"invoke_loop"})       # Default: frozenset()

  async def on_load(self, agent) -> None:   # Required
    agent.pipeline.hook("before:invoke_loop", self._hook)

  async def on_unload(self, agent) -> None: # Optional (default: no-op)
    pass
```

**Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `str` | *abstract* | Unique plugin name (used for dedup and dependency resolution) |
| `version` | `str` | `"0.1.0"` | Semantic version string |
| `description` | `str` | `""` | Human-readable description |
| `requires` | `frozenset[str]` | `frozenset()` | Plugin names that must be loaded first |
| `conflicts` | `frozenset[str]` | `frozenset()` | Plugin names that cannot coexist |
| `modifies` | `frozenset[str]` | `frozenset()` | Pipeline phases this plugin modifies (for overlap warnings) |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `on_load` | `await plugin.on_load(agent)` | Called when plugin is loaded onto an agent (abstract) |
| `on_unload` | `await plugin.on_unload(agent)` | Called when plugin is removed from an agent (default: no-op) |
| `to_dict` | `plugin.to_dict() -> dict` | Serialize plugin metadata to a plain dict |

```python
from definable.agent.plugin import Plugin

class AnalyticsPlugin(Plugin):
  @property
  def name(self):
    return "analytics"

  @property
  def version(self):
    return "2.0.0"

  @property
  def requires(self):
    return frozenset({"metrics"})  # Must be loaded after MetricsPlugin

  async def on_load(self, agent):
    agent.pipeline.hook("after:store", self._report)

  async def _report(self, state):
    print(f"Run {state.run_id} completed")
    return state

plugin = AnalyticsPlugin()
print(plugin.to_dict())
# {'name': 'analytics', 'version': '2.0.0', 'description': '',
#  'requires': ['metrics'], 'conflicts': [], 'modifies': []}
```

### PluginRegistry

Manages plugin registration, dependency ordering, conflict detection, and lifecycle.

```python
from definable.agent.plugin import PluginRegistry

registry = PluginRegistry()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `registry.add(plugin) -> PluginRegistry` | Register a plugin (chainable). Raises `ValueError` on duplicate name |
| `remove` | `registry.remove(name) -> Plugin \| None` | Remove a plugin. Raises `RuntimeError` if still loaded |
| `load_all` | `await registry.load_all(agent)` | Load all non-disabled plugins in dependency order |
| `unload_all` | `await registry.unload_all(agent)` | Unload all loaded plugins in reverse order |
| `load_one` | `await registry.load_one(name, agent)` | Load a single plugin (checks deps are loaded) |
| `unload_one` | `await registry.unload_one(name, agent)` | Unload a single plugin (checks no dependents) |
| `disable` | `registry.disable(name)` | Mark plugin as disabled (skipped during `load_all`) |
| `enable` | `registry.enable(name)` | Re-enable a disabled plugin |
| `get` | `registry.get(name) -> Plugin \| None` | Get a plugin by name |
| `is_loaded` | `registry.is_loaded(name) -> bool` | Check if a plugin is currently loaded |

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `plugin_names` | `list[str]` | All registered plugin names |
| `loaded_names` | `list[str]` | Names of currently loaded plugins (in load order) |
| `disabled_names` | `list[str]` | Names of disabled plugins |

**Supports:** `in` (name check), `len`, `iter` (iterates Plugin instances).

```python
from definable.agent.plugin import PluginRegistry
from definable.agent.plugin.builtin import LoggingPlugin, MetricsPlugin

registry = PluginRegistry()
registry.add(LoggingPlugin())
registry.add(MetricsPlugin())

print("logging" in registry)   # True
print(len(registry))           # 2
print(registry.plugin_names)   # ["logging", "metrics"]

# Disable a plugin (skipped during load_all)
registry.disable("logging")
print(registry.disabled_names)  # ["logging"]

# Re-enable
registry.enable("logging")
```

## Built-in Plugins

### LoggingPlugin

Structured logging for pipeline phase transitions and run lifecycle events. Registers wildcard hooks on all phases (`before:*` and `after:*`).

```python
from definable.agent.plugin.builtin import LoggingPlugin

lp = LoggingPlugin(
  verbose=False,    # If True, log full state details (default: summary only)
  log_fn=None,      # Custom log function (default: log_info)
)

print(lp.name)         # "logging"
print(lp.modifies)     # frozenset({"*"})
```

When loaded, logs:
- `[logging] Entering phase: prepare`
- `[logging] Completed phase: prepare`
- (verbose) message count, tool count, output preview

### MetricsPlugin

Collects per-run metrics: phase durations, tool call counts, message counts. Stores history capped at `max_history`.

```python
from definable.agent.plugin.builtin import MetricsPlugin

mp = MetricsPlugin(
  max_history=100,  # Max run metrics to keep (default 100)
)

print(mp.name)  # "metrics"

# After agent runs:
# mp.last              -- RunMetrics for the most recent run
# mp.last.total_duration_ms
# mp.last.phase_durations  -- dict[phase_name, duration_ms]
# mp.last.tool_call_count
# mp.last.message_count
# mp.history           -- list[RunMetrics]
# mp.average_duration_ms  -- average across all recorded runs
```

**RunMetrics dataclass:**

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Run identifier |
| `phase_durations` | `dict[str, float]` | Duration per phase in milliseconds |
| `total_duration_ms` | `float` | Sum of all phase durations |
| `tool_call_count` | `int` | Number of tool calls in the run |
| `message_count` | `int` | Total messages at end of run |

### CachingPlugin

LRU cache for identical prompts. Caches model responses keyed on SHA-256 hash of system prompt + user messages. Cache hits skip the `invoke_loop` phase entirely.

```python
from definable.agent.plugin.builtin import CachingPlugin

cp = CachingPlugin(
  max_size=256,       # Max cached responses (default 256)
  ttl_seconds=0,      # Time-to-live per entry (0 = no expiry, default 0)
)

print(cp.name)       # "caching"
print(cp.modifies)   # frozenset({"invoke_loop"})

# After usage:
# cp.size              -- current cache size
# cp.hit_count         -- total cache hits
# cp.miss_count        -- total cache misses
# cp.clear()           -- clear cache and reset counters
```

## Patterns

### Plugin with Dependencies

```python
from definable.agent.plugin import Plugin

class DashboardPlugin(Plugin):
  @property
  def name(self):
    return "dashboard"

  @property
  def requires(self):
    return frozenset({"metrics"})  # MetricsPlugin must be loaded first

  async def on_load(self, agent):
    # Safe to access MetricsPlugin data here
    agent.pipeline.hook("after:store", self._update_dashboard)

  async def _update_dashboard(self, state):
    # Access metrics from the MetricsPlugin
    return state
```

### Plugin with Conflict Detection

```python
from definable.agent.plugin import Plugin

class NewCachePlugin(Plugin):
  @property
  def name(self):
    return "new-cache"

  @property
  def conflicts(self):
    return frozenset({"caching"})  # Cannot coexist with CachingPlugin

  async def on_load(self, agent):
    agent.pipeline.hook("before:invoke_loop", self._check)

  async def _check(self, state):
    return state
```

### Chaining Registration

```python
from definable.agent.plugin import PluginRegistry
from definable.agent.plugin.builtin import LoggingPlugin, MetricsPlugin, CachingPlugin

registry = PluginRegistry()
registry.add(LoggingPlugin()).add(MetricsPlugin()).add(CachingPlugin())
```

### Selective Loading

```python
from definable.agent.plugin import PluginRegistry
from definable.agent.plugin.builtin import LoggingPlugin, MetricsPlugin

registry = PluginRegistry()
registry.add(LoggingPlugin())
registry.add(MetricsPlugin())

# Load only metrics
await registry.load_one("metrics", agent)
print(registry.is_loaded("metrics"))   # True
print(registry.is_loaded("logging"))   # False

# Unload metrics
await registry.unload_one("metrics", agent)
```

## Gotchas

| Issue | Solution |
|-------|----------|
| Duplicate plugin name on `add()` | Raises `ValueError`. Each plugin name must be unique |
| `remove()` while plugin is loaded | Raises `RuntimeError`. Call `unload_one()` or `unload_all()` first |
| Missing dependency on `load_all()` | Raises `ValueError` listing the unregistered dependency names |
| Active conflict on `load_all()` | Raises `ValueError` listing the conflicting plugin names |
| Dependency cycle | Raises `ValueError` during topological sort (Kahn's algorithm) |
| Two plugins modify same phase | Emits a warning (not error) so the user is aware of potential interactions |
| `unload_one()` with dependents | Raises `RuntimeError` if other loaded plugins require this one |
| `load_one()` with unloaded deps | Raises `ValueError` listing the unloaded dependency names |
| `on_unload` is optional | Default implementation is a no-op; override only if cleanup is needed |
| `disable()` only affects `load_all()` | Already-loaded plugins are not unloaded by `disable()` |

## Related Modules

- **[Pipeline](../pipeline/README.md)** -- Plugins register hooks on pipeline phases via `agent.pipeline.hook(...)`
- **[Agent](../README.md)** -- Agent accepts plugins via the `plugins=` parameter
- **[Scheduler](../scheduler/README.md)** -- Plugins can interact with the scheduler via the agent
