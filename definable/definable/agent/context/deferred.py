"""Deferred tool loading — progressive disclosure of tool schemas.

Instead of sending all tool schemas on every model call (expensive),
only tool names + one-line descriptions are included in the system
prompt. A built-in ``load_tools`` tool lets the model request full
schemas when it actually needs them. Loaded schemas are injected
into the loop for subsequent iterations.

This is the same pattern Anthropic uses in Claude Code (deferred tools
+ ToolSearch) and recommends in their advanced tool-use guide.
"""

from typing import TYPE_CHECKING, Dict, Optional

from definable.tool.function import Function

if TYPE_CHECKING:
  pass

# Name of the auto-injected loader tool
LOAD_TOOLS_NAME = "load_tools"


class DeferredToolManager:
  """Manages deferred tool loading for an agent run.

  Given the agent's full tool registry, produces:
  1. A compact text catalog for the system prompt (~3 tokens/tool).
  2. A ``load_tools`` Function that the model can call.
  3. Methods to inject loaded schemas into the loop's tool set.

  Lifecycle:
    - Created once per ``Agent`` (reused across runs).
    - ``prepare_for_run()`` called at the start of each run to reset state.
    - ``build_catalog()`` returns the text catalog for the system prompt.
    - ``get_loader_tool()`` returns the built-in ``load_tools`` Function.
    - ``get_active_tools()`` returns currently loaded tools (for the loop).
    - ``handle_load_result()`` called after load_tools executes, returns
      the schemas that should be injected.

  Example:
    mgr = DeferredToolManager(all_tools)
    catalog = mgr.build_catalog()        # → text for system prompt
    loader = mgr.get_loader_tool()       # → Function for tools param
    # After model calls load_tools(["search", "write_file"]):
    mgr.load(["search", "write_file"])   # → activates those tools
    active = mgr.get_active_tools()      # → {load_tools, search, write_file}
  """

  def __init__(self, all_tools: Dict[str, Function]) -> None:
    self._all_tools = dict(all_tools)  # full registry (never modified)
    self._loaded: Dict[str, Function] = {}  # tools activated this run
    self._loader_tool: Optional[Function] = None

  @property
  def all_tool_names(self) -> list[str]:
    return list(self._all_tools.keys())

  @property
  def loaded_tool_names(self) -> list[str]:
    return list(self._loaded.keys())

  def prepare_for_run(self) -> None:
    """Reset loaded tools for a new run."""
    self._loaded.clear()

  def build_catalog(self) -> str:
    """Build a compact text catalog of all tools for the system prompt.

    Format: one line per tool — ``- tool_name: description``
    Roughly ~3-5 tokens per tool.

    Returns:
      Formatted catalog string, or empty string if no tools.
    """
    if not self._all_tools:
      return ""

    lines = ["## Available Tools", "", "Call `load_tools` with the tool names you need before using them.", ""]

    for name, fn in self._all_tools.items():
      desc = _get_short_description(fn)
      lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)

  def get_loader_tool(self) -> Function:
    """Return the built-in ``load_tools`` Function.

    This tool is always included in the tools parameter. When the
    model calls it, the requested tool schemas become available
    for subsequent iterations.

    Returns:
      A Function instance for ``load_tools``.
    """
    if self._loader_tool is not None:
      return self._loader_tool

    available_names = list(self._all_tools.keys())

    def load_tools(names: list[str]) -> str:
      """Load tool schemas so you can call them.

      Args:
        names: List of tool names to load from the available tools catalog.

      Returns:
        Confirmation of which tools were loaded and are now available.
      """
      loaded_now: list[str] = []
      not_found: list[str] = []

      for name in names:
        if name in self._all_tools:
          self._loaded[name] = self._all_tools[name]
          loaded_now.append(name)
        elif name == LOAD_TOOLS_NAME:
          continue  # Skip self-reference
        else:
          not_found.append(name)

      parts: list[str] = []
      if loaded_now:
        parts.append(f"Loaded: {', '.join(loaded_now)}. You can now call these tools.")
      if not_found:
        parts.append(f"Not found: {', '.join(not_found)}. Available: {', '.join(available_names)}")

      return " ".join(parts) if parts else "No tools specified."

    # Convert the function into a definable Function
    from definable.tool.decorator import tool

    loader_fn = tool(load_tools)
    self._loader_tool = loader_fn
    return loader_fn

  def load(self, names: list[str]) -> list[str]:
    """Programmatically load tools (used internally after load_tools executes).

    Args:
      names: Tool names to activate.

    Returns:
      Names of tools successfully loaded.
    """
    loaded: list[str] = []
    for name in names:
      if name in self._all_tools:
        self._loaded[name] = self._all_tools[name]
        loaded.append(name)
    return loaded

  def get_active_tools(self) -> Dict[str, Function]:
    """Return tools that should be in the loop's tool set.

    Always includes ``load_tools``. Includes any tools that have
    been loaded via ``load_tools`` calls.

    Returns:
      Dict of tool_name → Function for the current iteration.
    """
    active: Dict[str, Function] = {}
    # Always include the loader
    active[LOAD_TOOLS_NAME] = self.get_loader_tool()
    # Include all loaded tools
    active.update(self._loaded)
    return active

  def get_tools_dicts(self) -> list[dict]:
    """Return OpenAI-format tool dicts for the active tool set.

    Returns:
      List of tool schema dicts for the model API.
    """
    active = self.get_active_tools()
    return [{"type": "function", "function": t.to_dict()} for t in active.values()]

  def is_deferred(self) -> bool:
    """Return True if this manager has deferred tools."""
    return len(self._all_tools) > 0


def _get_short_description(fn: Function) -> str:
  """Extract a one-line description from a Function.

  Uses the first line of the function's description or docstring.
  Falls back to 'No description' if empty.
  """
  desc = fn.description or ""
  if not desc and hasattr(fn, "entrypoint") and fn.entrypoint:
    doc = getattr(fn.entrypoint, "__doc__", "") or ""
    desc = doc.strip()

  # Take only the first line
  first_line = desc.split("\n")[0].strip() if desc else ""
  return first_line or "No description"
