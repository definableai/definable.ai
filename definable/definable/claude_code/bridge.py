"""ToolBridge — converts Definable @tool functions into MCP tool definitions.

Handles registration, MCP config generation, name prefixing, and async/sync dispatch.
"""

import asyncio
from inspect import iscoroutinefunction
from typing import Any, Dict, List, Optional

from definable.tool.function import Function
from definable.utils.log import log_debug, log_error


class ToolBridge:
  """Bridges Definable tools into the Claude Code MCP protocol.

  Registers tools from ``@tool`` decorated functions and Skill objects,
  generates MCP server configuration for CLI initialization, and dispatches
  tool calls back from the CLI.
  """

  def __init__(self, tools: Optional[List[Function]] = None, skills: Optional[List[Any]] = None):
    self._tools: Dict[str, Function] = {}
    if tools:
      self._register_tools(tools)
    if skills:
      self._register_skill_tools(skills)

  def _register_tools(self, tools: List[Function]) -> None:
    """Register a list of Definable Function objects."""
    for fn in tools:
      if not isinstance(fn, Function):
        log_error(f"Skipping non-Function tool: {type(fn).__name__}")  # type: ignore[unreachable]
        continue
      # Ensure entrypoint is processed
      if fn.entrypoint and not fn.description:
        fn.process_entrypoint()
      self._tools[fn.name] = fn
      log_debug(f"Registered MCP tool: {fn.name}")

  def _register_skill_tools(self, skills: List[Any]) -> None:
    """Extract and register tools from Skill objects."""
    for skill in skills:
      skill_tools = getattr(skill, "tools", None)
      if skill_tools:
        for fn in skill_tools:
          if isinstance(fn, Function):
            self._tools[fn.name] = fn
            log_debug(f"Registered skill tool: {fn.name} (from {getattr(skill, 'name', 'unknown')})")

  @property
  def tool_count(self) -> int:
    return len(self._tools)

  def get_mcp_config(self, server_name: str = "definable") -> dict:
    """Generate MCP server config for CLI initialization.

    Returns a dict suitable for the ``mcp_servers`` field of the
    CLI init control request.
    """
    if not self._tools:
      return {}
    return {
      server_name: {
        "type": "sdk",
        "tools": [
          {
            "name": fn.name,
            "description": fn.description or fn.name,
            "inputSchema": fn.parameters,
          }
          for fn in self._tools.values()
        ],
      }
    }

  def get_tool_names(self, server_name: str = "definable") -> List[str]:
    """Get MCP-prefixed tool names for ``--allowedTools``."""
    return [f"mcp__{server_name}__{name}" for name in self._tools]

  async def execute(self, tool_name: str, args: Dict[str, Any]) -> dict:
    """Execute a Definable tool and return MCP-format result.

    Strips the MCP prefix (``mcp__definable__deploy`` → ``deploy``)
    before looking up the tool. Handles both sync and async entrypoints.

    Returns:
      MCP tool result dict with ``content`` list and optional ``isError``.
    """
    # Strip MCP prefix: "mcp__definable__deploy" → "deploy"
    bare_name = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name
    fn = self._tools.get(bare_name)
    if not fn:
      return {
        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
        "isError": True,
      }

    if fn.entrypoint is None:
      return {
        "content": [{"type": "text", "text": f"Tool '{bare_name}' has no entrypoint"}],
        "isError": True,
      }

    try:
      result = fn.entrypoint(**args)
      if iscoroutinefunction(fn.entrypoint):
        result = await result
      elif asyncio.iscoroutine(result):
        result = await result
      return {"content": [{"type": "text", "text": str(result)}]}
    except Exception as exc:
      log_error(f"Tool '{bare_name}' execution failed: {exc}")
      return {
        "content": [{"type": "text", "text": f"Error executing {bare_name}: {exc}"}],
        "isError": True,
      }
