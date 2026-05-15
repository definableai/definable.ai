"""ToolRegistry — flat name-keyed registry across Tool/Toolkit/MCP/Skill sources.

Owns parallel dispatch, per-call error capture, and event emission for tool
execution.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from definable.agent.core.events import (
  EventBus,
  ToolCall,
  ToolCallCompleted,
  ToolCallFailed,
  ToolCallStarted,
  ToolResult,
)

if TYPE_CHECKING:
  from definable.agent.toolkit.function import Function


class ToolRegistry:
  """Name-keyed registry across all tool sources.

  Sources flatten at construction. Once registered, the harness doesn't
  care whether a tool came from a Tool, Toolkit, MCPToolkit, or Skill —
  every source must expose `.tools` as either a list of `Function` or a
  callable returning one.

  Duplicate names raise on construction; the registry is fail-fast on
  ambiguity rather than picking a winner silently.
  """

  def __init__(
    self,
    *,
    tools: Sequence[Function] | None = None,
    toolkits: Sequence[Any] | None = None,
    mcp: Sequence[Any] | None = None,
    skills: Sequence[Any] | None = None,
  ) -> None:
    self._by_name: dict[str, Function] = {}
    for t in tools or []:
      self._add(t)
    for source in (*(toolkits or []), *(mcp or []), *(skills or [])):
      for t in self._extract(source):
        self._add(t)

  @staticmethod
  def _extract(source: Any) -> Iterable[Function]:
    tools_attr = getattr(source, "tools", None)
    if callable(tools_attr):
      result = tools_attr()
      return result if result is not None else []
    return tools_attr or []

  def _add(self, fn: Function) -> None:
    if fn.name in self._by_name:
      raise ValueError(f"Duplicate tool name in registry: {fn.name!r}")
    self._by_name[fn.name] = fn

  def add_from(self, source: Any) -> int:
    """Re-extract tools from a source and add them to the registry.

    Used by Agent.aopen() to register tools that only become available
    after a toolkit's async lifecycle has run (e.g. MCPToolkit, which
    only knows its tools after connecting to its servers). Idempotent:
    tools already registered under the same name are skipped silently.

    Returns the number of tools newly added.
    """
    added = 0
    for fn in self._extract(source):
      if fn.name in self._by_name:
        continue
      self._by_name[fn.name] = fn
      added += 1
    return added

  def all(self) -> list[Function]:
    """Return every registered tool, flattened across sources."""
    return list(self._by_name.values())

  def names(self) -> list[str]:
    """Return registered tool names, in registration order."""
    return list(self._by_name)

  def get(self, name: str) -> Function | None:
    """Look up a tool by name. Returns None if unknown."""
    return self._by_name.get(name)

  async def dispatch_parallel(
    self,
    calls: Sequence[ToolCall],
    *,
    events: EventBus,
    run_id: str,
  ) -> list[ToolResult]:
    """Execute calls concurrently. Capture per-call errors, emit lifecycle events.

    Each call fires `ToolCallStarted`, then either `ToolCallCompleted` or
    `ToolCallFailed`. Returns a list of `ToolResult` in the same order as
    the input calls.

    Unknown tool names produce a failed `ToolResult`; they do not raise.
    """
    return await asyncio.gather(*(self._run_one(c, events=events, run_id=run_id) for c in calls))

  async def _run_one(self, call: ToolCall, *, events: EventBus, run_id: str) -> ToolResult:
    events.emit(ToolCallStarted(run_id=run_id, timestamp=time.time(), call=call))

    fn = self._by_name.get(call.name)
    if fn is None:
      err = f"Unknown tool: {call.name!r}"
      events.emit(ToolCallFailed(run_id=run_id, timestamp=time.time(), call=call, error=err))
      return ToolResult(call=call, success=False, error=err)

    try:
      # Late import keeps the harness loadable without pulling Function machinery.
      from definable.agent.toolkit.function import FunctionCall

      fc = FunctionCall(function=fn, arguments=dict(call.args), call_id=call.id)
      exec_result = await fc.aexecute()
    except Exception as e:
      msg = str(e) or e.__class__.__name__
      events.emit(ToolCallFailed(run_id=run_id, timestamp=time.time(), call=call, error=msg))
      return ToolResult(call=call, success=False, error=msg)

    if exec_result.status == "success":
      events.emit(ToolCallCompleted(run_id=run_id, timestamp=time.time(), call=call, output=exec_result.result))
      return ToolResult(call=call, success=True, output=exec_result.result)

    err = exec_result.error or "tool returned failure status"
    events.emit(ToolCallFailed(run_id=run_id, timestamp=time.time(), call=call, error=err))
    return ToolResult(call=call, success=False, error=err)
