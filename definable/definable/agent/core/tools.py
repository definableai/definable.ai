"""ToolRegistry — flat name-keyed registry across Tool/Toolkit/MCP/Skill sources.

Owns parallel dispatch, per-call error capture, before/after-tool hooks, and
tool step events.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from definable.agent.core.events import (
  EventBus,
  StepBegin,
  StepEnd,
  ToolCall,
  ToolResult,
)
from definable.agent.core.hooks import AbortRun, Hook, SkipTool, ToolHookContext

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

  async def dispatch(
    self,
    calls: Sequence[ToolCall],
    *,
    events: EventBus,
    run_id: str,
    turn: int,
    hooks: Sequence[Hook] = (),
  ) -> list[ToolResult]:
    """Execute calls concurrently with before/after-tool hooks.

    Per call the order is: before_tool → StepBegin(tool) → execute →
    StepEnd(tool) → after_tool (the step is nested inside the hooks).
    Returns `ToolResult`s in input order. Unknown tools and tool exceptions
    become failed results, never raised. A before_tool hook raising
    `SkipTool` vetoes the call (no step events emitted); raising `AbortRun`
    marks the result aborted (the loop stops after the batch).
    """
    return list(await asyncio.gather(*(self._run_one(c, events=events, run_id=run_id, turn=turn, hooks=hooks) for c in calls)))

  async def _run_one(self, call: ToolCall, *, events: EventBus, run_id: str, turn: int, hooks: Sequence[Hook]) -> ToolResult:
    sid = call.id or f"{run_id}:{turn}:tool:{call.name}"
    hctx = ToolHookContext(run_id=run_id, turn=turn, call=call, args=dict(call.args))

    # before_tool runs BEFORE the step opens — may edit args, or veto (no step emitted).
    try:
      for h in hooks:
        await h.before_tool(hctx)
    except SkipTool as e:
      return ToolResult(call=call, success=False, error=str(e) or "skipped", skipped=True)
    except AbortRun as e:
      return ToolResult(call=call, success=False, error=str(e) or "aborted", aborted=True)

    # The tool step is nested INSIDE the hooks: StepBegin … execute … StepEnd.
    start = time.time()
    events.emit(StepBegin(run_id=run_id, timestamp=start, id=sid, type="tool", turn=turn, name=call.name, args=hctx.args))
    result = await self._execute(call, hctx.args)
    end = time.time()
    events.emit(
      StepEnd(
        run_id=run_id,
        timestamp=end,
        id=sid,
        type="tool",
        data=str(result.output) if result.success else None,
        success=result.success,
        error=result.error,
        duration_ms=max(0.0, (end - start) * 1000.0),
      )
    )

    # after_tool runs AFTER the step closed — may post-process the result the loop records.
    hctx.result = result
    try:
      for h in hooks:
        await h.after_tool(hctx)
    except AbortRun as e:
      return ToolResult(call=call, success=False, error=str(e) or "aborted", aborted=True)
    return hctx.result or result

  async def _execute(self, call: ToolCall, args: dict[str, Any]) -> ToolResult:
    """Run one tool by name. Unknown names and exceptions become failed results."""
    fn = self._by_name.get(call.name)
    if fn is None:
      return ToolResult(call=call, success=False, error=f"Unknown tool: {call.name!r}")
    try:
      # Late import keeps the harness loadable without pulling Function machinery.
      from definable.agent.toolkit.function import FunctionCall

      fc = FunctionCall(function=fn, arguments=dict(args), call_id=call.id)
      exec_result = await fc.aexecute()
    except Exception as e:
      return ToolResult(call=call, success=False, error=str(e) or e.__class__.__name__)

    if exec_result.status == "success":
      return ToolResult(call=call, success=True, output=exec_result.result)
    return ToolResult(call=call, success=False, error=exec_result.error or "tool returned failure status")
