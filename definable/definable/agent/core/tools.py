"""ToolRegistry — flat name-keyed registry across Tool/Toolkit/MCP/Skill sources.

Owns parallel dispatch, per-call error capture, and event emission for tool
execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
  from definable.agent.core.events import EventBus
  from definable.tool.function import Function, FunctionCall


class ToolRegistry:
  """Name-keyed registry across all tool sources.

  Sources flatten at construction. Once registered, the harness doesn't
  care whether a tool came from a Tool, Toolkit, MCPToolkit, or Skill.

  Phase 2: skeleton — bodies land in Phase 5.
  """

  def __init__(
    self,
    *,
    tools: Sequence[Function] | None = None,
    toolkits: Sequence[Any] | None = None,
    mcp: Sequence[Any] | None = None,
    skills: Sequence[Any] | None = None,
  ) -> None:
    raise NotImplementedError("Phase 5")

  def all(self) -> list[Function]:
    """Return every registered tool, flattened across sources."""
    raise NotImplementedError("Phase 5")

  def names(self) -> list[str]:
    """Return registered tool names."""
    raise NotImplementedError("Phase 5")

  async def dispatch_parallel(
    self,
    calls: Sequence[FunctionCall],
    *,
    events: EventBus,
    run_id: str,
  ) -> list[Any]:
    """Execute calls concurrently. Capture per-call errors, emit lifecycle events."""
    raise NotImplementedError("Phase 5")
