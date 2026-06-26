"""Hooks — the control plane.

Four seams around the loop: before_model, after_model, before_tool,
after_tool. Unlike EventBus subscribers (observe-only), a hook receives a
*mutable* context and may change it or raise to alter control flow:

- edit ``ctx.messages`` in before_model, ``ctx.response`` in after_model
- edit ``ctx.args`` in before_tool, ``ctx.result`` in after_tool
- raise ``SkipTool`` in before_tool to skip that call
- raise ``AbortRun`` in any hook to stop the run (exit_reason="aborted")

Subclass :class:`Hook` and override only the seams you need; the rest are
no-ops. Pass instances to ``Agent(hooks=[...])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from definable.agent.core.events import ToolCall, ToolResult
  from definable.model.message import Message
  from definable.model.response import ModelResponse


class AbortRun(Exception):
  """Raise in any hook to stop the run gracefully (exit_reason='aborted')."""


class SkipTool(Exception):
  """Raise in before_tool to skip executing this tool call."""


@dataclass
class ModelHookContext:
  """Passed to before_model / after_model. Mutate in place."""

  run_id: str
  turn: int
  messages: list[Message]  # before_model may edit
  tools: list[dict[str, Any]] | None
  response: ModelResponse | None = None  # set for after_model; may be replaced


@dataclass
class ToolHookContext:
  """Passed to before_tool / after_tool. Mutate in place."""

  run_id: str
  turn: int
  call: ToolCall
  args: dict[str, Any]  # before_tool may edit; loop reads this back
  result: ToolResult | None = None  # set for after_tool; may be replaced


class Hook:
  """Base hook — override the seams you need; defaults are no-ops."""

  async def before_model(self, ctx: ModelHookContext) -> None: ...

  async def after_model(self, ctx: ModelHookContext) -> None: ...

  async def before_tool(self, ctx: ToolHookContext) -> None: ...

  async def after_tool(self, ctx: ToolHookContext) -> None: ...
