"""Renderers for tool call events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.agent import ToolCallCompletedEvent, ToolCallStartedEvent
  from definable.run.base import BaseRunOutputEvent


def _truncate(text: str, max_len: int) -> str:
  if len(text) <= max_len:
    return text
  return text[:max_len] + "..."


class ToolCallRenderer:
  """Renders ToolCallStarted and ToolCallCompleted events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import ToolCallCompletedEvent, ToolCallStartedEvent

    return isinstance(event, (ToolCallStartedEvent, ToolCallCompletedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import ToolCallCompletedEvent, ToolCallStartedEvent

    if isinstance(event, ToolCallStartedEvent):
      self._render_started(event, console, config)
    elif isinstance(event, ToolCallCompletedEvent):
      self._render_completed(event, console, config)

  def _render_started(self, event: "ToolCallStartedEvent", console: "Console", config: "CLIConfig") -> None:

    if not event.tool:
      return
    name = event.tool.tool_name or "?"
    line = f"  [magenta]> {name}[/magenta]"
    if config.show_tool_args and event.tool.tool_args:
      args = _truncate(str(event.tool.tool_args), config.max_content_display)
      line += f"({args})"
    console.print(line, highlight=False)

  def _render_completed(self, event: "ToolCallCompletedEvent", console: "Console", config: "CLIConfig") -> None:

    if not config.show_tool_results:
      return
    if not event.tool:
      return
    result = event.tool.result
    if result is not None:
      truncated = _truncate(str(result), config.max_content_display)
      console.print(f"  [dim]  > {truncated}[/dim]", highlight=False)
