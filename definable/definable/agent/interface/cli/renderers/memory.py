"""Renderers for memory recall and update events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.base import BaseRunOutputEvent


class MemoryRenderer:
  """Renders memory recall and update lifecycle events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import (
      MemoryRecallCompletedEvent,
      MemoryRecallStartedEvent,
      MemoryUpdateCompletedEvent,
      MemoryUpdateStartedEvent,
    )

    return isinstance(
      event,
      (
        MemoryRecallStartedEvent,
        MemoryRecallCompletedEvent,
        MemoryUpdateStartedEvent,
        MemoryUpdateCompletedEvent,
      ),
    )

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import (
      MemoryRecallCompletedEvent,
      MemoryRecallStartedEvent,
      MemoryUpdateCompletedEvent,
      MemoryUpdateStartedEvent,
    )

    if isinstance(event, MemoryRecallStartedEvent):
      console.print("  [yellow]Recalling memory...[/yellow]", highlight=False)
    elif isinstance(event, MemoryRecallCompletedEvent):
      parts = [f"{event.chunks_included} chunks loaded"]
      if event.duration_ms is not None:
        parts.append(f"{event.duration_ms:.0f}ms")
      console.print(f"  [dim yellow]{' | '.join(parts)}[/dim yellow]", highlight=False)
    elif isinstance(event, MemoryUpdateStartedEvent):
      console.print("  [yellow]Updating memory...[/yellow]", highlight=False)
    elif isinstance(event, MemoryUpdateCompletedEvent):
      parts = [f"{event.message_count} messages"]
      if event.duration_ms is not None:
        parts.append(f"{event.duration_ms:.0f}ms")
      console.print(f"  [dim yellow]{' | '.join(parts)}[/dim yellow]", highlight=False)
