"""Renderers for sub-agent lifecycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.run.base import BaseRunOutputEvent


class SubAgentRenderer:
  """Renders SubAgentSpawned, Completed, Failed, Killed events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.agent.run.agent import (
      SubAgentCompletedEvent,
      SubAgentFailedEvent,
      SubAgentKilledEvent,
      SubAgentSpawnedEvent,
    )

    return isinstance(event, (SubAgentSpawnedEvent, SubAgentCompletedEvent, SubAgentFailedEvent, SubAgentKilledEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.agent.run.agent import (
      SubAgentCompletedEvent,
      SubAgentFailedEvent,
      SubAgentKilledEvent,
      SubAgentSpawnedEvent,
    )

    if isinstance(event, SubAgentSpawnedEvent):
      goal = event.goal or "sub-agent"
      console.print(f"  [cyan]Spawned: {goal}[/cyan]", highlight=False)
    elif isinstance(event, SubAgentCompletedEvent):
      console.print("  [dim cyan]Sub-agent completed[/dim cyan]", highlight=False)
    elif isinstance(event, SubAgentFailedEvent):
      error = event.error or "unknown"
      console.print(f"  [red]Sub-agent failed: {error}[/red]", highlight=False)
    elif isinstance(event, SubAgentKilledEvent):
      reason = event.reason or "killed"
      console.print(f"  [dim red]Sub-agent killed: {reason}[/dim red]", highlight=False)
