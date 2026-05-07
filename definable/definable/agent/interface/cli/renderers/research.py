"""Renderers for deep research events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.base import BaseRunOutputEvent


class DeepResearchRenderer:
  """Renders DeepResearchStarted, Progress, and Completed events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import (
      DeepResearchCompletedEvent,
      DeepResearchProgressEvent,
      DeepResearchStartedEvent,
    )

    return isinstance(event, (DeepResearchStartedEvent, DeepResearchProgressEvent, DeepResearchCompletedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import (
      DeepResearchCompletedEvent,
      DeepResearchProgressEvent,
      DeepResearchStartedEvent,
    )

    if isinstance(event, DeepResearchStartedEvent):
      from rich.panel import Panel

      query = event.query or "?"
      console.print(Panel(f"[bold blue]Deep Research[/bold blue]: {query}", style="blue", expand=False))
    elif isinstance(event, DeepResearchProgressEvent):
      msg = event.message or f"Wave {event.wave}"
      parts = [msg]
      if event.sources_read:
        parts.append(f"sources={event.sources_read}")
      if event.facts_extracted:
        parts.append(f"facts={event.facts_extracted}")
      console.print(f"  [dim blue]{' | '.join(parts)}[/dim blue]", highlight=False)
    elif isinstance(event, DeepResearchCompletedEvent):
      parts = []
      if event.sources_used:
        parts.append(f"{event.sources_used} sources")
      if event.facts_extracted:
        parts.append(f"{event.facts_extracted} facts")
      if event.waves_executed:
        parts.append(f"{event.waves_executed} waves")
      if event.duration_ms is not None:
        parts.append(f"{event.duration_ms:.0f}ms")
      console.print(f"  [dim blue]Research complete: {' | '.join(parts)}[/dim blue]", highlight=False)
