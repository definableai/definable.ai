"""Renderers for knowledge retrieval events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.base import BaseRunOutputEvent


class KnowledgeRenderer:
  """Renders KnowledgeRetrievalStarted and KnowledgeRetrievalCompleted events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import (
      KnowledgeRetrievalCompletedEvent,
      KnowledgeRetrievalStartedEvent,
    )

    return isinstance(event, (KnowledgeRetrievalStartedEvent, KnowledgeRetrievalCompletedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import (
      KnowledgeRetrievalCompletedEvent,
      KnowledgeRetrievalStartedEvent,
    )

    if isinstance(event, KnowledgeRetrievalStartedEvent):
      console.print("  [blue]Searching knowledge...[/blue]", highlight=False)
    elif isinstance(event, KnowledgeRetrievalCompletedEvent):
      parts = [f"Found {event.documents_found} docs"]
      if event.duration_ms is not None:
        parts.append(f"{event.duration_ms:.0f}ms")
      console.print(f"  [dim blue]{' | '.join(parts)}[/dim blue]", highlight=False)
