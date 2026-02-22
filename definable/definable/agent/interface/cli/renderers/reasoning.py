"""Renderers for reasoning/thinking events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.run.base import BaseRunOutputEvent


def _truncate(text: str, max_len: int) -> str:
  if len(text) <= max_len:
    return text
  return text[:max_len] + "..."


class ReasoningRenderer:
  """Renders ReasoningStarted, ReasoningStep, ReasoningCompleted events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.agent.run.agent import (
      ReasoningCompletedEvent,
      ReasoningStartedEvent,
      ReasoningStepEvent,
    )

    return isinstance(event, (ReasoningStartedEvent, ReasoningStepEvent, ReasoningCompletedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    if not config.show_thinking:
      return

    from definable.agent.run.agent import (
      ReasoningCompletedEvent,
      ReasoningStartedEvent,
      ReasoningStepEvent,
    )

    if isinstance(event, ReasoningStartedEvent):
      console.print("  [dim cyan]Thinking...[/dim cyan]", highlight=False)
    elif isinstance(event, ReasoningStepEvent):
      if event.reasoning_content:
        truncated = _truncate(event.reasoning_content, config.max_content_display)
        console.print(f"  [dim cyan]{truncated}[/dim cyan]", highlight=False)
    elif isinstance(event, ReasoningCompletedEvent):
      pass  # Silent completion — thinking content was already shown
