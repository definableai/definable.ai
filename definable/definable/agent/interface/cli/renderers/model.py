"""Renderers for model call events (turn headers, token metrics)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.run.agent import ModelCallCompletedEvent, ModelCallStartedEvent
  from definable.agent.run.base import BaseRunOutputEvent


class ModelCallRenderer:
  """Renders ModelCallStarted and ModelCallCompleted events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.agent.run.agent import ModelCallCompletedEvent, ModelCallStartedEvent

    return isinstance(event, (ModelCallStartedEvent, ModelCallCompletedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.agent.run.agent import ModelCallCompletedEvent, ModelCallStartedEvent

    if isinstance(event, ModelCallStartedEvent):
      self._render_started(event, console, config)
    elif isinstance(event, ModelCallCompletedEvent):
      self._render_completed(event, console, config)

  def _render_started(self, event: "ModelCallStartedEvent", console: "Console", config: "CLIConfig") -> None:

    if event.turn > 1:
      console.print(f"  [bold yellow]Turn {event.turn}[/bold yellow]", highlight=False)

  def _render_completed(self, event: "ModelCallCompletedEvent", console: "Console", config: "CLIConfig") -> None:

    if not config.show_metrics:
      return
    if event.metrics:
      m = event.metrics
      parts = []
      if hasattr(m, "input_tokens") and m.input_tokens:
        parts.append(f"in={m.input_tokens}")
      if hasattr(m, "output_tokens") and m.output_tokens:
        parts.append(f"out={m.output_tokens}")
      if parts:
        console.print(f"  [dim]{' '.join(parts)}[/dim]", highlight=False)
