"""Renderers for guardrail events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.run.base import BaseRunOutputEvent


class GuardrailRenderer:
  """Renders GuardrailChecked and GuardrailBlocked events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    try:
      from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

      return isinstance(event, (GuardrailCheckedEvent, GuardrailBlockedEvent))
    except ImportError:
      return False

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.agent.guardrail.events import GuardrailBlockedEvent, GuardrailCheckedEvent

    if isinstance(event, GuardrailBlockedEvent):
      from rich.panel import Panel

      console.print(
        Panel(
          f"[bold red]Blocked by {event.guardrail_name}[/bold red]: {event.reason}",
          style="red",
          expand=False,
        )
      )
    elif isinstance(event, GuardrailCheckedEvent):
      if event.action == "warn" and event.message:
        console.print(f"  [dim yellow]Guardrail warning ({event.guardrail_name}): {event.message}[/dim yellow]", highlight=False)
