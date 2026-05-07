"""Renderers for run lifecycle events (start, complete, error, cancel)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.agent import RunCancelledEvent, RunCompletedEvent, RunErrorEvent, RunStartedEvent
  from definable.run.base import BaseRunOutputEvent


def _truncate(text: str, max_len: int) -> str:
  if len(text) <= max_len:
    return text
  return text[:max_len] + "..."


class RunRenderer:
  """Renders RunStarted, RunCompleted, RunError, RunCancelled events."""

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import (
      RunCancelledEvent,
      RunCompletedEvent,
      RunErrorEvent,
      RunStartedEvent,
    )

    return isinstance(event, (RunStartedEvent, RunCompletedEvent, RunErrorEvent, RunCancelledEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import (
      RunCancelledEvent,
      RunCompletedEvent,
      RunErrorEvent,
      RunStartedEvent,
    )

    if isinstance(event, RunStartedEvent):
      self._render_started(event, console, config)
    elif isinstance(event, RunCompletedEvent):
      self._render_completed(event, console, config)
    elif isinstance(event, RunErrorEvent):
      self._render_error(event, console, config)
    elif isinstance(event, RunCancelledEvent):
      self._render_cancelled(event, console, config)

  def _render_started(self, event: "RunStartedEvent", console: "Console", config: "CLIConfig") -> None:

    pass  # Startup is silent — the banner already shows model info

  def _render_completed(self, event: "RunCompletedEvent", console: "Console", config: "CLIConfig") -> None:

    if not config.show_metrics:
      return
    parts = []
    if event.metrics:
      m = event.metrics
      if hasattr(m, "total_tokens") and m.total_tokens:
        parts.append(f"tokens={m.total_tokens}")
      if hasattr(m, "time_to_first_token") and m.time_to_first_token:
        parts.append(f"ttft={m.time_to_first_token:.2f}s")
    if parts:
      console.print(f"  [dim green]{' | '.join(parts)}[/dim green]")

  def _render_error(self, event: "RunErrorEvent", console: "Console", config: "CLIConfig") -> None:
    from rich.panel import Panel

    msg = event.content or "Unknown error"
    console.print(Panel(f"[bold red]Error[/bold red]: {msg}", style="red", expand=False))

  def _render_cancelled(self, event: "RunCancelledEvent", console: "Console", config: "CLIConfig") -> None:
    reason = event.reason or "cancelled"
    console.print(f"[dim red]Run cancelled: {reason}[/dim red]")
