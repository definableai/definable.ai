"""Debug exporter — rich console output for agent debug mode.

Pretty-prints model calls, tool calls, and run lifecycle to stderr.

Usage:
    agent = Agent(model="openai/gpt-4o-mini", debug=True)

    # Or manually:
    from definable.agent.tracing import Tracing, DebugExporter
    agent = Agent(model=model, tracing=Tracing(exporters=[DebugExporter()]))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.events import BaseRunOutputEvent
  from definable.agent.run.agent import (
    ModelCallCompletedEvent,
    ModelCallStartedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
  )


def _truncate(text: str, max_len: int) -> str:
  if len(text) <= max_len:
    return text
  return text[:max_len] + "..."


class DebugExporter:
  """Rich console exporter for agent debug mode.

  Prints a color-coded breakdown of every model call to stderr:
  messages sent, response received, tool calls dispatched,
  tool results returned, and token metrics per turn.

  Args:
    max_content_length: Max chars to display for content fields.
    show_tools: Whether to display tool definitions on ModelCallStarted.
    show_metrics: Whether to display token metrics on ModelCallCompleted.
  """

  def __init__(
    self,
    *,
    max_content_length: int = 500,
    show_tools: bool = True,
    show_metrics: bool = True,
  ) -> None:
    self._max_len = max_content_length
    self._show_tools = show_tools
    self._show_metrics = show_metrics
    self._console: "Console | None" = None

  def _get_console(self) -> "Console":
    if self._console is None:
      from rich.console import Console

      self._console = Console(stderr=True, highlight=False)
    return self._console

  def export(self, event: "BaseRunOutputEvent") -> None:
    from definable.agent.run.agent import (
      ModelCallCompletedEvent,
      ModelCallStartedEvent,
      RunCompletedEvent,
      RunStartedEvent,
      ToolCallCompletedEvent,
      ToolCallStartedEvent,
    )

    if isinstance(event, RunStartedEvent):
      self._handle_run_started(event)
    elif isinstance(event, ModelCallStartedEvent):
      self._handle_model_call_started(event)
    elif isinstance(event, ModelCallCompletedEvent):
      self._handle_model_call_completed(event)
    elif isinstance(event, ToolCallStartedEvent):
      self._handle_tool_call_started(event)
    elif isinstance(event, ToolCallCompletedEvent):
      self._handle_tool_call_completed(event)
    elif isinstance(event, RunCompletedEvent):
      self._handle_run_completed(event)

  def flush(self) -> None:
    if self._console is not None:
      self._console.file.flush()

  def shutdown(self) -> None:
    pass

  # -- Event handlers --

  def _handle_run_started(self, event: "RunStartedEvent") -> None:
    console = self._get_console()
    from rich.panel import Panel

    model = event.model or "unknown"
    provider = event.model_provider or "unknown"
    run_id = event.run_id or "?"
    console.print(
      Panel(
        f"[bold]Run started[/bold] | model=[cyan]{model}[/cyan] ({provider}) | run_id={run_id}",
        style="dim",
        expand=False,
      )
    )

  def _handle_model_call_started(self, event: "ModelCallStartedEvent") -> None:
    console = self._get_console()
    from rich.table import Table

    turn = event.turn
    model_id = event.model_id or "unknown"
    provider = event.model_provider or ""

    # Turn header
    header = f"[bold yellow]Turn {turn}[/bold yellow] | model=[cyan]{model_id}[/cyan]"
    if provider:
      header += f" ({provider})"
    if event.response_format:
      header += " | [dim]structured output[/dim]"
    console.print(f"\n{header}")

    # Messages table
    if event.messages:
      table = Table(show_header=True, header_style="bold", expand=False, padding=(0, 1))
      table.add_column("#", style="dim", width=4)
      table.add_column("Role", width=10)
      table.add_column("Content", overflow="ellipsis")

      role_styles = {
        "system": "cyan",
        "user": "green",
        "assistant": "yellow",
        "tool": "magenta",
      }

      for i, msg in enumerate(event.messages):
        role = msg.role or "?"
        style = role_styles.get(role, "white")
        content = msg.content or ""
        if msg.tool_calls:
          content = f"[tool_calls: {len(msg.tool_calls)}]"
        elif role == "tool" and msg.name:
          content = f"[{msg.name}] {content}"
        content = _truncate(str(content), self._max_len)
        table.add_row(str(i), f"[{style}]{role}[/{style}]", content)

      console.print(table)

    # Tool definitions
    if self._show_tools and event.tool_definitions:
      tool_names = [t.get("function", {}).get("name", "?") for t in event.tool_definitions]
      console.print(f"  [dim]Tools ({len(tool_names)}): {', '.join(tool_names)}[/dim]")

  def _handle_model_call_completed(self, event: "ModelCallCompletedEvent") -> None:
    console = self._get_console()

    # Response content
    if event.content:
      truncated = _truncate(event.content, self._max_len)
      console.print(f"  [yellow]Response:[/yellow] {truncated}")

    # Tool calls
    if event.tool_calls:
      for tc in event.tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        args = _truncate(fn.get("arguments", ""), 200)
        console.print(f"  [magenta]-> {name}[/magenta]({args})")

    # Metrics
    if self._show_metrics and event.metrics:
      m = event.metrics
      parts = []
      if hasattr(m, "input_tokens") and m.input_tokens:
        parts.append(f"in={m.input_tokens}")
      if hasattr(m, "output_tokens") and m.output_tokens:
        parts.append(f"out={m.output_tokens}")
      if hasattr(m, "total_tokens") and m.total_tokens:
        parts.append(f"total={m.total_tokens}")
      if parts:
        console.print(f"  [dim]Tokens: {', '.join(parts)}[/dim]")

  def _handle_tool_call_started(self, event: "ToolCallStartedEvent") -> None:
    console = self._get_console()
    if event.tool:
      name = event.tool.tool_name or "?"
      args = _truncate(str(event.tool.tool_args or ""), 200)
      console.print(f"  [magenta]Tool: {name}[/magenta]({args})")

  def _handle_tool_call_completed(self, event: "ToolCallCompletedEvent") -> None:
    console = self._get_console()
    if event.tool:
      result = _truncate(str(event.tool.result or ""), self._max_len)
      console.print(f"  [dim]  -> {result}[/dim]")

  def _handle_run_completed(self, event: "RunCompletedEvent") -> None:
    console = self._get_console()
    from rich.panel import Panel

    parts = []
    if event.content:
      parts.append(f"content={_truncate(str(event.content), 100)}")
    if event.metrics:
      m = event.metrics
      if hasattr(m, "total_tokens") and m.total_tokens:
        parts.append(f"total_tokens={m.total_tokens}")

    summary = " | ".join(parts) if parts else "done"
    console.print(
      Panel(
        f"[bold]Run completed[/bold] | {summary}",
        style="green",
        expand=False,
      )
    )
