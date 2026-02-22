"""OutputManager — coordinates Rich console output for the CLI interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.agent import Agent
  from definable.agent.interface.cli.config import CLIConfig


class OutputManager:
  """Manages all console output for the CLI interface.

  Owns a Rich Console instance and provides helper methods for
  printing banners, agent responses, and status messages.

  Args:
    config: CLI configuration.
  """

  def __init__(self, *, config: "CLIConfig") -> None:
    self._config = config
    self._console: Optional["Console"] = None

  @property
  def console(self) -> "Console":
    """Lazy-initialized Rich console (stdout, no highlight by default)."""
    if self._console is None:
      from rich.console import Console

      self._console = Console(highlight=False)
    return self._console

  def print_banner(self, agent: "Agent") -> None:
    """Print the startup banner with agent info."""
    if not self._config.show_banner:
      return

    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, expand=False, padding=(0, 2), box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Agent", agent.agent_name or "unnamed")

    model_id = str(getattr(agent, "_model_id", None) or getattr(agent.model, "id", "?"))
    provider = str(getattr(agent.model, "provider", "?"))
    table.add_row("Model", f"{model_id} ({provider})")

    # Tools
    tool_count = len(agent.tools) if agent.tools else 0
    if tool_count:
      tool_names = [getattr(t, "name", "?") for t in (agent.tools or [])]
      table.add_row("Tools", f"{tool_count} ({', '.join(tool_names[:5])}{'...' if tool_count > 5 else ''})")

    # Features
    features = []
    if getattr(agent, "_memory", None):
      features.append("memory")
    if getattr(agent, "_knowledge", None):
      features.append("knowledge")
    if getattr(agent, "_thinking", None):
      features.append("thinking")
    if getattr(agent, "_tracing", None):
      features.append("tracing")
    if getattr(agent, "_deep_research", None):
      features.append("research")
    if features:
      table.add_row("Features", ", ".join(features))

    self.console.print(
      Panel(
        table,
        title="[bold]Definable AI[/bold]",
        subtitle="[dim]Type /help for commands, Ctrl+C to exit[/dim]",
        expand=False,
      )
    )
    self.console.print()

  def print_response(self, content: str) -> None:
    """Print an agent response, optionally as rendered Markdown."""
    if self._config.markdown_output:
      from rich.markdown import Markdown

      self.console.print(Markdown(content))
    else:
      self.console.print(content)
    self.console.print()
