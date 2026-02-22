"""Built-in CLI slash commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
  from definable.agent.interface.cli.commands import CommandContext, CommandRegistry


class HelpCommand:
  """List all available commands."""

  @property
  def name(self) -> str:
    return "help"

  @property
  def description(self) -> str:
    return "Show available commands"

  @property
  def aliases(self) -> List[str]:
    return ["h", "?"]

  async def execute(self, args: str, context: CommandContext) -> None:
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", expand=False, padding=(0, 2))
    table.add_column("Command", style="cyan")
    table.add_column("Aliases", style="dim")
    table.add_column("Description")

    # Access the command registry through the interface
    interface = context.interface
    registry = getattr(interface, "_command_registry", None)
    if registry is None:
      return

    for cmd in registry.all_commands:
      alias_str = ", ".join(f"/{a}" for a in cmd.aliases) if cmd.aliases else ""
      table.add_row(f"/{cmd.name}", alias_str, cmd.description)

    context.output.console.print(table)


class InfoCommand:
  """Show agent configuration details."""

  @property
  def name(self) -> str:
    return "info"

  @property
  def description(self) -> str:
    return "Show agent configuration"

  @property
  def aliases(self) -> List[str]:
    return ["i"]

  async def execute(self, args: str, context: CommandContext) -> None:
    from rich.table import Table

    agent = context.agent
    table = Table(show_header=False, expand=False, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Name", agent.agent_name or "unnamed")
    table.add_row("Model", str(getattr(agent, "_model_id", None) or getattr(agent.model, "id", "?")))
    table.add_row("Provider", str(getattr(agent.model, "provider", "?")))

    # Tools count
    tool_count = len(agent.tools) if agent.tools else 0
    table.add_row("Tools", str(tool_count))

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
    if getattr(agent, "_guardrails", None):
      features.append("guardrails")
    if getattr(agent, "_sub_agents", None):
      features.append("sub-agents")
    table.add_row("Features", ", ".join(features) if features else "none")

    # Session
    table.add_row("Session", context.session.session_id)
    table.add_row("Messages", str(len(context.session.messages or [])))

    context.output.console.print(table)


class ToolsCommand:
  """Show available tools."""

  @property
  def name(self) -> str:
    return "tools"

  @property
  def description(self) -> str:
    return "List available tools"

  @property
  def aliases(self) -> List[str]:
    return ["t"]

  async def execute(self, args: str, context: CommandContext) -> None:
    from rich.table import Table

    agent = context.agent
    tools = agent.tools or []

    if not tools:
      context.output.console.print("[dim]No tools configured[/dim]")
      return

    table = Table(show_header=True, header_style="bold", expand=False, padding=(0, 2))
    table.add_column("Tool", style="magenta")
    table.add_column("Description")

    for tool in tools:
      name = getattr(tool, "name", "?")
      description = getattr(tool, "description", "") or ""
      # Truncate long descriptions
      if len(description) > 80:
        description = description[:77] + "..."
      table.add_row(name, description)

    context.output.console.print(table)


class ModelCommand:
  """Show model details."""

  @property
  def name(self) -> str:
    return "model"

  @property
  def description(self) -> str:
    return "Show model details"

  @property
  def aliases(self) -> List[str]:
    return ["m"]

  async def execute(self, args: str, context: CommandContext) -> None:
    from rich.table import Table

    model = context.agent.model
    table = Table(show_header=False, expand=False, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("ID", str(getattr(model, "id", "?")))
    table.add_row("Provider", str(getattr(model, "provider", "?")))

    # Optional attributes
    for attr in ("temperature", "max_tokens", "top_p"):
      val = getattr(model, attr, None)
      if val is not None:
        table.add_row(attr.replace("_", " ").title(), str(val))

    context.output.console.print(table)


class ClearCommand:
  """Clear the terminal screen."""

  @property
  def name(self) -> str:
    return "clear"

  @property
  def description(self) -> str:
    return "Clear the screen"

  @property
  def aliases(self) -> List[str]:
    return ["cls"]

  async def execute(self, args: str, context: CommandContext) -> None:
    context.output.console.clear()


class HistoryCommand:
  """Show conversation messages."""

  @property
  def name(self) -> str:
    return "history"

  @property
  def description(self) -> str:
    return "Show conversation history"

  @property
  def aliases(self) -> List[str]:
    return ["hist"]

  async def execute(self, args: str, context: CommandContext) -> None:
    messages = context.session.messages
    if not messages:
      context.output.console.print("[dim]No messages in session[/dim]")
      return

    role_styles = {
      "system": "cyan",
      "user": "green",
      "assistant": "yellow",
      "tool": "magenta",
    }

    for i, msg in enumerate(messages):
      role = msg.role or "?"
      style = role_styles.get(role, "white")
      content = msg.content or ""
      if len(str(content)) > 200:
        content = str(content)[:197] + "..."
      context.output.console.print(f"  [{style}]{role}[/{style}]: {content}", highlight=False)


class ExportCommand:
  """Export conversation history to a JSON file."""

  @property
  def name(self) -> str:
    return "export"

  @property
  def description(self) -> str:
    return "Export history to file (JSON)"

  @property
  def aliases(self) -> List[str]:
    return []

  async def execute(self, args: str, context: CommandContext) -> None:
    path = args.strip() or "chat_history.json"
    messages = context.session.messages
    if not messages:
      context.output.console.print("[dim]No messages to export[/dim]")
      return

    data = [m.to_dict() for m in messages]
    with open(path, "w") as f:
      json.dump(data, f, indent=2)
    context.output.console.print(f"[green]Exported {len(messages)} messages to {path}[/green]")


class ResetCommand:
  """Clear session and start fresh."""

  @property
  def name(self) -> str:
    return "reset"

  @property
  def description(self) -> str:
    return "Clear session and start fresh"

  @property
  def aliases(self) -> List[str]:
    return ["new"]

  async def execute(self, args: str, context: CommandContext) -> None:
    if context.session.messages is not None:
      context.session.messages.clear()
    context.session.last_run_output = None
    context.output.console.print("[green]Session reset[/green]")


class QuitCommand:
  """Exit the CLI."""

  @property
  def name(self) -> str:
    return "quit"

  @property
  def description(self) -> str:
    return "Exit the CLI"

  @property
  def aliases(self) -> List[str]:
    return ["exit", "q"]

  async def execute(self, args: str, context: CommandContext) -> None:
    context.output.console.print("[dim]Goodbye.[/dim]")
    # Signal the interface to stop
    interface = context.interface
    if hasattr(interface, "_running"):
      interface._running = False  # type: ignore[attr-defined]


def register_builtins(registry: "CommandRegistry") -> None:
  """Register all built-in commands on a CommandRegistry."""

  registry.register(HelpCommand())  # type: ignore[arg-type]
  registry.register(InfoCommand())  # type: ignore[arg-type]
  registry.register(ToolsCommand())  # type: ignore[arg-type]
  registry.register(ModelCommand())  # type: ignore[arg-type]
  registry.register(ClearCommand())  # type: ignore[arg-type]
  registry.register(HistoryCommand())  # type: ignore[arg-type]
  registry.register(ExportCommand())  # type: ignore[arg-type]
  registry.register(ResetCommand())  # type: ignore[arg-type]
  registry.register(QuitCommand())  # type: ignore[arg-type]
