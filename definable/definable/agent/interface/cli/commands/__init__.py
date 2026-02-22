"""Command registry and base protocol for CLI slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.cli.output import OutputManager
  from definable.agent.interface.session import InterfaceSession


@dataclass
class CommandContext:
  """Context passed to command handlers.

  Attributes:
    agent: The bound Agent instance.
    session: The current CLI session.
    output: The OutputManager for rendering.
    interface: The CLIInterface instance (for state mutation like reset).
  """

  agent: "Agent"
  session: "InterfaceSession"
  output: "OutputManager"
  interface: object  # CLIInterface — avoid circular import


@runtime_checkable
class BaseCommand(Protocol):
  """Protocol for CLI slash commands."""

  @property
  def name(self) -> str: ...

  @property
  def description(self) -> str: ...

  @property
  def aliases(self) -> List[str]: ...

  async def execute(self, args: str, context: CommandContext) -> None: ...


class CommandRegistry:
  """Maps command names and aliases to BaseCommand instances.

  Example::

      registry = CommandRegistry()
      registry.register(HelpCommand())
      cmd = registry.get("help")  # or registry.get("h")
  """

  def __init__(self) -> None:
    self._commands: Dict[str, BaseCommand] = {}
    self._all: List[BaseCommand] = []

  def register(self, command: BaseCommand) -> "CommandRegistry":
    """Register a command. Returns self for chaining."""
    self._commands[command.name] = command
    for alias in command.aliases:
      self._commands[alias] = command
    self._all.append(command)
    return self

  def get(self, name: str) -> Optional[BaseCommand]:
    """Look up a command by name or alias."""
    return self._commands.get(name)

  @property
  def all_commands(self) -> List[BaseCommand]:
    """All registered commands (no duplicates from aliases)."""
    return list(self._all)
