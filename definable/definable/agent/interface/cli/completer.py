"""Slash-command completer for prompt_toolkit integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from prompt_toolkit.completion import CompleteEvent, Completion, Completer
from prompt_toolkit.document import Document

if TYPE_CHECKING:
  from definable.agent.interface.cli.commands import CommandRegistry


class SlashCommandCompleter(Completer):
  """Provides tab/dropdown completions for CLI slash commands.

  Activates only when the input starts with the command prefix (default ``/``).
  Yields one ``Completion`` per registered command (name + description),
  plus one per alias.  Stops completing once a space is typed (user is
  entering arguments, not a command name).

  Args:
    registry: The live ``CommandRegistry`` — completions reflect
      commands registered at call time, so dynamically-added
      commands appear immediately.
    prefix: The command prefix character (default ``"/"``).
  """

  def __init__(self, registry: "CommandRegistry", *, prefix: str = "/") -> None:
    self._registry = registry
    self._prefix = prefix

  def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
    text = document.text_before_cursor

    # Only activate when the line starts with the prefix
    if not text.startswith(self._prefix):
      return

    # Stop completing after a space (user is typing args)
    if " " in text:
      return

    # The partial command name typed so far (without prefix)
    partial = text[len(self._prefix) :]

    # Build a set of already-yielded names to avoid duplicates
    seen: set[str] = set()

    for cmd in self._registry.all_commands:
      # Primary name
      full = f"{self._prefix}{cmd.name}"
      if cmd.name.startswith(partial) and cmd.name not in seen:
        seen.add(cmd.name)
        yield Completion(
          full,
          start_position=-len(text),
          display=full,
          display_meta=cmd.description,
        )

      # Aliases
      for alias in cmd.aliases:
        full_alias = f"{self._prefix}{alias}"
        if alias.startswith(partial) and alias not in seen:
          seen.add(alias)
          yield Completion(
            full_alias,
            start_position=-len(text),
            display=full_alias,
            display_meta=f"alias of {self._prefix}{cmd.name}",
          )
