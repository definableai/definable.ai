"""Slash command completion — filtered popup for / commands."""

from __future__ import annotations

from typing import List, Optional, Tuple

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class _CommandRow(Static):
  """A single row in the completion list."""

  DEFAULT_CSS = """
  _CommandRow {
    height: 1;
    padding: 0 1;
    width: 100%;
  }

  _CommandRow.highlighted {
    background: $accent-darken-2;
    color: $text;
    text-style: bold;
  }
  """


class SlashComplete(Widget):
  """Filtered completion popup for slash commands.

  Displays matching commands as the user types ``/`` in the prompt.
  Purely visual — no focus management. The parent screen forwards
  Up/Down/Tab/Enter/Escape events to navigate and select.
  """

  DEFAULT_CSS = """
  SlashComplete {
    dock: bottom;
    height: auto;
    max-height: 12;
    background: $surface;
    border-top: solid $primary-darken-2;
    border-bottom: solid $primary-darken-2;
    padding: 0;
    display: none;
  }

  SlashComplete .complete-header {
    height: 1;
    padding: 0 1;
    color: $text-disabled;
    text-style: italic;
  }

  SlashComplete Vertical {
    height: auto;
    max-height: 11;
  }
  """

  is_shown: reactive[bool] = reactive(False)

  def __init__(self) -> None:
    super().__init__()
    self._commands: List[Tuple[str, str, List[str]]] = []  # (name, description, aliases)
    self._filtered: List[Tuple[str, str, List[str]]] = []
    self._highlighted_index: int = 0
    self._rows: List[_CommandRow] = []
    self._query = ""

  def compose(self) -> ComposeResult:
    yield Static("Commands", classes="complete-header")
    yield Vertical(id="complete-list")

  def set_commands(self, commands: List[Tuple[str, str, List[str]]]) -> None:
    """Set the full list of available commands.

    Args:
      commands: List of (name, description, aliases) tuples.
    """
    self._commands = commands

  def show(self, query: str = "") -> None:
    """Show the completion popup with the given filter query."""
    self._query = query
    self._filter(query)
    self._highlighted_index = 0
    self._render_list()
    self.display = True
    self.is_shown = True

  def hide(self) -> None:
    """Hide the completion popup."""
    self.display = False
    self.is_shown = False

  def update_filter(self, query: str) -> None:
    """Update the filter with new query text."""
    self._query = query
    self._filter(query)
    self._highlighted_index = 0
    self._render_list()

  def move_up(self) -> None:
    """Move highlight up."""
    if self._filtered:
      self._highlighted_index = max(0, self._highlighted_index - 1)
      self._update_highlights()

  def move_down(self) -> None:
    """Move highlight down."""
    if self._filtered:
      self._highlighted_index = min(len(self._filtered) - 1, self._highlighted_index + 1)
      self._update_highlights()

  @property
  def selected_command(self) -> Optional[str]:
    """The currently highlighted command name, or None."""
    if self._filtered and 0 <= self._highlighted_index < len(self._filtered):
      return self._filtered[self._highlighted_index][0]
    return None

  @property
  def has_matches(self) -> bool:
    return len(self._filtered) > 0

  def _filter(self, query: str) -> None:
    """Filter commands by prefix match on name and aliases."""
    query = query.lower().strip()
    if not query:
      self._filtered = list(self._commands)
      return

    matches: List[Tuple[str, str, List[str]]] = []
    for name, desc, aliases in self._commands:
      # Check name starts with query
      if name.lower().startswith(query):
        matches.append((name, desc, aliases))
        continue
      # Check aliases
      for alias in aliases:
        if alias.lower().startswith(query):
          matches.append((name, desc, aliases))
          break
    self._filtered = matches

  def _render_list(self) -> None:
    """Re-render the command list."""
    try:
      container = self.query_one("#complete-list", Vertical)
    except Exception:
      return

    # Remove old rows
    container.remove_children()
    self._rows.clear()

    for i, (name, desc, aliases) in enumerate(self._filtered):
      alias_str = f" ({', '.join(aliases)})" if aliases else ""
      text = f"/{name}{alias_str}  \u2014  {desc}"
      row = _CommandRow(text)
      if i == self._highlighted_index:
        row.add_class("highlighted")
      self._rows.append(row)
      container.mount(row)

  def _update_highlights(self) -> None:
    """Update which row is highlighted."""
    for i, row in enumerate(self._rows):
      if i == self._highlighted_index:
        row.add_class("highlighted")
      else:
        row.remove_class("highlighted")
