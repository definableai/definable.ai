"""Footer bar — context-sensitive keybinding hints."""

from __future__ import annotations

import contextlib

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


# Mode-specific keybinding hints
_HINTS_IDLE = "F1 Help \u2502 /cmd Commands \u2502 Ctrl+F Search \u2502 Ctrl+T New Session \u2502 Ctrl+L Clear \u2502 Ctrl+Q Quit"
_HINTS_RUNNING = "\u23f3 Agent running\u2026 \u2502 Ctrl+C Cancel \u2502 Ctrl+C\u00d72 Quit"
_HINTS_SEARCHING = "Enter Next \u2502 Shift+Enter Prev \u2502 \u2191\u2193 Navigate \u2502 Esc Close"


class FooterBar(Widget):
  """Bottom hint bar showing context-sensitive keyboard shortcuts.

  Switches between three modes:
  - ``idle``      — default keybindings
  - ``running``   — cancel/quit hints
  - ``searching`` — search navigation hints
  """

  DEFAULT_CSS = """
  FooterBar {
    dock: bottom;
    height: 1;
    background: $primary-darken-3;
    color: $text-disabled;
  }

  FooterBar Static {
    width: 100%;
    height: 1;
    padding: 0 1;
    text-overflow: ellipsis;
  }
  """

  mode: reactive[str] = reactive("idle")

  def compose(self) -> ComposeResult:
    yield Static(_HINTS_IDLE, id="footer-hints")

  def watch_mode(self, value: str) -> None:
    """Update hint text when mode changes."""
    if value == "running":
      text = _HINTS_RUNNING
    elif value == "searching":
      text = _HINTS_SEARCHING
    else:
      text = _HINTS_IDLE
    with contextlib.suppress(Exception):
      self.query_one("#footer-hints", Static).update(text)
