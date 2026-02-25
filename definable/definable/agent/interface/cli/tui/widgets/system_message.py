"""System message block — displays command output and notifications."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class SystemMessage(Widget):
  """A system/command output message in the conversation.

  Used for displaying slash command output (/help, /info, /tools, etc.)
  and system notifications within the conversation flow.
  """

  DEFAULT_CSS = """
  SystemMessage {
    margin: 0 0 1 0;
    padding: 0 1;
    height: auto;
  }

  SystemMessage .system-label {
    width: 4;
    color: $text-disabled;
    text-style: bold italic;
  }

  SystemMessage .system-body {
    width: 1fr;
    height: auto;
    padding: 0 0 0 1;
  }

  SystemMessage .system-content {
    color: $text-muted;
  }
  """

  def __init__(self, content: str, label: str = "Sys") -> None:
    super().__init__()
    self._content = content
    self._label = label

  def compose(self) -> ComposeResult:
    with Vertical():
      yield Static(self._label, classes="system-label")
      yield Static(self._content, classes="system-content")

  @property
  def content(self) -> str:
    return self._content
