"""User message block — displays what the user typed."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static


class UserMessage(Widget):
  """A user message in the conversation."""

  DEFAULT_CSS = """
  UserMessage {
    margin: 0 0 1 0;
    padding: 0 1;
    height: auto;
  }

  UserMessage .user-label {
    width: 4;
    color: $accent;
    text-style: bold;
  }

  UserMessage .user-content {
    width: 1fr;
  }
  """

  def __init__(self, text: str) -> None:
    super().__init__()
    self._text = text

  def compose(self) -> ComposeResult:
    with Horizontal():
      yield Static("You", classes="user-label")
      yield Static(self._text, classes="user-content")
