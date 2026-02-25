"""User message block — displays what the user typed."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.widgets import Markdown, Static


class UserMessage(HorizontalGroup):
  """A user message in the conversation.

  Uses HorizontalGroup with a prompt indicator and Markdown content,
  matching the Toad TUI pattern for proper layout in VerticalScroll.
  """

  DEFAULT_CSS = """
  UserMessage {
    border-left: blank $secondary;
    background: $secondary 15%;
    padding: 1 1 1 0;
    margin: 1 1 1 0;
    height: auto;

    Markdown {
      padding: 0 2 0 0;
    }

    Markdown > MarkdownBlock:last-child {
      margin-bottom: 0;
    }
  }

  UserMessage .user-indicator {
    width: auto;
    padding: 0 1;
    color: $secondary;
    text-style: bold;
  }
  """

  def __init__(self, text: str) -> None:
    super().__init__()
    self._text = text

  def compose(self) -> ComposeResult:
    yield Static("\u276f", classes="user-indicator")
    yield Markdown(self._text)
