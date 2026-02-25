"""System message block — displays command output and notifications."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.widgets import Markdown


class SystemMessage(VerticalGroup):
  """A system/command output message in the conversation.

  Used for displaying slash command output (/help, /info, /tools, etc.)
  and system notifications within the conversation flow.
  """

  DEFAULT_CSS = """
  SystemMessage {
    margin: 0 0 1 0;
    padding: 0 1;
    height: auto;
    border-left: thick $primary-darken-3;

    Markdown {
      margin: 0;
      padding: 0;
    }

    Markdown > MarkdownBlock:last-child {
      margin-bottom: 0;
    }
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
    yield Markdown(self._content, classes="system-content")

  @property
  def content(self) -> str:
    return self._content
