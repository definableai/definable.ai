"""Thinking/reasoning display — collapsible inner monologue."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Collapsible, RichLog


class ThinkingBlock(Widget):
  """Displays agent reasoning/thinking in a collapsible block.

  Uses RichLog for efficient streaming — each chunk is appended
  without re-rendering the entire content. Word wrap is enabled
  for readable reasoning output.
  """

  DEFAULT_CSS = """
  ThinkingBlock {
    margin: 0 0 0 5;
    height: auto;
  }

  ThinkingBlock Collapsible {
    padding: 0;
    border-top: none;
    border-bottom: none;
  }

  ThinkingBlock RichLog {
    color: $text-muted;
    text-style: italic;
    padding: 0 1;
    height: auto;
    max-height: 20;
    scrollbar-size: 1 1;
  }
  """

  def __init__(self) -> None:
    super().__init__()
    self._content = ""
    self._log: RichLog | None = None
    self._collapsible: Collapsible | None = None
    self._finished = False
    self._pending_text = ""

  def compose(self) -> ComposeResult:
    self._collapsible = Collapsible(title="\u2026 Thinking", collapsed=True)
    with self._collapsible:
      self._log = RichLog(wrap=True, markup=False, auto_scroll=True, max_lines=200)
      yield self._log

  async def append_chunk(self, text: str) -> None:
    """Append a chunk of reasoning content."""
    self._content += text
    self._pending_text += text

    # Write complete lines to the log for clean rendering
    if self._log is not None:
      while "\n" in self._pending_text:
        line, self._pending_text = self._pending_text.split("\n", 1)
        self._log.write(line)

  async def finish(self) -> None:
    """Mark thinking as complete, update title."""
    self._finished = True

    # Flush any remaining text
    if self._log is not None and self._pending_text:
      self._log.write(self._pending_text)
      self._pending_text = ""

    if self._collapsible is not None:
      # Show a brief summary in the title
      preview = self._content[:80].replace("\n", " ").strip()
      if len(self._content) > 80:
        preview += "\u2026"
      self._collapsible.title = f"Thought: {preview}" if preview else "Thought"

  @property
  def content(self) -> str:
    return self._content
