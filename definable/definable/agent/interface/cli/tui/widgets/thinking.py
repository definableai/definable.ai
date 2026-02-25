"""Thinking/reasoning display — compact, expandable inner monologue."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.reactive import var
from textual.widgets import RichLog, Static


class ThinkingHeader(Static):
  """Clickable header for the thinking block."""

  DEFAULT_CSS = """
  ThinkingHeader {
    width: 1fr;
    height: auto;
    padding: 0 1;

    &:hover {
      background: $surface;
    }
  }
  """


class ThinkingBlock(VerticalGroup):
  """Displays agent reasoning/thinking in a compact, expandable block.

  Uses RichLog for efficient streaming — each chunk is appended
  without re-rendering the entire content.
  """

  expanded: var[bool] = var(False, toggle_class="-expanded")

  DEFAULT_CSS = """
  ThinkingBlock {
    margin: 0 0 0 2;
    height: auto;
    layout: stream;
    border-left: thick $secondary-darken-2;

    #thinking-content {
      display: none;
      padding: 0 1 0 2;
      height: auto;
    }

    &.-expanded #thinking-content {
      display: block;
    }

    RichLog {
      color: $text-muted;
      text-style: italic;
      padding: 0;
      height: auto;
      max-height: 20;
      scrollbar-size: 1 1;
      background: transparent;
    }
  }
  """

  def __init__(self) -> None:
    super().__init__()
    self._content = ""
    self._log: RichLog | None = None
    self._header: ThinkingHeader | None = None
    self._finished = False
    self._pending_text = ""

  def compose(self) -> ComposeResult:
    self._header = ThinkingHeader("\u25b6 \U0001f4ad Thinking\u2026")
    yield self._header
    content_area = VerticalGroup(id="thinking-content")
    with content_area:
      self._log = RichLog(wrap=True, markup=False, auto_scroll=True, max_lines=200)
      yield self._log

  @on(events.Click, "ThinkingHeader")
  def _on_header_click(self, event: events.Click) -> None:
    event.stop()
    self.expanded = not self.expanded

  def watch_expanded(self) -> None:
    if self._header is None:
      return
    icon = "\u25bc" if self.expanded else "\u25b6"
    title = self._get_title()
    self._header.update(f"{icon} {title}")

  def _get_title(self) -> str:
    if self._finished:
      preview = self._content[:80].replace("\n", " ").strip()
      if len(self._content) > 80:
        preview += "\u2026"
      return f"\U0001f4ad Thought: {preview}" if preview else "\U0001f4ad Thought"
    return "\U0001f4ad Thinking\u2026"

  async def append_chunk(self, text: str) -> None:
    """Append a chunk of reasoning content."""
    self._content += text
    self._pending_text += text

    if self._log is not None:
      while "\n" in self._pending_text:
        line, self._pending_text = self._pending_text.split("\n", 1)
        self._log.write(line)

  async def finish(self) -> None:
    """Mark thinking as complete, update title."""
    self._finished = True

    if self._log is not None and self._pending_text:
      self._log.write(self._pending_text)
      self._pending_text = ""

    # Update header with summary
    if self._header is not None:
      icon = "\u25bc" if self.expanded else "\u25b6"
      self._header.update(f"{icon} {self._get_title()}")

  @property
  def content(self) -> str:
    return self._content
