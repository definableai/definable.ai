"""Tool call widget — compact, expandable tool execution display.

Uses CSS class toggling instead of Collapsible for proper height: auto
behavior in layout: stream containers (following Toad's pattern).
"""

from __future__ import annotations

import contextlib
import re
from typing import Optional

from textual import events, on
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.reactive import var
from textual.widgets import RichLog, Static


# Unicode status indicators
_ICON_PENDING = "\u25b6"  # ▶
_ICON_SUCCESS = "\u2714"  # ✔
_ICON_ERROR = "\u2717"  # ✗

# Regex for ANSI escape sequences
_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")

# Heuristic: text looks like a unified diff
_DIFF_MARKERS = ("--- ", "+++ ", "@@ ", "diff --git")


def _has_ansi(text: str) -> bool:
  return bool(_ANSI_RE.search(text))


def _looks_like_diff(text: str) -> bool:
  lines = text.split("\n", 10)
  markers = sum(1 for line in lines if any(line.startswith(m) for m in _DIFF_MARKERS))
  return markers >= 2


class ToolCallHeader(Static):
  """Clickable header for the tool call block."""

  DEFAULT_CSS = """
  ToolCallHeader {
    width: 1fr;
    height: auto;
    padding: 0 1;

    &:hover {
      background: $surface;
    }
  }
  """


class ToolCallBlock(VerticalGroup):
  """Displays a tool call with name, arguments, result, and timing.

  Shows as a compact block with a clickable header:
  - ▶ tool_name while running
  - ✔ tool_name 150ms on success
  - ✗ tool_name on error

  Content area (args + result) is hidden by default, toggled on click.

  Auto-expand behavior is controlled by ``tools_expand``:
  - "always"  — expand on start and on completion
  - "success" — expand only on successful completion
  - "fail"    — expand only on error
  - "both"    — expand on success or error
  - "never"   — always stay collapsed
  """

  expanded: var[bool] = var(False, toggle_class="-expanded")
  has_content: var[bool] = var(False, toggle_class="-has-content")

  DEFAULT_CSS = """
  ToolCallBlock {
    margin: 0 0 0 2;
    height: auto;
    layout: stream;
    border-left: thick $primary-darken-2;

    #tool-content {
      display: none;
      padding: 0 1 0 2;
      height: auto;
    }

    &.-has-content.-expanded #tool-content {
      display: block;
    }

    .tool-section-label {
      color: $text-muted;
      text-style: bold;
      padding: 0 0 0 0;
    }

    .tool-args {
      color: $text-muted;
      padding: 0 0 0 1;
    }

    .tool-result {
      padding: 0 0 0 1;
    }

    .tool-result-log {
      padding: 0 0 0 1;
      height: auto;
      max-height: 20;
      background: transparent;
      scrollbar-size: 1 1;
    }

    .tool-error {
      color: $error;
      padding: 0 0 0 1;
    }

    .tool-separator {
      color: $text-muted;
      padding: 0;
    }
  }
  """

  def __init__(
    self,
    tool_name: str,
    arguments: str = "",
    call_id: str = "",
    tools_expand: str = "success",
  ) -> None:
    super().__init__()
    self.tool_name = tool_name
    self.arguments = arguments
    self.call_id = call_id
    self.tools_expand = tools_expand
    self._result: Optional[str] = None
    self._error: Optional[str] = None
    self._duration_ms: Optional[float] = None
    self._header: ToolCallHeader | None = None
    self._content_area: VerticalGroup | None = None
    self._completed = False

  def compose(self) -> ComposeResult:
    expand_icon = "\u25bc " if self.expanded else "\u25b6 "
    self._header = ToolCallHeader(f"{expand_icon}\U0001f527 {self.tool_name} \u231b")
    yield self._header
    self._content_area = VerticalGroup(id="tool-content")
    with self._content_area:
      if self.arguments:
        yield Static("Arguments", classes="tool-section-label")
        display_args = self.arguments[:500] + "\u2026" if len(self.arguments) > 500 else self.arguments
        yield Static(display_args, classes="tool-args")
    self.has_content = bool(self.arguments)

  @on(events.Click, "ToolCallHeader")
  def _on_header_click(self, event: events.Click) -> None:
    event.stop()
    if self.has_content:
      self.expanded = not self.expanded
    else:
      self.app.bell()

  def watch_expanded(self) -> None:
    """Update the header icon when expand state changes."""
    if self._header is None:
      return
    # Rebuild header text with current icon
    icon = "\u25bc" if self.expanded else "\u25b6"
    status = self._get_status_text()
    self._header.update(f"{icon} {status}")

  def _get_status_text(self) -> str:
    """Build the status text for the header."""
    timing = f" {self._duration_ms:.0f}ms" if self._duration_ms is not None else ""
    if self._error:
      return f"\U0001f527 {self.tool_name}{timing} {_ICON_ERROR}"
    if self._completed:
      return f"\U0001f527 {self.tool_name}{timing} {_ICON_SUCCESS}"
    return f"\U0001f527 {self.tool_name} \u231b"

  def complete(
    self,
    result: str = "",
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
  ) -> None:
    """Mark the tool call as completed."""
    self._completed = True
    self._result = result
    self._error = error
    self._duration_ms = duration_ms

    # Update header
    icon = "\u25bc" if self.expanded else "\u25b6"
    status = self._get_status_text()
    if self._header is not None:
      self._header.update(f"{icon} {status}")

    # Add result to content area
    if self._content_area is not None:
      with contextlib.suppress(Exception):
        if error:
          display = error[:1000] + "\u2026" if len(error) > 1000 else error
          self._content_area.mount(
            Static("\u2500" * 30, classes="tool-separator"),
            Static("Error", classes="tool-section-label"),
            Static(display, classes="tool-error"),
          )
          self.has_content = True
        elif result:
          display = result[:2000] + "\u2026" if len(result) > 2000 else result
          self._content_area.mount(
            Static("\u2500" * 30, classes="tool-separator"),
            Static("Result", classes="tool-section-label"),
            self._render_result(display),
          )
          self.has_content = True

    # Auto-expand logic
    should_expand = False
    if self.tools_expand in ("always", "both"):
      should_expand = True
    elif self.tools_expand == "success" and not error:
      should_expand = True
    elif self.tools_expand == "fail" and error:
      should_expand = True

    if should_expand:
      self.expanded = True

  def _render_result(self, text: str) -> Static | RichLog:
    """Create the appropriate widget for a tool result."""
    from rich.text import Text

    if _has_ansi(text):
      log = RichLog(wrap=True, max_lines=100, classes="tool-result-log")
      log.write(Text.from_ansi(text))
      return log

    if _looks_like_diff(text):
      log = RichLog(wrap=True, max_lines=100, classes="tool-result-log")
      for line in text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
          log.write(Text(line, style="green"))
        elif line.startswith("-") and not line.startswith("---"):
          log.write(Text(line, style="red"))
        elif line.startswith("@@"):
          log.write(Text(line, style="cyan"))
        elif line.startswith("diff "):
          log.write(Text(line, style="bold"))
        else:
          log.write(Text(line))
      return log

    return Static(text, classes="tool-result")

  @property
  def is_completed(self) -> bool:
    return self._completed

  @property
  def is_error(self) -> bool:
    return self._error is not None
