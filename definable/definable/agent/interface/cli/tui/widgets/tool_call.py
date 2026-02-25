"""Tool call widget — collapsible tool execution display.

Supports ANSI-colored output and diff highlighting in results.
"""

from __future__ import annotations

import contextlib
import re
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Collapsible, RichLog, Static


# Unicode status indicators
_ICON_RUNNING = "\u25b6"  # ▶
_ICON_SUCCESS = "\u2713"  # ✓
_ICON_ERROR = "\u2717"  # ✗

# Regex for ANSI escape sequences
_ANSI_RE = re.compile(r"\x1b\[[\d;]*m")

# Heuristic: text looks like a unified diff
_DIFF_MARKERS = ("--- ", "+++ ", "@@ ", "diff --git")


def _has_ansi(text: str) -> bool:
  """Check if text contains ANSI escape sequences."""
  return bool(_ANSI_RE.search(text))


def _looks_like_diff(text: str) -> bool:
  """Heuristic check for unified diff format."""
  lines = text.split("\n", 10)
  markers = sum(1 for line in lines if any(line.startswith(m) for m in _DIFF_MARKERS))
  return markers >= 2


class ToolCallBlock(Widget):
  """Displays a tool call with name, arguments, result, and timing.

  Shows as a collapsible block with a status indicator:
  - ▶ while running
  - ✓ on success
  - ✗ on error

  Results with ANSI escape sequences are rendered with colors.
  Results that look like unified diffs get syntax coloring.

  Auto-expand behavior is controlled by ``tools_expand``:
  - "always"  — expand on start and on completion
  - "success" — expand only on successful completion
  - "fail"    — expand only on error
  - "both"    — expand on success or error (same as always-on-complete)
  - "never"   — always stay collapsed
  """

  DEFAULT_CSS = """
  ToolCallBlock {
    margin: 0 0 0 5;
    height: auto;
  }

  ToolCallBlock Collapsible {
    padding: 0;
    border-top: none;
    border-bottom: none;
  }

  ToolCallBlock .tool-section-label {
    color: $text-disabled;
    text-style: bold;
    padding: 0 1 0 1;
  }

  ToolCallBlock .tool-args {
    color: $text-muted;
    padding: 0 1 0 2;
  }

  ToolCallBlock .tool-result {
    padding: 0 1 0 2;
  }

  ToolCallBlock .tool-result-log {
    padding: 0 1 0 2;
    height: auto;
    max-height: 30;
    background: transparent;
    scrollbar-size: 1 1;
  }

  ToolCallBlock .tool-error {
    color: $error;
    padding: 0 1 0 2;
  }

  ToolCallBlock .tool-separator {
    color: $text-disabled;
    padding: 0 1;
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
    self._collapsible: Collapsible | None = None
    self._body: Vertical | None = None
    self._completed = False

  def compose(self) -> ComposeResult:
    title = f"{_ICON_RUNNING} {self.tool_name}"
    start_expanded = self.tools_expand == "always"
    self._collapsible = Collapsible(title=title, collapsed=not start_expanded)
    with self._collapsible:
      self._body = Vertical()
      with self._body:
        if self.arguments:
          yield Static("Arguments", classes="tool-section-label")
          display_args = self.arguments
          if len(display_args) > 500:
            display_args = display_args[:500] + "\u2026"
          yield Static(display_args, classes="tool-args")
        else:
          yield Static("No arguments", classes="tool-args")

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

    timing = f" {duration_ms:.0f}ms" if duration_ms is not None else ""

    # Update title with status icon and timing
    if self._collapsible is not None:
      if error:
        self._collapsible.title = f"{_ICON_ERROR} {self.tool_name} {timing}"
      else:
        self._collapsible.title = f"{_ICON_SUCCESS} {self.tool_name} {timing}"

    # Auto-expand logic
    should_expand = False
    if self.tools_expand == "always" or self.tools_expand == "both":
      should_expand = True
    elif self.tools_expand == "success" and not error:
      should_expand = True
    elif self.tools_expand == "fail" and error:
      should_expand = True

    if should_expand and self._collapsible is not None:
      self._collapsible.collapsed = False

    # Update body — add result section below arguments
    if self._body is not None:
      with contextlib.suppress(Exception):
        if error:
          separator = Static("\u2500" * 40, classes="tool-separator")
          label = Static("Error", classes="tool-section-label")
          display = error
          if len(display) > 1000:
            display = display[:1000] + "\u2026"
          content = Static(display, classes="tool-error")
          self._body.mount(separator, label, content)
        elif result:
          separator = Static("\u2500" * 40, classes="tool-separator")
          label = Static("Result", classes="tool-section-label")
          display = result
          if len(display) > 2000:
            display = display[:2000] + "\u2026"
          result_widget = self._render_result(display)
          self._body.mount(separator, label, result_widget)

  def _render_result(self, text: str) -> Widget:
    """Create the appropriate widget for a tool result.

    - ANSI text → RichLog with ``Text.from_ansi()``
    - Diff text → RichLog with color-coded lines
    - Plain text → Static (lightweight)
    """
    from rich.text import Text

    if _has_ansi(text):
      log = RichLog(wrap=True, max_lines=200, classes="tool-result-log")
      log.write(Text.from_ansi(text))
      return log

    if _looks_like_diff(text):
      log = RichLog(wrap=True, max_lines=200, classes="tool-result-log")
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
