"""Prompt widget — input textarea for user messages."""

from __future__ import annotations

import contextlib

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup
from textual.events import Key
from textual.timer import Timer
from textual.widgets import Static, TextArea

from definable.agent.interface.cli.tui.messages import (
  AcceptSlashComplete,
  HideSlashComplete,
  NavigateSlashComplete,
  ShowSlashComplete,
  SlashCommandRequested,
  UserSubmitted,
)

# Braille spinner frames for running indicator
_SPINNER_FRAMES = ["\u28cb", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]


class PromptInput(TextArea):
  """Text input area with submit-on-enter behavior.

  Enter submits the prompt. Shift+Enter inserts a newline.
  Slash commands (/) are detected and routed separately.

  Features:
  - Input history (Up/Down arrows)
  - Slash command detection → triggers completion popup
  - Tab to accept slash completion
  """

  BINDINGS = [
    Binding("enter", "submit", "Submit", show=False, priority=True),
    Binding("shift+enter,ctrl+j", "newline", "New line", show=False, priority=True),
    Binding("escape", "blur", "Unfocus", show=False),
  ]

  DEFAULT_CSS = """
  PromptInput {
    width: 1fr;
    height: auto;
    min-height: 1;
    max-height: 50vh;
    border: none;
    padding: 0 1 0 0;
    background: transparent;
  }

  PromptInput:focus {
    border: none;
  }

  PromptInput:blur {
    text-opacity: 50%;
  }
  """

  def __init__(self) -> None:
    super().__init__(
      language=None,
      theme="css",
      soft_wrap=True,
      tab_behavior="indent",
      show_line_numbers=False,
    )
    self._enabled = True
    self._history: list[str] = []
    self._history_index: int = -1
    self._history_temp: str = ""  # stores current input when navigating history
    self._slash_completing = False

  def action_newline(self) -> None:
    """Insert a newline (Shift+Enter or Ctrl+J)."""
    self.insert("\n")

  def action_submit(self) -> None:
    """Submit the current text."""
    if not self._enabled:
      return
    text = self.text.strip()
    if not text:
      return

    # If slash completion is active, accept it instead of submitting
    if self._slash_completing:
      self.post_message(AcceptSlashComplete())
      return

    # Add to history
    if text and (not self._history or self._history[-1] != text):
      self._history.append(text)
    self._history_index = -1

    # Check for slash commands
    if text.startswith("/"):
      parts = text[1:].split(maxsplit=1)
      cmd = parts[0].lower() if parts else ""
      args = parts[1] if len(parts) > 1 else ""
      self.post_message(SlashCommandRequested(command=cmd, args=args))
    else:
      self.post_message(UserSubmitted(text=text))

    self.clear()
    self.post_message(HideSlashComplete())

  def on_key(self, event: Key) -> None:
    """Handle key events for history and completion navigation."""
    # Tab: accept slash completion
    if event.key == "tab" and self._slash_completing:
      event.prevent_default()
      self.post_message(AcceptSlashComplete())
      return

    # Escape: dismiss slash completion
    if event.key == "escape" and self._slash_completing:
      event.prevent_default()
      self.post_message(HideSlashComplete())
      self._slash_completing = False
      return

    # Up/Down: navigate history or completion
    if event.key == "up":
      if self._slash_completing:
        event.prevent_default()
        self.post_message(NavigateSlashComplete(direction=-1))
        return
      # History navigation (only on first line)
      row, _ = self.cursor_location
      if row == 0:
        event.prevent_default()
        self._history_up()
        return

    if event.key == "down":
      if self._slash_completing:
        event.prevent_default()
        self.post_message(NavigateSlashComplete(direction=1))
        return
      # History navigation (only on last line)
      row, _ = self.cursor_location
      lines = self.text.split("\n")
      if row >= len(lines) - 1:
        event.prevent_default()
        self._history_down()
        return

  def on_text_area_changed(self, event: TextArea.Changed) -> None:
    """Detect slash command input and trigger completion."""
    text = self.text
    if text.startswith("/") and "\n" not in text:
      # Extract the partial command (after /)
      query = text[1:].split()[0] if text[1:].strip() else ""
      # Only show completion if no space yet (still typing command name)
      if " " not in text[1:]:
        self._slash_completing = True
        self.post_message(ShowSlashComplete(query=query))
      else:
        self._slash_completing = False
        self.post_message(HideSlashComplete())
    else:
      if self._slash_completing:
        self._slash_completing = False
        self.post_message(HideSlashComplete())

  def set_text(self, text: str) -> None:
    """Set the input text and move cursor to end."""
    self.clear()
    self.insert(text)

  def _history_up(self) -> None:
    """Navigate to previous history entry."""
    if not self._history:
      return
    if self._history_index == -1:
      # Save current input before navigating
      self._history_temp = self.text
      self._history_index = len(self._history) - 1
    elif self._history_index > 0:
      self._history_index -= 1
    else:
      return  # Already at oldest

    self.set_text(self._history[self._history_index])

  def _history_down(self) -> None:
    """Navigate to next history entry."""
    if self._history_index == -1:
      return  # Not navigating history

    if self._history_index < len(self._history) - 1:
      self._history_index += 1
      self.set_text(self._history[self._history_index])
    else:
      # Restore the text that was being typed
      self._history_index = -1
      self.set_text(self._history_temp)

  def set_enabled(self, enabled: bool) -> None:
    """Enable or disable input."""
    self._enabled = enabled
    self.read_only = not enabled

  @property
  def input_history(self) -> list[str]:
    """The input history list."""
    return list(self._history)


class PromptContainer(VerticalGroup):
  """Wrapper around the prompt input with focus-aware border."""

  DEFAULT_CSS = """
  PromptContainer {
    height: auto;
    border: tall transparent;
    margin: 0 0 1 0;

    &:focus-within {
      border: tall $secondary;
    }

    #text-prompt {
      height: auto;
    }

    #prompt-ind {
      width: auto;
      padding: 0 1;
      text-opacity: 30%;
    }

    &:focus-within #prompt-ind {
      text-opacity: 100%;
    }
  }
  """


class Prompt(VerticalGroup):
  """Full prompt bar with indicator and input area."""

  DEFAULT_CSS = """
  Prompt {
    dock: bottom;
    height: auto;
    padding: 0;
  }
  """

  def __init__(self, indicator: str = "\u276f") -> None:
    super().__init__()
    self._indicator = indicator
    self._input: PromptInput | None = None
    self._spinner_timer: Timer | None = None
    self._spinner_index = 0
    self._running = False

  def compose(self) -> ComposeResult:
    with PromptContainer(id="prompt-container"):
      with HorizontalGroup(id="text-prompt"):
        yield Static(self._indicator, id="prompt-ind", classes="prompt-indicator")
        self._input = PromptInput()
        yield self._input

  def on_mount(self) -> None:
    """Focus the input on mount."""
    if self._input is not None:
      self._input.focus()

  def focus_input(self) -> None:
    """Focus the prompt input."""
    if self._input is not None:
      self._input.focus()

  def set_enabled(self, enabled: bool) -> None:
    """Enable or disable input."""
    if self._input is not None:
      self._input.set_enabled(enabled)

  def set_running(self, running: bool) -> None:
    """Set running state — shows animated spinner when True."""
    self._running = running
    if self._input is not None:
      self._input.set_enabled(not running)
    if running:
      self._spinner_index = 0
      self._spinner_timer = self.set_interval(0.1, self._advance_spinner)
      self._update_indicator_running()
    else:
      if self._spinner_timer is not None:
        self._spinner_timer.stop()
        self._spinner_timer = None
      self._restore_indicator()

  def _advance_spinner(self) -> None:
    """Advance the spinner to the next frame."""
    self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
    self._update_indicator_running()

  def _update_indicator_running(self) -> None:
    """Update the indicator with the current spinner frame."""
    frame = _SPINNER_FRAMES[self._spinner_index]
    with contextlib.suppress(Exception):
      widget = self.query_one("#prompt-ind", Static)
      widget.update(f" {frame} ")
      widget.remove_class("prompt-ready")
      widget.add_class("prompt-running")

  def _restore_indicator(self) -> None:
    """Restore the indicator to the normal prompt string."""
    with contextlib.suppress(Exception):
      widget = self.query_one("#prompt-ind", Static)
      widget.update(self._indicator)
      widget.remove_class("prompt-running")
      widget.add_class("prompt-ready")

  def set_text(self, text: str) -> None:
    """Set the input text."""
    if self._input is not None:
      self._input.set_text(text)

  @property
  def is_running(self) -> bool:
    """Whether the prompt is in running state."""
    return self._running

  @property
  def input_widget(self) -> PromptInput | None:
    """The underlying PromptInput widget."""
    return self._input
