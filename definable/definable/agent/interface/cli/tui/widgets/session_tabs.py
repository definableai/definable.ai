"""Session tabs — horizontal tab bar for switching conversations."""

from __future__ import annotations

import contextlib
from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class _Tab(Static):
  """A single session tab."""

  DEFAULT_CSS = """
  _Tab {
    height: 1;
    padding: 0 2;
    width: auto;
    color: $text-muted;
  }

  _Tab:hover {
    background: $surface-darken-1;
  }

  _Tab.active {
    color: $text;
    text-style: bold;
    background: $surface;
    border-bottom: tall $accent;
  }
  """

  def __init__(self, label: str, session_key: str) -> None:
    super().__init__(label)
    self.session_key = session_key


class SessionTabs(Widget):
  """Horizontal tab bar for switching between conversation sessions.

  Displays active sessions as clickable tabs. The active tab is highlighted.
  Emits ``SessionTabs.TabSelected`` when the user clicks a tab.
  """

  class TabSelected(Message):
    """Fired when a tab is selected."""

    def __init__(self, session_key: str) -> None:
      super().__init__()
      self.session_key = session_key

  DEFAULT_CSS = """
  SessionTabs {
    dock: top;
    height: 1;
    background: $primary-darken-3;
    display: none;
  }

  SessionTabs Horizontal {
    height: 1;
    width: 100%;
  }

  SessionTabs .tabs-label {
    width: auto;
    padding: 0 1;
    color: $text-disabled;
    text-style: italic;
  }
  """

  active_key: reactive[str] = reactive("")

  def __init__(self) -> None:
    super().__init__()
    self._tabs: List[_Tab] = []
    self._sessions: List[tuple[str, str]] = []  # (key, label)

  def compose(self) -> ComposeResult:
    with Horizontal():
      yield Static("Sessions:", classes="tabs-label")

  def set_sessions(self, sessions: List[tuple[str, str]], active_key: str = "") -> None:
    """Update the tab list.

    Args:
      sessions: List of (session_key, display_label) tuples.
      active_key: The currently active session key.
    """
    self._sessions = sessions
    self.active_key = active_key
    self._render_tabs()

    # Show tabs only when there are multiple sessions
    self.display = len(sessions) > 1

  def _render_tabs(self) -> None:
    """Re-render all tabs."""
    try:
      container = self.query_one("Horizontal", Horizontal)
    except Exception:
      return

    # Remove old tabs (keep the label)
    for tab in self._tabs:
      with contextlib.suppress(Exception):
        tab.remove()
    self._tabs.clear()

    for key, label in self._sessions:
      tab = _Tab(label, session_key=key)
      if key == self.active_key:
        tab.add_class("active")
      self._tabs.append(tab)
      container.mount(tab)

  def watch_active_key(self, value: str) -> None:
    """Update tab highlighting when active key changes."""
    for tab in self._tabs:
      if tab.session_key == value:
        tab.add_class("active")
      else:
        tab.remove_class("active")

  def on_click(self, event: object) -> None:
    """Handle click on a tab — find clicked _Tab and emit selection."""
    # Walk up from the click target to find a _Tab
    target = getattr(event, "widget", None)
    while target is not None:
      if isinstance(target, _Tab):
        if target.session_key != self.active_key:
          self.post_message(self.TabSelected(session_key=target.session_key))
        return
      target = getattr(target, "parent", None)

  @property
  def session_count(self) -> int:
    return len(self._sessions)

  @property
  def active_session(self) -> Optional[str]:
    return self.active_key or None
