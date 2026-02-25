"""Search bar — inline conversation search with match navigation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static


class SearchBar(Widget):
  """Inline search bar for finding text within conversation blocks.

  Toggle visibility with ``show_search()`` / ``hide_search()``.
  Emits ``SearchChanged`` as the user types and ``SearchNavigate``
  when the user presses Enter or clicks the navigation arrows.
  """

  class SearchChanged(Message):
    """Emitted when the search query changes."""

    def __init__(self, query: str) -> None:
      super().__init__()
      self.query = query

  class SearchNavigate(Message):
    """Navigate to next or previous match."""

    def __init__(self, direction: int) -> None:
      super().__init__()
      self.direction = direction  # 1 = next, -1 = prev

  class SearchDismissed(Message):
    """Search bar was dismissed."""

  DEFAULT_CSS = """
  SearchBar {
    dock: top;
    height: 1;
    display: none;
    background: $surface;
  }

  SearchBar.active {
    display: block;
  }

  SearchBar Horizontal {
    height: 1;
    width: 100%;
  }

  SearchBar .search-label {
    width: auto;
    padding: 0 1;
    color: $accent;
    text-style: bold;
  }

  SearchBar Input {
    width: 1fr;
    height: 1;
    border: none;
    padding: 0 1;
    background: $surface-darken-1;
  }

  SearchBar .search-count {
    width: auto;
    padding: 0 1;
    color: $text-muted;
  }

  SearchBar .search-nav {
    width: 3;
    content-align: center middle;
    color: $accent;
  }

  SearchBar .search-nav:hover {
    background: $surface-darken-1;
  }

  SearchBar .search-close {
    width: 3;
    content-align: center middle;
    color: $text-disabled;
  }

  SearchBar .search-close:hover {
    color: $error;
  }
  """

  match_count: reactive[int] = reactive(0)
  current_match: reactive[int] = reactive(0)

  def __init__(self) -> None:
    super().__init__()
    self._input: Input | None = None

  def compose(self) -> ComposeResult:
    with Horizontal():
      yield Static("Find:", classes="search-label")
      self._input = Input(placeholder="Search...", id="search-input")
      yield self._input
      yield Static("", id="search-count", classes="search-count")
      yield Static("\u25b2", id="search-prev", classes="search-nav")
      yield Static("\u25bc", id="search-next", classes="search-nav")
      yield Static("\u2715", id="search-close", classes="search-close")

  def show_search(self) -> None:
    """Show the search bar and focus the input."""
    self.add_class("active")
    if self._input is not None:
      self._input.value = ""
      self._input.focus()

  def hide_search(self) -> None:
    """Hide the search bar and reset state."""
    self.remove_class("active")
    self.match_count = 0
    self.current_match = 0
    self._update_count_display()

  @property
  def is_active(self) -> bool:
    """Whether the search bar is currently visible."""
    return self.has_class("active")

  @property
  def search_query(self) -> str:
    """Current search query text."""
    if self._input is not None:
      return self._input.value
    return ""

  def set_match_info(self, current: int, total: int) -> None:
    """Update the match count display."""
    self.current_match = current
    self.match_count = total
    self._update_count_display()

  def _update_count_display(self) -> None:
    """Refresh the count label."""
    try:
      if self.match_count > 0:
        text = f"{self.current_match}/{self.match_count}"
      elif self.search_query:
        text = "No matches"
      else:
        text = ""
      self.query_one("#search-count", Static).update(text)
    except Exception:
      pass

  def on_input_changed(self, event: Input.Changed) -> None:
    """Forward input changes as search events."""
    self.post_message(self.SearchChanged(query=event.value))

  def on_input_submitted(self, event: Input.Submitted) -> None:
    """Enter navigates to the next match."""
    self.post_message(self.SearchNavigate(direction=1))

  def on_click(self, event: object) -> None:
    """Handle navigation and close button clicks."""
    target = getattr(event, "widget", None)
    if target is None:
      return
    target_id = getattr(target, "id", "")
    if target_id == "search-close":
      self.post_message(self.SearchDismissed())
    elif target_id == "search-prev":
      self.post_message(self.SearchNavigate(direction=-1))
    elif target_id == "search-next":
      self.post_message(self.SearchNavigate(direction=1))

  def on_key(self, event: object) -> None:
    """Escape dismisses the search bar."""
    key = getattr(event, "key", "")
    if key == "escape":
      self.post_message(self.SearchDismissed())
      if hasattr(event, "prevent_default"):
        event.prevent_default()  # type: ignore[union-attr]
