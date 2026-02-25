"""Textual messages — bridge between pipeline events and TUI widgets."""

from __future__ import annotations

from typing import Any, Dict, Optional

from textual.message import Message


# --- Agent response streaming ---


class StreamChunk(Message):
  """A chunk of streaming content from the agent."""

  def __init__(self, text: str, run_id: str) -> None:
    super().__init__()
    self.text = text
    self.run_id = run_id


class StreamComplete(Message):
  """Agent finished streaming content for a run."""

  def __init__(self, run_id: str) -> None:
    super().__init__()
    self.run_id = run_id


# --- Run lifecycle ---


class RunStarted(Message):
  """Agent run has started."""

  def __init__(self, run_id: str, input_text: str = "") -> None:
    super().__init__()
    self.run_id = run_id
    self.input_text = input_text


class RunCompleted(Message):
  """Agent run completed."""

  def __init__(
    self,
    run_id: str,
    *,
    content: str = "",
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    time_to_first_token: Optional[float] = None,
    total_time: Optional[float] = None,
  ) -> None:
    super().__init__()
    self.run_id = run_id
    self.content = content
    self.total_tokens = total_tokens
    self.prompt_tokens = prompt_tokens
    self.completion_tokens = completion_tokens
    self.time_to_first_token = time_to_first_token
    self.total_time = total_time


class RunError(Message):
  """Agent run errored."""

  def __init__(self, run_id: str, error: str) -> None:
    super().__init__()
    self.run_id = run_id
    self.error = error


# --- Tool calls ---


class ToolCallStarted(Message):
  """A tool call has started."""

  def __init__(self, tool_name: str, arguments: str = "", call_id: str = "") -> None:
    super().__init__()
    self.tool_name = tool_name
    self.arguments = arguments
    self.call_id = call_id


class ToolCallCompleted(Message):
  """A tool call has completed."""

  def __init__(
    self,
    tool_name: str,
    result: str = "",
    call_id: str = "",
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
  ) -> None:
    super().__init__()
    self.tool_name = tool_name
    self.result = result
    self.call_id = call_id
    self.duration_ms = duration_ms
    self.error = error


# --- Thinking/reasoning ---


class ThinkingStarted(Message):
  """Agent started reasoning."""

  def __init__(self, run_id: str = "") -> None:
    super().__init__()
    self.run_id = run_id


class ThinkingChunk(Message):
  """A chunk of reasoning content."""

  def __init__(self, text: str) -> None:
    super().__init__()
    self.text = text


class ThinkingCompleted(Message):
  """Agent finished reasoning."""

  def __init__(self, run_id: str = "") -> None:
    super().__init__()
    self.run_id = run_id


# --- Model calls ---


class ModelCallUpdate(Message):
  """Model call metrics update (turn counter)."""

  def __init__(
    self,
    turn: int,
    *,
    model_id: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
  ) -> None:
    super().__init__()
    self.turn = turn
    self.model_id = model_id
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens


# --- Knowledge & memory ---


class KnowledgeUpdate(Message):
  """Knowledge retrieval status update."""

  def __init__(self, status: str, doc_count: int = 0, duration_ms: float = 0) -> None:
    super().__init__()
    self.status = status  # "searching" | "complete"
    self.doc_count = doc_count
    self.duration_ms = duration_ms


class MemoryUpdate(Message):
  """Memory recall/update status."""

  def __init__(self, status: str, entry_count: int = 0, duration_ms: float = 0) -> None:
    super().__init__()
    self.status = status  # "recalling" | "recalled" | "updating" | "updated"
    self.entry_count = entry_count
    self.duration_ms = duration_ms


# --- Status bar ---


class StatusUpdate(Message):
  """Generic status bar update."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__()
    self.updates: Dict[str, Any] = kwargs


# --- User actions ---


class UserSubmitted(Message):
  """User submitted a prompt."""

  def __init__(self, text: str) -> None:
    super().__init__()
    self.text = text


class SlashCommandRequested(Message):
  """User entered a slash command."""

  def __init__(self, command: str, args: str = "") -> None:
    super().__init__()
    self.command = command
    self.args = args


# --- Slash completion ---


class ShowSlashComplete(Message):
  """Request to show/update the slash completion popup."""

  def __init__(self, query: str) -> None:
    super().__init__()
    self.query = query


class HideSlashComplete(Message):
  """Request to hide the slash completion popup."""


class AcceptSlashComplete(Message):
  """User accepted a slash completion (Tab/Enter on popup)."""


class NavigateSlashComplete(Message):
  """Navigate the slash completion popup (Up/Down)."""

  def __init__(self, direction: int) -> None:
    super().__init__()
    self.direction = direction  # -1 = up, 1 = down


# --- Search ---


class ToggleSearch(Message):
  """Toggle the conversation search bar."""


class SearchQueryChanged(Message):
  """Search query text changed."""

  def __init__(self, query: str) -> None:
    super().__init__()
    self.query = query


class SearchNavigateMatch(Message):
  """Navigate to next/previous search match."""

  def __init__(self, direction: int) -> None:
    super().__init__()
    self.direction = direction  # 1 = next, -1 = prev


class SearchDismiss(Message):
  """Dismiss the search bar."""
