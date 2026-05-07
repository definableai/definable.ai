"""Renderer for streaming content deltas (RunContentEvent)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
  from rich.console import Console

  from definable.agent.interface.cli.config import CLIConfig
  from definable.run.base import BaseRunOutputEvent


class StreamingRenderer:
  """Renders RunContentEvent deltas inline as they arrive.

  Tracks whether content was streamed for a given run so that
  ``_send_response()`` can skip re-printing the full response.

  The ``streamed_run_id`` is set on the first content delta and
  cleared on RunStartedEvent (new run).
  """

  def __init__(self) -> None:
    self._streamed_run_id: Optional[str] = None
    self._needs_newline = False

  @property
  def streamed_run_id(self) -> Optional[str]:
    """The run_id of the run whose content was streamed, or None."""
    return self._streamed_run_id

  def reset(self) -> None:
    """Reset state for a new run."""
    if self._needs_newline:
      # Print newline to end the streamed content block
      print("")
      self._needs_newline = False
    self._streamed_run_id = None

  def handles(self, event: "BaseRunOutputEvent") -> bool:
    from definable.run.agent import RunContentEvent, RunStartedEvent

    return isinstance(event, (RunContentEvent, RunStartedEvent))

  def render(self, event: "BaseRunOutputEvent", console: "Console", config: "CLIConfig") -> None:
    from definable.run.agent import RunContentEvent, RunStartedEvent

    if isinstance(event, RunStartedEvent):
      self.reset()
      return

    if isinstance(event, RunContentEvent):
      if event.content is not None:
        content_str = str(event.content)
        if content_str:
          # Print raw to stdout for true streaming (no Rich markup)
          print(content_str, end="", flush=True)
          self._streamed_run_id = event.run_id
          self._needs_newline = True

  def finish(self) -> None:
    """Call after the run completes to add a trailing newline if needed."""
    if self._needs_newline:
      print("")
      self._needs_newline = False
