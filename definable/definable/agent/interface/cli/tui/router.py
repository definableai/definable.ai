"""Event router — bridges pipeline events to Textual messages."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from definable.utils.log import log_debug, log_warning

from definable.agent.interface.cli.tui.messages import (
  KnowledgeUpdate,
  MemoryUpdate,
  ModelCallUpdate,
  RunCompleted,
  RunError,
  RunStarted,
  StreamChunk,
  ThinkingChunk,
  ThinkingCompleted,
  ThinkingStarted,
  ToolCallCompleted,
  ToolCallStarted,
)

if TYPE_CHECKING:
  from textual.app import App

  from definable.run.base import BaseRunOutputEvent


class EventRouter:
  """Converts pipeline events into Textual messages.

  Subscribes to the agent's pipeline EventStream and posts
  corresponding Textual messages to the app. The app's screen
  handles these messages to update widgets.

  All event processing is wrapped in error handling so a malformed
  event never crashes the UI.
  """

  def __init__(self, app: "App") -> None:
    self._app = app
    self._turn_count = 0
    self._run_start_time: Optional[float] = None
    self._first_token_time: Optional[float] = None
    self._total_tokens = 0
    self._streamed_run_id: Optional[str] = None

  def _post(self, message) -> None:
    """Post a Textual message to the active screen.

    Messages must be posted to the screen (not the app) so that
    ``@on()`` handlers on ``MainScreen`` receive them.
    """
    screen = self._app.screen
    screen.post_message(message)

  def handle(self, event: "BaseRunOutputEvent") -> None:
    """Route a pipeline event to the appropriate Textual message.

    This is a sync handler — called directly by EventStream.
    Posts messages to the active screen's message queue.
    """
    try:
      self._dispatch(event)
    except Exception:
      event_type = type(event).__name__
      log_warning(f"TUI EventRouter: failed to process {event_type}")
      log_debug(f"TUI EventRouter error for {event_type}", exc_info=True)

  def _dispatch(self, event: "BaseRunOutputEvent") -> None:
    """Inner dispatch — separated so handle() can catch exceptions."""
    from definable.agent.events import (
      KnowledgeRetrievalCompletedEvent,
      KnowledgeRetrievalStartedEvent,
      MemoryRecallCompletedEvent,
      MemoryRecallStartedEvent,
      MemoryUpdateCompletedEvent,
      MemoryUpdateStartedEvent,
      ModelCallCompletedEvent,
      ModelCallStartedEvent,
      ReasoningCompletedEvent,
      ReasoningContentDeltaEvent,
      ReasoningStartedEvent,
      ReasoningStepEvent,
      RunCancelledEvent,
      RunCompletedEvent as PipelineRunCompleted,
      RunContentEvent,
      RunErrorEvent,
      RunStartedEvent as PipelineRunStarted,
      ToolCallCompletedEvent as PipelineToolCompleted,
      ToolCallErrorEvent,
      ToolCallStartedEvent as PipelineToolStarted,
    )

    # --- Run lifecycle ---
    if isinstance(event, PipelineRunStarted):
      self._turn_count = 0
      self._run_start_time = time.monotonic()
      self._first_token_time = None
      self._total_tokens = 0
      run_id = getattr(event, "run_id", "")
      input_text = getattr(event, "input", "")
      if isinstance(input_text, dict):
        input_text = str(input_text)
      self._streamed_run_id = run_id
      self._post(RunStarted(run_id=run_id, input_text=str(input_text or "")))
      return

    if isinstance(event, PipelineRunCompleted):
      run_id = getattr(event, "run_id", "")
      content = getattr(event, "content", "") or ""
      metrics = getattr(event, "metrics", None)

      total_tokens = 0
      ttft: Optional[float] = None
      total_time: Optional[float] = None

      if metrics is not None:
        total_tokens = getattr(metrics, "total_tokens", 0) or 0

      if self._first_token_time is not None and self._run_start_time is not None:
        ttft = (self._first_token_time - self._run_start_time) * 1000

      if self._run_start_time is not None:
        total_time = (time.monotonic() - self._run_start_time) * 1000

      self._post(
        RunCompleted(
          run_id=run_id,
          content=str(content),
          total_tokens=total_tokens,
          time_to_first_token=ttft,
          total_time=total_time,
        )
      )
      self._streamed_run_id = None
      return

    if isinstance(event, RunErrorEvent):
      run_id = getattr(event, "run_id", "")
      error_msg = getattr(event, "content", "") or getattr(event, "error_type", "") or "Unknown error"
      self._post(RunError(run_id=run_id, error=str(error_msg)))
      self._streamed_run_id = None
      return

    if isinstance(event, RunCancelledEvent):
      run_id = getattr(event, "run_id", "")
      self._post(RunError(run_id=run_id, error="Cancelled"))
      self._streamed_run_id = None
      return

    # --- Streaming content ---
    if isinstance(event, RunContentEvent):
      content = getattr(event, "content", "") or ""
      if content:
        if self._first_token_time is None:
          self._first_token_time = time.monotonic()
        run_id = self._streamed_run_id or ""
        self._post(StreamChunk(text=str(content), run_id=run_id))
      return

    # --- Reasoning/thinking ---
    if isinstance(event, ReasoningStartedEvent):
      run_id = getattr(event, "run_id", "")
      self._post(ThinkingStarted(run_id=run_id))
      return

    if isinstance(event, (ReasoningStepEvent, ReasoningContentDeltaEvent)):
      text = ""
      if isinstance(event, ReasoningStepEvent):
        text = getattr(event, "reasoning_content", "") or getattr(event, "content", "") or ""
      else:
        text = getattr(event, "delta", "") or getattr(event, "reasoning_content", "") or getattr(event, "content", "") or ""
      if text:
        self._post(ThinkingChunk(text=str(text)))
      return

    if isinstance(event, ReasoningCompletedEvent):
      run_id = getattr(event, "run_id", "")
      self._post(ThinkingCompleted(run_id=run_id))
      return

    # --- Tool calls ---
    # Events carry a `tool` field (ToolExecution) with tool_name, tool_args, tool_call_id, result
    if isinstance(event, PipelineToolStarted):
      tool = getattr(event, "tool", None)
      tool_name = getattr(tool, "tool_name", "") or "" if tool else ""
      arguments = getattr(tool, "tool_args", "") or "" if tool else ""
      call_id = getattr(tool, "tool_call_id", "") or "" if tool else ""
      if isinstance(arguments, dict):
        import json

        arguments = json.dumps(arguments, indent=2)
      self._post(
        ToolCallStarted(
          tool_name=str(tool_name),
          arguments=str(arguments),
          call_id=str(call_id),
        )
      )
      return

    if isinstance(event, PipelineToolCompleted):
      tool = getattr(event, "tool", None)
      tool_name = getattr(tool, "tool_name", "") or "" if tool else ""
      result = getattr(tool, "result", "") or "" if tool else ""
      call_id = getattr(tool, "tool_call_id", "") or "" if tool else ""
      metrics = getattr(tool, "metrics", None)
      duration = getattr(metrics, "time", None) if metrics else None
      # Convert seconds to ms if present
      if duration is not None:
        duration = duration * 1000
      self._post(
        ToolCallCompleted(
          tool_name=str(tool_name),
          result=str(result),
          call_id=str(call_id),
          duration_ms=duration,
        )
      )
      return

    if isinstance(event, ToolCallErrorEvent):
      tool = getattr(event, "tool", None)
      tool_name = getattr(tool, "tool_name", "") or "" if tool else ""
      error_msg = getattr(tool, "result", "") or getattr(event, "content", "") or ""
      call_id = getattr(tool, "tool_call_id", "") or "" if tool else ""
      self._post(
        ToolCallCompleted(
          tool_name=str(tool_name),
          result="",
          call_id=str(call_id),
          error=str(error_msg),
        )
      )
      return

    # --- Model calls ---
    if isinstance(event, ModelCallStartedEvent):
      self._turn_count += 1
      model_id = getattr(event, "model_id", "") or ""
      self._post(
        ModelCallUpdate(
          turn=self._turn_count,
          model_id=str(model_id),
        )
      )
      return

    if isinstance(event, ModelCallCompletedEvent):
      metrics = getattr(event, "metrics", None)
      in_tokens = 0
      out_tokens = 0
      if metrics is not None:
        in_tokens = getattr(metrics, "input_tokens", 0) or 0
        out_tokens = getattr(metrics, "output_tokens", 0) or 0
        self._total_tokens += in_tokens + out_tokens
      self._post(
        ModelCallUpdate(
          turn=self._turn_count,
          input_tokens=in_tokens,
          output_tokens=out_tokens,
        )
      )
      return

    # --- Knowledge ---
    if isinstance(event, KnowledgeRetrievalStartedEvent):
      self._post(KnowledgeUpdate(status="searching"))
      return

    if isinstance(event, KnowledgeRetrievalCompletedEvent):
      docs = getattr(event, "documents", []) or []
      duration = getattr(event, "duration_ms", 0) or 0
      self._post(
        KnowledgeUpdate(
          status="complete",
          doc_count=len(docs),
          duration_ms=duration,
        )
      )
      return

    # --- Memory ---
    if isinstance(event, MemoryRecallStartedEvent):
      self._post(MemoryUpdate(status="recalling"))
      return

    if isinstance(event, MemoryRecallCompletedEvent):
      entries = getattr(event, "entries", []) or []
      duration = getattr(event, "duration_ms", 0) or 0
      self._post(
        MemoryUpdate(
          status="recalled",
          entry_count=len(entries),
          duration_ms=duration,
        )
      )
      return

    if isinstance(event, MemoryUpdateStartedEvent):
      self._post(MemoryUpdate(status="updating"))
      return

    if isinstance(event, MemoryUpdateCompletedEvent):
      duration = getattr(event, "duration_ms", 0) or 0
      self._post(
        MemoryUpdate(
          status="updated",
          duration_ms=duration,
        )
      )
      return
