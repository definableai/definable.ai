"""Replay — structured view of a past agent run."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from definable.agent.replay.types import (
  KnowledgeRetrievalRecord,
  MemoryRecallRecord,
  ReplayStep,
  ReplayTokens,
  ToolCallRecord,
)
from definable.agent.events import BaseRunOutputEvent


@dataclass
class Replay:
  """Structured view of a past agent run, built from trace events or RunOutput."""

  run_id: str = ""
  session_id: str = ""
  agent_id: str = ""
  agent_name: str = ""
  model: str = ""
  model_provider: str = ""

  # Original input
  input: Optional[Any] = None  # RunInput

  # Final output
  content: Optional[Any] = None

  # Conversation messages
  messages: List[Any] = field(default_factory=list)  # List[Message]

  # Tool calls (ordered)
  tool_calls: List[ToolCallRecord] = field(default_factory=list)

  # Aggregated metrics
  tokens: ReplayTokens = field(default_factory=ReplayTokens)
  cost: Optional[float] = None
  duration: Optional[float] = None

  # Per-step timing
  steps: List[ReplayStep] = field(default_factory=list)

  # Knowledge & memory
  knowledge_retrievals: List[KnowledgeRetrievalRecord] = field(default_factory=list)
  memory_recalls: List[MemoryRecallRecord] = field(default_factory=list)

  # Status
  status: str = "completed"
  error: Optional[str] = None

  # Raw events
  events: List[BaseRunOutputEvent] = field(default_factory=list)

  # Source
  source: str = "trace_file"

  @classmethod
  def from_events(
    cls,
    events: List[BaseRunOutputEvent],
    *,
    run_id: Optional[str] = None,
  ) -> "Replay":
    """Build a Replay from a list of trace events.

    Args:
      events: List of deserialized trace events.
      run_id: Filter to a specific run. If None, uses the first run found.

    Returns:
      Replay instance with all fields populated from events.
    """
    from definable.agent.events import (
      KnowledgeRetrievalCompletedEvent,
      MemoryRecallCompletedEvent,
      RunCancelledEvent,
      RunCompletedEvent,
      RunErrorEvent,
      RunStartedEvent,
      ToolCallCompletedEvent,
      ToolCallStartedEvent,
    )

    # Find the target run_id
    if run_id is None:
      for evt in events:
        if hasattr(evt, "run_id") and evt.run_id:
          run_id = evt.run_id
          break
    if run_id is None:
      return cls(source="trace_file")

    # Filter events for this run
    run_events = [e for e in events if getattr(e, "run_id", None) == run_id]

    replay = cls(
      run_id=run_id,
      events=run_events,
      source="trace_file",
    )

    # Track tool call starts for pairing with completions
    tool_starts: dict[str, ToolCallStartedEvent] = {}

    for evt in run_events:
      if isinstance(evt, RunStartedEvent):
        replay.agent_id = evt.agent_id
        replay.agent_name = evt.agent_name
        replay.session_id = evt.session_id or ""
        replay.model = evt.model
        replay.model_provider = evt.model_provider
        replay.input = evt.run_input

        replay.steps.append(
          ReplayStep(
            step_type="model_call",
            name="model",
            started_at=evt.created_at,
          )
        )

      elif isinstance(evt, ToolCallStartedEvent):
        tool = evt.tool
        tool_name = tool.tool_name if tool else ""
        tool_call_id = tool.tool_call_id if tool else None

        if tool_call_id:
          tool_starts[tool_call_id] = evt

        replay.steps.append(
          ReplayStep(
            step_type="tool_call",
            name=tool_name,
            started_at=evt.created_at,
          )
        )

      elif isinstance(evt, ToolCallCompletedEvent):
        tool = evt.tool
        if tool:
          started_at = evt.created_at
          duration_ms: Optional[float] = None
          # Try to match with start event
          if tool.tool_call_id and tool.tool_call_id in tool_starts:
            start_evt = tool_starts[tool.tool_call_id]
            started_at = start_evt.created_at
            if evt.created_at and start_evt.created_at:
              duration_ms = (evt.created_at - start_evt.created_at) * 1000.0

          record = ToolCallRecord(
            tool_name=tool.tool_name or "",
            tool_args=tool.tool_args,
            result=tool.result,
            error=tool.tool_call_error,
            started_at=started_at,
            completed_at=evt.created_at,
            duration_ms=duration_ms,
          )
          replay.tool_calls.append(record)

          # Update matching step
          for step in reversed(replay.steps):
            if step.step_type == "tool_call" and step.name == tool.tool_name and step.completed_at is None:
              step.completed_at = evt.created_at
              step.duration_ms = duration_ms
              break

      elif isinstance(evt, KnowledgeRetrievalCompletedEvent):
        replay.knowledge_retrievals.append(
          KnowledgeRetrievalRecord(
            query=evt.query,
            documents_found=evt.documents_found,
            documents_used=evt.documents_used,
            duration_ms=evt.duration_ms,
          )
        )
        replay.steps.append(
          ReplayStep(
            step_type="knowledge_retrieval",
            name="knowledge",
            started_at=evt.created_at,
            completed_at=evt.created_at,
            duration_ms=evt.duration_ms,
          )
        )

      elif isinstance(evt, MemoryRecallCompletedEvent):
        replay.memory_recalls.append(
          MemoryRecallRecord(
            query=evt.query,
            tokens_used=evt.tokens_used,
            chunks_included=evt.chunks_included,
            chunks_available=evt.chunks_available,
            duration_ms=evt.duration_ms,
          )
        )
        replay.steps.append(
          ReplayStep(
            step_type="memory_recall",
            name="memory",
            started_at=evt.created_at,
            completed_at=evt.created_at,
            duration_ms=evt.duration_ms,
          )
        )

      elif isinstance(evt, RunCompletedEvent):
        replay.content = evt.content
        replay.status = "completed"

        # Update model_call step completion
        for step in replay.steps:
          if step.step_type == "model_call" and step.completed_at is None:
            step.completed_at = evt.created_at
            break

        # Extract metrics
        if evt.metrics:
          replay.tokens = ReplayTokens(
            input_tokens=evt.metrics.input_tokens,
            output_tokens=evt.metrics.output_tokens,
            total_tokens=evt.metrics.total_tokens,
            reasoning_tokens=evt.metrics.reasoning_tokens,
            cache_read_tokens=evt.metrics.cache_read_tokens,
            cache_write_tokens=evt.metrics.cache_write_tokens,
          )
          replay.cost = evt.metrics.cost
          replay.duration = evt.metrics.duration

      elif isinstance(evt, RunErrorEvent):
        replay.status = "error"
        replay.error = evt.content

      elif isinstance(evt, RunCancelledEvent):
        replay.status = "cancelled"

    # Compute duration from first to last event if not set from metrics
    if replay.duration is None and run_events:
      first_ts = run_events[0].created_at  # type: ignore[attr-defined]
      last_ts = run_events[-1].created_at  # type: ignore[attr-defined]
      if first_ts and last_ts and last_ts > first_ts:
        replay.duration = float(last_ts - first_ts)

    return replay

  @classmethod
  def from_run_output(cls, run_output: Any) -> "Replay":
    """Build a Replay from a RunOutput object.

    Args:
      run_output: RunOutput returned by agent.run() or agent.arun().

    Returns:
      Replay instance populated from the RunOutput.
    """
    from definable.agent.events import RunStatus

    replay = cls(
      run_id=run_output.run_id or "",
      session_id=run_output.session_id or "",
      agent_id=run_output.agent_id or "",
      agent_name=run_output.agent_name or "",
      model=run_output.model or "",
      model_provider=run_output.model_provider or "",
      input=run_output.input,
      content=run_output.content,
      messages=run_output.messages or [],
      source="run_output",
    )

    # Tool calls
    if run_output.tools:
      for tool in run_output.tools:
        replay.tool_calls.append(
          ToolCallRecord(
            tool_name=tool.tool_name or "",
            tool_args=tool.tool_args,
            result=tool.result,
            error=tool.tool_call_error,
            started_at=tool.created_at,
          )
        )

    # Metrics
    if run_output.metrics:
      replay.tokens = ReplayTokens(
        input_tokens=run_output.metrics.input_tokens,
        output_tokens=run_output.metrics.output_tokens,
        total_tokens=run_output.metrics.total_tokens,
        reasoning_tokens=run_output.metrics.reasoning_tokens,
        cache_read_tokens=run_output.metrics.cache_read_tokens,
        cache_write_tokens=run_output.metrics.cache_write_tokens,
      )
      replay.cost = run_output.metrics.cost
      replay.duration = run_output.metrics.duration

    # Status
    if run_output.status == RunStatus.completed:
      replay.status = "completed"
    elif run_output.status == RunStatus.error:
      replay.status = "error"
    elif run_output.status == RunStatus.cancelled:
      replay.status = "cancelled"
    else:
      replay.status = run_output.status.value if isinstance(run_output.status, RunStatus) else str(run_output.status)

    # Events
    if run_output.events:
      replay.events = list(run_output.events)

    return replay

  @classmethod
  def from_trace_file(
    cls,
    path: Union[str, Path],
    *,
    run_id: Optional[str] = None,
  ) -> "Replay":
    """Build a Replay from a JSONL trace file.

    Args:
      path: Path to the JSONL trace file.
      run_id: Filter to a specific run. If None, uses the first run found.

    Returns:
      Replay instance built from the trace file events.
    """
    from definable.agent.tracing.jsonl import read_trace_events

    events = read_trace_events(Path(path))
    return cls.from_events(events, run_id=run_id)
