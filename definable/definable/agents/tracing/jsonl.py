"""JSONL file-based trace exporter."""

import contextlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Dict

if TYPE_CHECKING:
  from definable.run.base import BaseRunOutputEvent


class JSONLExporter:
  """
  Exports traces to JSONL files, one file per session.

  Each event is written as a single line of JSON, making it easy
  to stream, parse, and analyze traces. Files are organized by
  session_id for grouping related runs together.

  File structure:
      {trace_dir}/
          {session_id_1}.jsonl
          {session_id_2}.jsonl
          default.jsonl  # For events without session_id

  Example:
      exporter = JSONLExporter("./traces")
      exporter.export(event)
      exporter.flush()
      exporter.shutdown()

  JSONL format (one event per line):
      {"created_at": 1706500000, "event": "RunStarted", "run_id": "abc", ...}
      {"created_at": 1706500001, "event": "ToolCallStarted", "run_id": "abc", ...}
      {"created_at": 1706500002, "event": "RunCompleted", "run_id": "abc", ...}
  """

  def __init__(self, trace_dir: str):
    """
    Initialize the JSONL exporter.

    Args:
        trace_dir: Directory path where trace files will be written.
            Will be created if it doesn't exist.
    """
    self.trace_dir = Path(trace_dir)
    self.trace_dir.mkdir(parents=True, exist_ok=True)
    self._handles: Dict[str, IO[str]] = {}

  def _get_handle(self, session_id: str) -> IO[str]:
    """
    Get or create a file handle for the given session.

    Args:
        session_id: The session identifier.

    Returns:
        File handle opened for appending.
    """
    if session_id not in self._handles:
      path = self.trace_dir / f"{session_id}.jsonl"
      self._handles[session_id] = open(path, "a", encoding="utf-8")
    return self._handles[session_id]

  def export(self, event: "BaseRunOutputEvent") -> None:
    """
    Export a single event to the appropriate session file.

    Args:
        event: The event to export.
    """
    session_id = getattr(event, "session_id", None) or "default"
    handle = self._get_handle(session_id)

    # Use the event's built-in serialization
    json_line = event.to_json(indent=None)
    handle.write(json_line + "\n")

  def flush(self) -> None:
    """Flush all open file handles."""
    for handle in self._handles.values():
      with contextlib.suppress(Exception):
        handle.flush()

  def shutdown(self) -> None:
    """Close all file handles and release resources."""
    for handle in self._handles.values():
      with contextlib.suppress(Exception):
        handle.close()
    self._handles.clear()

  @property
  def open_sessions(self) -> int:
    """Return the number of open session files."""
    return len(self._handles)

  def get_trace_path(self, session_id: str) -> Path:
    """
    Get the file path for a session's trace file.

    Args:
        session_id: The session identifier.

    Returns:
        Path to the trace file (may not exist yet).
    """
    return self.trace_dir / f"{session_id}.jsonl"

  def __enter__(self) -> "JSONLExporter":
    """Context manager entry."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Context manager exit - ensures proper cleanup."""
    self.shutdown()


def read_trace_file(path: Path) -> list:
  """
  Read and parse a JSONL trace file.

  Args:
      path: Path to the JSONL trace file.

  Returns:
      List of parsed event dictionaries.

  Example:
      events = read_trace_file(Path("./traces/session_xyz.jsonl"))
      for event in events:
          print(event["event"], event["run_id"])
  """
  import json

  events = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        events.append(json.loads(line))
  return events


def read_trace_events(path: Path) -> list:
  """
  Read and deserialize a JSONL trace file to event objects.

  Args:
      path: Path to the JSONL trace file.

  Returns:
      List of deserialized BaseRunOutputEvent objects.

  Example:
      from definable.agents.tracing.jsonl import read_trace_events

      events = read_trace_events(Path("./traces/session_xyz.jsonl"))
      for event in events:
          if event.event == "RunCompleted":
              print(f"Run {event.run_id} completed")
  """
  import json

  from definable.run.agent import run_output_event_from_dict

  events = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        data = json.loads(line)
        event = run_output_event_from_dict(data)
        events.append(event)
  return events
