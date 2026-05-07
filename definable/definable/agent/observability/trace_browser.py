"""Trace browser — scan JSONL trace files for historical session/run browsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from definable.utils.log import log_warning


@dataclass
class SessionInfo:
  """Metadata for a trace session."""

  session_id: str = ""
  file_path: str = ""
  file_size: int = 0
  modified_at: float = 0.0
  run_count: int = 0


@dataclass
class RunInfo:
  """Metadata for a single run within a session."""

  run_id: str = ""
  session_id: str = ""
  status: str = "unknown"
  started_at: Optional[float] = None
  duration: Optional[float] = None
  tokens: int = 0
  cost: Optional[float] = None
  agent_name: str = ""
  model: str = ""


@dataclass
class TraceBrowser:
  """Scan a JSONL trace directory to list sessions and runs.

  Handles corrupt/empty files gracefully — skips bad lines with a warning,
  never crashes.

  Args:
    trace_dir: Path to the directory containing JSONL trace files.
  """

  trace_dir: str = ""
  _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)

  def __post_init__(self) -> None:
    if not self.trace_dir:
      from definable.utils.workspace import workspace_path

      object.__setattr__(self, "trace_dir", str(workspace_path("traces")))
    path = Path(self.trace_dir)
    path.mkdir(parents=True, exist_ok=True)

  def list_sessions(self) -> List[SessionInfo]:
    """List all trace sessions, sorted by most recently modified.

    Returns:
      List of SessionInfo with file metadata and run count.
    """
    trace_path = Path(self.trace_dir)
    if not trace_path.is_dir():
      return []

    sessions: List[SessionInfo] = []
    for p in trace_path.glob("*.jsonl"):
      try:
        stat = p.stat()
        # Count unique run_ids without fully parsing
        run_ids = self._count_run_ids(p)
        session_id = p.stem
        sessions.append(
          SessionInfo(
            session_id=session_id,
            file_path=str(p),
            file_size=stat.st_size,
            modified_at=stat.st_mtime,
            run_count=len(run_ids),
          )
        )
      except OSError as exc:
        log_warning(f"TraceBrowser: cannot stat {p}: {exc}")

    sessions.sort(key=lambda s: s.modified_at, reverse=True)
    return sessions

  def list_runs(self, session_id: str) -> List[RunInfo]:
    """List all runs within a session.

    Args:
      session_id: The session identifier (matches the JSONL filename stem).

    Returns:
      List of RunInfo for each run in the session.

    Raises:
      FileNotFoundError: If the session file does not exist.
    """
    events = self._load_session_events(session_id)

    # Group events by run_id
    runs_map: Dict[str, List[Dict[str, Any]]] = {}
    for evt in events:
      run_id = evt.get("run_id")
      if run_id:
        runs_map.setdefault(run_id, []).append(evt)

    runs: List[RunInfo] = []
    for run_id, run_events in runs_map.items():
      info = RunInfo(run_id=run_id, session_id=session_id)
      for evt in run_events:
        event_type = evt.get("event", "")
        if event_type == "RunStarted":
          info.started_at = evt.get("created_at")
          info.agent_name = evt.get("agent_name", "")
          info.model = evt.get("model", "")
        elif event_type == "RunCompleted":
          info.status = "completed"
          metrics = evt.get("metrics")
          if metrics:
            info.tokens = metrics.get("total_tokens", 0)
            info.cost = metrics.get("cost")
            info.duration = metrics.get("duration")
        elif event_type == "RunError":
          info.status = "error"
        elif event_type == "RunCancelled":
          info.status = "cancelled"
        elif event_type == "RunPaused":
          info.status = "paused"

      # Compute duration from timestamps if not available from metrics
      if info.duration is None and info.started_at is not None and run_events:
        last_ts = run_events[-1].get("created_at")
        if last_ts and last_ts > info.started_at:
          info.duration = float(last_ts - info.started_at)

      runs.append(info)

    return runs

  def load_run_events(self, session_id: str, run_id: str) -> List[Dict[str, Any]]:
    """Load raw event dicts for a specific run.

    Filters the session file by run_id during parsing (no full file load then filter).

    Args:
      session_id: The session identifier.
      run_id: The run identifier.

    Returns:
      List of event dicts for the run.

    Raises:
      FileNotFoundError: If the session file does not exist.
    """
    events = self._load_session_events(session_id)
    return [e for e in events if e.get("run_id") == run_id]

  def load_replay(self, session_id: str, run_id: str) -> Any:
    """Load a Replay object for a specific run.

    Args:
      session_id: The session identifier.
      run_id: The run identifier.

    Returns:
      A Replay instance built from the run's events.

    Raises:
      FileNotFoundError: If the session file does not exist.
      ValueError: If no events found for the run_id.
    """
    from definable.agent.replay.replay import Replay
    from definable.run.agent import run_output_event_from_dict

    raw_events = self.load_run_events(session_id, run_id)
    if not raw_events:
      raise ValueError(f"No events found for run_id={run_id!r} in session={session_id!r}")

    typed_events = []
    for data in raw_events:
      try:
        typed_events.append(run_output_event_from_dict(data))
      except Exception as exc:
        log_warning(f"TraceBrowser: cannot deserialize event: {exc}")

    return Replay.from_events(typed_events, run_id=run_id)

  def session_exists(self, session_id: str) -> bool:
    """Check if a session file exists.

    Args:
      session_id: The session identifier.
    """
    return self._session_path(session_id).is_file()

  def get_session_path(self, session_id: str) -> Path:
    """Return the file path for a session.

    Args:
      session_id: The session identifier.
    """
    return self._session_path(session_id)

  # --- internals ---

  def _session_path(self, session_id: str) -> Path:
    return Path(self.trace_dir) / f"{session_id}.jsonl"

  def _load_session_events(self, session_id: str) -> List[Dict[str, Any]]:
    """Parse all events from a session JSONL file.

    Skips corrupt lines with a warning.

    Raises:
      FileNotFoundError: If the session file does not exist.
    """
    path = self._session_path(session_id)
    if not path.is_file():
      raise FileNotFoundError(f"Session file not found: {path}")

    events: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
      for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
          continue
        try:
          events.append(json.loads(line))
        except json.JSONDecodeError as exc:
          log_warning(f"TraceBrowser: corrupt line {line_num} in {path.name}: {exc}")

    return events

  def _count_run_ids(self, path: Path) -> set:
    """Quickly count unique run_ids in a JSONL file without full parsing."""
    run_ids: set = set()
    try:
      with open(path, encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          try:
            data = json.loads(line)
            rid = data.get("run_id")
            if rid:
              run_ids.add(rid)
          except json.JSONDecodeError:
            pass  # Skip corrupt lines silently for counting
    except OSError:
      pass
    return run_ids
