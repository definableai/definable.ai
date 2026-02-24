"""Live event collector — implements TraceExporter for the observability dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set

from definable.utils.log import log_warning

if TYPE_CHECKING:
  from definable.agent.events import BaseRunOutputEvent


class ObservabilityExporter:
  """Collects agent events into a ring buffer for live SSE streaming and metrics.

  Implements the ``TraceExporter`` protocol so it plugs into the existing
  ``Tracing`` → ``TraceWriter`` pipeline with zero changes to event emission.

  Features:
    - Ring buffer (deque with maxlen) — oldest events auto-evicted
    - Per-run indexing for efficient run-scoped queries
    - SSE fan-out via per-client ``asyncio.Queue`` with backpressure
    - Thread-safe append (deque is thread-safe for append/popleft in CPython)

  Example:
      exporter = ObservabilityExporter(buffer_size=10_000)
      # Plug into Tracing:
      agent = Agent(model=model, tracing=Tracing(exporters=[exporter]))
  """

  def __init__(self, buffer_size: int = 10_000) -> None:
    self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
    self._run_index: Dict[str, List[Dict[str, Any]]] = {}
    self._clients: Set[asyncio.Queue[Optional[Dict[str, Any]]]] = set()
    self._event_count: int = 0

  def export(self, event: "BaseRunOutputEvent") -> None:
    """Export a single event — append to buffer and fan-out to SSE clients.

    Args:
      event: The event to export.
    """
    try:
      event_dict = event.to_dict()
    except Exception as exc:
      log_warning(f"ObservabilityExporter: failed to serialize event: {exc!r}")
      return

    event_dict.setdefault("_obs_ts", time.time())
    self._buffer.append(event_dict)
    self._event_count += 1

    # Index by run_id
    run_id = event_dict.get("run_id")
    if run_id:
      self._run_index.setdefault(run_id, []).append(event_dict)

    # Fan-out to SSE clients (non-blocking)
    dead_clients: List[asyncio.Queue[Optional[Dict[str, Any]]]] = []
    for q in self._clients:
      try:
        q.put_nowait(event_dict)
      except asyncio.QueueFull:
        # Client can't keep up — drop event for this client
        pass
      except Exception:
        dead_clients.append(q)

    for q in dead_clients:
      self._clients.discard(q)

  def flush(self) -> None:
    """No-op — events are immediately available in the buffer."""

  def shutdown(self) -> None:
    """Signal all SSE clients to disconnect and clean up."""
    for q in self._clients:
      with contextlib.suppress(Exception):
        q.put_nowait(None)  # Sentinel to close SSE generator
    self._clients.clear()

  # --- SSE client management ---

  def subscribe(self, maxsize: int = 256) -> asyncio.Queue[Optional[Dict[str, Any]]]:
    """Register a new SSE client.

    Args:
      maxsize: Maximum queued events before dropping.

    Returns:
      An asyncio.Queue that will receive event dicts (or None for shutdown).
    """
    q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=maxsize)
    self._clients.add(q)
    return q

  def unsubscribe(self, q: asyncio.Queue[Optional[Dict[str, Any]]]) -> None:
    """Remove an SSE client."""
    self._clients.discard(q)

  # --- Query API ---

  @property
  def buffer(self) -> Deque[Dict[str, Any]]:
    """Access the raw event ring buffer."""
    return self._buffer

  @property
  def client_count(self) -> int:
    """Number of connected SSE clients."""
    return len(self._clients)

  @property
  def event_count(self) -> int:
    """Total events exported since creation."""
    return self._event_count

  def get_events_for_run(self, run_id: str) -> List[Dict[str, Any]]:
    """Return all buffered events for a specific run.

    Args:
      run_id: The run identifier.

    Returns:
      List of event dicts for the run (may be empty).
    """
    return list(self._run_index.get(run_id, []))

  def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent events from the buffer.

    Args:
      limit: Maximum events to return.

    Returns:
      List of event dicts, most recent last.
    """
    buf = self._buffer
    if limit >= len(buf):
      return list(buf)
    return list(buf)[-limit:]

  def get_run_ids(self) -> List[str]:
    """Return all run IDs seen in the buffer.

    Returns:
      List of run IDs, in order of first appearance.
    """
    return list(self._run_index.keys())
