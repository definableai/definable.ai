"""Base tracing infrastructure with protocol-based extensibility."""

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
  from definable.agent.events import BaseRunOutputEvent


@dataclass
class Tracing:
  """Standalone tracing block — a composable lego piece.

  Snaps directly into an Agent without a config wrapper:

      from definable.agent.tracing import Tracing, JSONLExporter
      agent = Agent(model=model, tracing=Tracing(exporters=[JSONLExporter("./traces")]))

  Attributes:
    enabled: Whether tracing is active.
    exporters: List of TraceExporter implementations.
    event_filter: Optional filter function for events.
    batch_size: Batch size for export (1 = immediate).
    flush_interval_ms: Flush interval in milliseconds.
  """

  enabled: bool = True
  exporters: Optional[List["TraceExporter"]] = field(default=None)
  event_filter: Optional[Callable[["BaseRunOutputEvent"], bool]] = field(default=None)
  batch_size: int = 1
  flush_interval_ms: int = 5000


@runtime_checkable
class TraceExporter(Protocol):
  """
  Protocol for trace exporters - supports multiple backends.

  Implement this protocol to create custom trace exporters for
  different backends (files, databases, OpenTelemetry, etc.).

  Example:
      class MyExporter:
          def export(self, event: BaseRunOutputEvent) -> None:
              # Send event to your backend
              pass

          def flush(self) -> None:
              # Ensure all events are persisted
              pass

          def shutdown(self) -> None:
              # Clean up resources
              pass
  """

  def export(self, event: "BaseRunOutputEvent") -> None:
    """
    Export a single event to the backend.

    Args:
        event: The event to export.
    """
    ...

  def flush(self) -> None:
    """Ensure all buffered events are persisted."""
    ...

  def shutdown(self) -> None:
    """Clean up resources and close connections."""
    ...


class TraceWriter:
  """
  Manages multiple trace exporters with filtering support.

  TraceWriter coordinates writing events to multiple exporters,
  applying filters, and managing lifecycle.

  Example:
      config = Tracing(
          exporters=[JSONLExporter("./traces"), OTelExporter(...)],
          event_filter=lambda e: e.event != "RunContent",  # Skip streaming
      )
      writer = TraceWriter(config)
      writer.write(event)
      writer.shutdown()
  """

  def __init__(self, config: "Tracing"):
    """
    Initialize the trace writer.

    Args:
        config: Tracing instance with exporters and filters.
    """
    self.config = config
    self._exporters: List[TraceExporter] = config.exporters or []

  def write(self, event: "BaseRunOutputEvent") -> None:
    """
    Write an event to all configured exporters.

    Applies filtering before export. If tracing is disabled or
    the event is filtered out, this is a no-op.

    Args:
        event: The event to write.
    """
    if not self.config.enabled:
      return

    if self.config.event_filter and not self.config.event_filter(event):
      return

    for exporter in self._exporters:
      with contextlib.suppress(Exception):
        # Tracing should never break the main execution flow
        # In production, you might want to log this
        exporter.export(event)

  def flush(self) -> None:
    """Flush all exporters to ensure events are persisted."""
    for exporter in self._exporters:
      with contextlib.suppress(Exception):
        exporter.flush()

  def shutdown(self) -> None:
    """
    Shutdown all exporters and release resources.

    This should be called when the agent is being disposed of.
    """
    self.flush()
    for exporter in self._exporters:
      with contextlib.suppress(Exception):
        exporter.shutdown()

  @property
  def exporter_count(self) -> int:
    """Return the number of configured exporters."""
    return len(self._exporters)

  def add_exporter(self, exporter: TraceExporter) -> None:
    """
    Add an exporter at runtime.

    Args:
        exporter: The exporter to add.
    """
    self._exporters.append(exporter)

  def remove_exporter(self, exporter: TraceExporter) -> bool:
    """
    Remove an exporter at runtime.

    Args:
        exporter: The exporter to remove.

    Returns:
        True if the exporter was found and removed, False otherwise.
    """
    try:
      self._exporters.remove(exporter)
      return True
    except ValueError:
      return False


class NoOpExporter:
  """
  A no-op exporter that discards all events.

  Useful for testing or disabling tracing without removing exporters.
  """

  def export(self, event: "BaseRunOutputEvent") -> None:
    """Discard the event."""
    pass

  def flush(self) -> None:
    """No-op."""
    pass

  def shutdown(self) -> None:
    """No-op."""
    pass
