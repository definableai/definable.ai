"""Unified event stream — single subscriber list for all event consumers."""

import asyncio
from typing import Callable, List

from definable.agent.run.base import BaseRunOutputEvent
from definable.utils.log import log_warning


class EventStream:
  """Unified event dispatch for pipeline execution.

  Replaces the dual TraceWriter + EventBus emission pattern with a
  single subscriber list. All consumers (tracing, EventBus, debug)
  subscribe here.

  Example::

      stream = EventStream()

      async def my_handler(event):
          print(f"Got: {event}")

      stream.subscribe(my_handler)
      await stream.emit(some_event)
  """

  def __init__(self) -> None:
    self._handlers: List[Callable] = []

  def subscribe(self, handler: Callable) -> None:
    """Register an event handler.

    Args:
      handler: Async or sync callable that accepts a BaseRunOutputEvent.
    """
    self._handlers.append(handler)

  def unsubscribe(self, handler: Callable) -> None:
    """Remove an event handler."""
    self._handlers = [h for h in self._handlers if h is not handler]

  async def emit(self, event: BaseRunOutputEvent) -> None:
    """Dispatch event to all subscribers.

    Errors in individual handlers are logged but do not propagate
    — event delivery is best-effort to prevent one broken handler
    from killing the pipeline.
    """
    for handler in self._handlers:
      try:
        result = handler(event)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
          await result
      except Exception as exc:
        log_warning(f"EventStream handler {handler!r} raised {exc!r}")

  def clear(self) -> None:
    """Remove all handlers."""
    self._handlers.clear()

  @property
  def handler_count(self) -> int:
    return len(self._handlers)
