"""User-registerable event callbacks for agent runs."""

import asyncio
from typing import Any, Callable, Dict, List

from definable.utils.log import log_warning


class EventBus:
  """User-registerable event callbacks alongside trace exporters.

  Example::

      bus = EventBus()

      @bus.on(ToolCallStartedEvent)
      def log_tool(event):
          print(f"Tool started: {event.tool.tool_name}")

      # The loop calls ``await bus.emit(event)`` for every event.
  """

  def __init__(self) -> None:
    self._handlers: Dict[type, List[Callable]] = {}

  def on(self, event_type: type, handler: Callable | None = None) -> Callable:
    """Register a handler for *event_type*.

    Can be used as a decorator (no args) or as a direct call::

        # Decorator
        @bus.on(RunCompletedEvent)
        def handle(event): ...

        # Direct
        bus.on(RunCompletedEvent, my_handler)

    Returns the handler so it can be used as a decorator.
    """
    if handler is not None:
      self._handlers.setdefault(event_type, []).append(handler)
      return handler

    # Used as ``@bus.on(EventType)``
    def decorator(fn: Callable) -> Callable:
      self._handlers.setdefault(event_type, []).append(fn)
      return fn

    return decorator

  def off(self, event_type: type, handler: Callable) -> None:
    """Remove a previously-registered handler."""
    handlers = self._handlers.get(event_type, [])
    if handler in handlers:
      handlers.remove(handler)

  async def emit(self, event: Any) -> None:
    """Dispatch *event* to all matching handlers (non-fatal).

    Handlers whose signature is a coroutine are awaited; sync handlers
    are called directly. Errors are logged but never propagate.
    """
    for event_type, handlers in self._handlers.items():
      if isinstance(event, event_type):
        for handler in handlers:
          try:
            result = handler(event)
            if asyncio.iscoroutine(result):
              await result
          except Exception as exc:
            log_warning(f"EventBus handler error for {event_type.__name__}: {exc}")
