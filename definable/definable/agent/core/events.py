"""EventBus + Event types — the harness's only observability primitive.

Independent of run lifecycle. Fire-and-forget. Subscribers can be sync
callbacks (fire on emit) or async consumers (via stream()).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, TypeVar

if TYPE_CHECKING:
  from definable.agent.core.debug import TurnSnapshot

log = logging.getLogger(__name__)


# ---- Tool primitives (harness-internal, kept minimal on purpose) ---------


@dataclass(frozen=True)
class ToolCall:
  """The harness's minimal view of a tool invocation request.

  Decoupled from `definable.tool.function.FunctionCall` so the loop
  doesn't drag in HITL/pre-hook/post-hook machinery.
  """

  id: str
  name: str
  args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
  """The harness's minimal view of a tool invocation outcome."""

  call: ToolCall
  success: bool
  output: Any | None = None
  error: str | None = None


# ---- Event hierarchy -----------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Event:
  """Base class — every harness event extends this."""

  run_id: str
  timestamp: float


@dataclass(frozen=True, kw_only=True)
class TurnStarted(Event):
  """Fires at the top of each loop iteration, before the model call."""

  snapshot: TurnSnapshot


@dataclass(frozen=True, kw_only=True)
class StreamChunkEvent(Event):
  """A delta from a streaming model call.

  `kind` is one of "content" | "reasoning" | "tool_call_delta".
  """

  kind: str
  data: str


@dataclass(frozen=True, kw_only=True)
class ModelResponded(Event):
  """Fires after a model call completes, before tool dispatch.

  `usage` is a flat dict of provider-reported token counts when available.
  Kept as a plain dict (not the `model.metrics.Metrics` dataclass) so this
  layer stays free of `definable.model` imports — observability serialization
  is trivial.
  """

  content: str | None
  tool_calls: list[ToolCall] = field(default_factory=list)
  usage: dict[str, int] | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCallStarted(Event):
  call: ToolCall


@dataclass(frozen=True, kw_only=True)
class ToolCallCompleted(Event):
  call: ToolCall
  output: Any


@dataclass(frozen=True, kw_only=True)
class ToolCallFailed(Event):
  call: ToolCall
  error: str


@dataclass(frozen=True, kw_only=True)
class MemoryAccessed(Event):
  """Fires when a memory tool reads/writes/lists/searches.

  `op` is one of "read" | "write" | "list" | "search".
  `key` is the file name for read/write, the query for search, None for list.
  """

  op: str
  key: str | None = None


@dataclass(frozen=True, kw_only=True)
class RunCompleted(Event):
  content: str | None
  turns: int
  exit_reason: str = "natural"


@dataclass(frozen=True, kw_only=True)
class RunErrored(Event):
  error: str
  turns: int


# ---- EventBus ------------------------------------------------------------

E = TypeVar("E", bound=Event)
SyncSubscriber = Callable[[Event], None]
Unsubscribe = Callable[[], None]


class EventBus:
  """Pub/sub for harness events.

  Sync subscribers fire inline on emit. Async consumers use stream(), which
  attaches a per-call asyncio.Queue and yields events as they arrive.
  """

  def __init__(self) -> None:
    self._subscribers: list[SyncSubscriber] = []

  def emit(self, event: Event) -> None:
    """Fire to every subscriber. Subscriber exceptions are logged, not raised."""
    # Iterate over a copy — subscribers may unsubscribe themselves during fire.
    for sub in list(self._subscribers):
      try:
        sub(event)
      except Exception:
        log.exception("Event subscriber raised; continuing")

  def on(self, event_type: type[E]) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
    """Decorator that subscribes a typed handler filtered by event_type.

    Usage::

        @bus.on(RunCompleted)
        def handle(e: RunCompleted) -> None:
          print(e.content)
    """

    def decorator(fn: Callable[[E], None]) -> Callable[[E], None]:
      def wrapper(e: Event) -> None:
        if isinstance(e, event_type):
          fn(e)

      self._subscribers.append(wrapper)
      return fn

    return decorator

  def subscribe(self, callback: SyncSubscriber) -> Unsubscribe:
    """Register a generic subscriber. Returns an unsubscribe callable."""
    self._subscribers.append(callback)

    def unsub() -> None:
      with contextlib.suppress(ValueError):
        self._subscribers.remove(callback)

    return unsub

  async def stream(self) -> AsyncIterator[Event]:
    """Async iterator yielding every emitted event for the lifetime of the iteration.

    Each call attaches its own queue, so multiple concurrent consumers are
    independent. Cancellation cleanly unsubscribes.

    Note: the per-consumer queue is unbounded — a slow consumer will grow
    memory until it catches up. For most agents the event rate is bounded
    by model latency, but high-throughput producers should consume
    promptly or attach a sync subscriber instead.
    """
    queue: asyncio.Queue[Event] = asyncio.Queue()

    def push(e: Event) -> None:
      queue.put_nowait(e)

    unsub = self.subscribe(push)
    try:
      while True:
        yield await queue.get()
    finally:
      unsub()
