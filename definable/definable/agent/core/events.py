"""EventBus + Event types — the harness's only observability primitive.

Independent of run lifecycle. Fire-and-forget. Subscribers can be sync
callbacks (fire on emit) or async consumers (via stream()).

Phase 2 scaffold: type definitions only. EventBus method bodies land in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, TypeVar

if TYPE_CHECKING:
  from definable.agent.core.debug import TurnSnapshot
  from definable.tool.function import FunctionCall


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
  """Fires after a model call completes, before tool dispatch."""

  content: str | None
  tool_calls: list[FunctionCall] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ToolCallStarted(Event):
  call: FunctionCall


@dataclass(frozen=True, kw_only=True)
class ToolCallCompleted(Event):
  call: FunctionCall
  result: Any


@dataclass(frozen=True, kw_only=True)
class ToolCallFailed(Event):
  call: FunctionCall
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
  iterates from an internal asyncio.Queue.

  Phase 2: skeleton — bodies land in Phase 3.
  """

  def __init__(self) -> None:
    raise NotImplementedError("Phase 3")

  def emit(self, event: Event) -> None:
    raise NotImplementedError("Phase 3")

  def on(self, event_type: type[E]) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
    raise NotImplementedError("Phase 3")

  def subscribe(self, callback: SyncSubscriber) -> Unsubscribe:
    raise NotImplementedError("Phase 3")

  def stream(self) -> AsyncIterator[Event]:
    raise NotImplementedError("Phase 3")
