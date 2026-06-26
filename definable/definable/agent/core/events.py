"""EventBus + step events — the harness's only observability primitive.

Everything the agent does is a *step*. A run is::

    AgentBegin
      StepBegin / StepDelta* / StepEnd   (type="reasoning")   ─┐ one model call
      StepBegin / StepDelta* / StepEnd   (type="content")     ─┘ (content carries usage)
      StepBegin / StepEnd                (type="tool")  × N
      ... repeat per turn ...
    AgentEnd  (or AgentError)

Events are observe-only and fan out fire-and-forget. To *act* on the run
(mutate messages/args, veto a tool, abort) use a Hook — see hooks.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal, TypeVar

log = logging.getLogger(__name__)


# ---- Tool primitives (harness-internal, kept minimal on purpose) ---------


@dataclass(frozen=True)
class ToolCall:
  """The harness's minimal view of a tool invocation request.

  Decoupled from `FunctionCall` so the loop doesn't drag in the heavier
  tool machinery.
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
  skipped: bool = False  # a before_tool hook raised SkipTool
  aborted: bool = False  # a hook raised AbortRun on this call


# ---- Step events ---------------------------------------------------------

StepType = Literal["content", "reasoning", "tool"]


@dataclass(frozen=True, kw_only=True)
class Event:
  """Base class — every harness event extends this."""

  run_id: str
  timestamp: float


@dataclass(frozen=True, kw_only=True)
class AgentBegin(Event):
  """Fires once, at the very start of a run."""


@dataclass(frozen=True, kw_only=True)
class StepBegin(Event):
  """A step opened. `id` correlates Begin/Delta/End.

  For tool steps `id` is the provider tool_call_id and `name`/`args` are set.
  For content/reasoning steps `id` is synthesized per model call.
  """

  id: str
  type: StepType
  turn: int
  name: str | None = None  # tool steps only
  args: dict[str, Any] | None = None  # tool steps only


@dataclass(frozen=True, kw_only=True)
class StepDelta(Event):
  """An incremental fragment for an open step (streaming only).

  `data` is a content/reasoning text fragment, or a partial tool-args
  fragment when the provider streams them (OpenAI).
  """

  id: str
  type: StepType
  data: str


@dataclass(frozen=True, kw_only=True)
class StepEnd(Event):
  """A step closed.

  - content/reasoning: `data` = final text; content also carries `usage`
    (the per-model-call token metrics — usage is per model call, not per tool).
  - tool: `data` = stringified result, plus `success`/`error`/`duration_ms`.
  """

  id: str
  type: StepType
  data: str | None = None
  usage: dict[str, int] | None = None  # content step only
  success: bool | None = None  # tool step only
  error: str | None = None  # tool step only
  duration_ms: float | None = None  # tool step only


@dataclass(frozen=True, kw_only=True)
class AgentEnd(Event):
  """Fires once, when the run completes. `usage` is the run total."""

  content: str | None
  turns: int
  usage: dict[str, int] | None = None
  exit_reason: Literal["natural", "max_turns", "aborted"] = "natural"


@dataclass(frozen=True, kw_only=True)
class AgentError(Event):
  """Fires once, when a model call raises and ends the run."""

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

        @bus.on(AgentEnd)
        def handle(e: AgentEnd) -> None:
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
