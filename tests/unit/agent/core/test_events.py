"""EventBus unit tests — pub/sub semantics, typed filtering, async stream."""

from __future__ import annotations

import asyncio
import time

import pytest

from definable.agent.core.debug import TurnSnapshot
from definable.agent.core.events import (
  Event,
  EventBus,
  MemoryAccessed,
  RunCompleted,
  TurnStarted,
)


def _now() -> float:
  return time.time()


def _ev_completed(content: str = "ok") -> RunCompleted:
  return RunCompleted(run_id="r1", timestamp=_now(), content=content, turns=1)


@pytest.mark.unit
def test_subscribe_fires_on_emit() -> None:
  bus = EventBus()
  received: list[Event] = []
  bus.subscribe(received.append)

  bus.emit(_ev_completed())
  bus.emit(MemoryAccessed(run_id="r1", timestamp=_now(), op="read", key="profile"))

  assert len(received) == 2
  assert isinstance(received[0], RunCompleted)
  assert isinstance(received[1], MemoryAccessed)


@pytest.mark.unit
def test_subscribe_returns_working_unsubscribe() -> None:
  bus = EventBus()
  received: list[Event] = []
  unsub = bus.subscribe(received.append)

  bus.emit(_ev_completed("first"))
  unsub()
  bus.emit(_ev_completed("second"))

  assert len(received) == 1
  assert isinstance(received[0], RunCompleted)
  assert received[0].content == "first"


@pytest.mark.unit
def test_on_decorator_filters_by_type() -> None:
  bus = EventBus()
  completed_seen: list[RunCompleted] = []
  memory_seen: list[MemoryAccessed] = []

  @bus.on(RunCompleted)
  def _h1(e: RunCompleted) -> None:
    completed_seen.append(e)

  @bus.on(MemoryAccessed)
  def _h2(e: MemoryAccessed) -> None:
    memory_seen.append(e)

  bus.emit(_ev_completed())
  bus.emit(MemoryAccessed(run_id="r1", timestamp=_now(), op="write", key="note"))
  bus.emit(_ev_completed())

  assert len(completed_seen) == 2
  assert len(memory_seen) == 1
  assert memory_seen[0].op == "write"


@pytest.mark.unit
def test_subscriber_exception_does_not_crash_emit() -> None:
  bus = EventBus()

  def boom(_: Event) -> None:
    raise RuntimeError("subscriber broken")

  received: list[Event] = []
  bus.subscribe(boom)
  bus.subscribe(received.append)

  bus.emit(_ev_completed())  # must not raise

  assert len(received) == 1


@pytest.mark.unit
async def test_stream_yields_events_in_order() -> None:
  bus = EventBus()

  async def consume(n: int) -> list[Event]:
    out: list[Event] = []
    async for ev in bus.stream():
      out.append(ev)
      if len(out) >= n:
        break
    return out

  consumer = asyncio.create_task(consume(3))
  await asyncio.sleep(0)  # let consumer subscribe

  bus.emit(_ev_completed("one"))
  bus.emit(MemoryAccessed(run_id="r1", timestamp=_now(), op="list"))
  snap = TurnSnapshot(turn=1, context_tokens=42, messages=2)
  bus.emit(TurnStarted(run_id="r1", timestamp=_now(), snapshot=snap))

  events = await asyncio.wait_for(consumer, timeout=1.0)

  assert len(events) == 3
  assert isinstance(events[0], RunCompleted)
  assert isinstance(events[1], MemoryAccessed)
  assert isinstance(events[2], TurnStarted)
