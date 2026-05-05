"""
Unit tests for EventBus user-registerable event callbacks.

Tests registration, decorator syntax, unregistration, type filtering,
and error isolation. No API calls.

Migrated from tests_e2e/unit/test_event_bus.py -- all original tests preserved.
"""

from dataclasses import dataclass

import pytest

from definable.agent.event_bus import EventBus


@dataclass
class FakeEvent:
  value: str = "test"


@dataclass
class OtherEvent:
  count: int = 0


@pytest.mark.unit
class TestEventBus:
  @pytest.mark.asyncio
  async def test_on_registers_handler(self):
    bus = EventBus()
    received = []

    def handler(event):
      received.append(event)

    bus.on(FakeEvent, handler)
    await bus.emit(FakeEvent(value="hello"))

    assert len(received) == 1
    assert received[0].value == "hello"

  @pytest.mark.asyncio
  async def test_on_decorator_registers_handler(self):
    bus = EventBus()
    received = []

    @bus.on(FakeEvent)
    def handler(event):
      received.append(event)

    await bus.emit(FakeEvent(value="decorated"))

    assert len(received) == 1
    assert received[0].value == "decorated"

  @pytest.mark.asyncio
  async def test_off_removes_handler(self):
    bus = EventBus()
    received = []

    def handler(event):
      received.append(event)

    bus.on(FakeEvent, handler)
    bus.off(FakeEvent, handler)
    await bus.emit(FakeEvent())

    assert len(received) == 0

  @pytest.mark.asyncio
  async def test_emit_only_matches_event_type(self):
    bus = EventBus()
    received = []

    def handler(event):
      received.append(event)

    bus.on(FakeEvent, handler)

    # Emit a different event type -- handler should NOT fire
    await bus.emit(OtherEvent(count=42))
    assert len(received) == 0

    # Emit the matching type -- handler SHOULD fire
    await bus.emit(FakeEvent(value="match"))
    assert len(received) == 1
    assert received[0].value == "match"

  @pytest.mark.asyncio
  async def test_handler_error_does_not_propagate(self):
    bus = EventBus()

    def bad_handler(event):
      raise RuntimeError("boom")

    bus.on(FakeEvent, bad_handler)

    # emit should not raise even though the handler does
    await bus.emit(FakeEvent())
