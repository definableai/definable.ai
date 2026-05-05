"""
Behavioral tests: StreamingMiddleware protocol.

Migrated from tests_e2e/behavioral/test_streaming_middleware.py.

Strategy:
  - Verify the StreamingMiddleware protocol is importable and usable
  - Create a class implementing the protocol and verify isinstance check
  - Verify protocol structure matches expected signature

Covers:
  - StreamingMiddleware protocol check via isinstance
  - Protocol is importable and the class structure works
"""

from typing import AsyncGenerator

import pytest

from definable.agent.middleware import StreamingMiddleware
from definable.agent.events import RunContext, RunOutputEvent


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------


class MyStreamingMiddleware:
  """A concrete implementation of the StreamingMiddleware protocol."""

  async def __call__(
    self,
    context: RunContext,
    event_stream: AsyncGenerator[RunOutputEvent, None],
  ) -> AsyncGenerator[RunOutputEvent, None]:
    async for event in event_stream:
      yield event


class NotAStreamingMiddleware:
  """A class that does NOT implement the protocol."""

  def do_something(self) -> str:
    return "not middleware"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.behavioral
class TestStreamingMiddleware:
  """StreamingMiddleware protocol compliance."""

  def test_streaming_middleware_protocol_check(self):
    """A class implementing StreamingMiddleware passes isinstance check."""
    mw = MyStreamingMiddleware()
    assert isinstance(mw, StreamingMiddleware), "MyStreamingMiddleware should satisfy the StreamingMiddleware protocol"

  def test_non_conforming_class_fails_protocol_check(self):
    """A class that does NOT implement __call__ with the right signature fails."""
    obj = NotAStreamingMiddleware()
    assert not isinstance(obj, StreamingMiddleware), "NotAStreamingMiddleware should NOT satisfy the StreamingMiddleware protocol"
