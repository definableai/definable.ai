"""Tests for the multi-interface serve() supervisor."""

import asyncio
import contextlib
from typing import Any, Optional

import pytest

from definable.agents.agent import Agent
from definable.agents.testing import MockModel
from definable.interfaces.base import BaseInterface
from definable.interfaces.config import InterfaceConfig
from definable.interfaces.message import InterfaceMessage, InterfaceResponse
from definable.interfaces.serve import serve


class StubInterface(BaseInterface):
  """Minimal interface for testing serve() supervisor behavior.

  Args:
    fail_after: If set, serve_forever() raises RuntimeError after this many seconds.
    stop_after: If set, serve_forever() returns cleanly after this many seconds.
    platform: Platform name for log identification.
  """

  def __init__(
    self,
    agent: Agent,
    *,
    fail_after: Optional[float] = None,
    stop_after: Optional[float] = None,
    platform: str = "stub",
  ):
    super().__init__(
      agent=agent,
      config=InterfaceConfig(platform=platform),
    )
    self.fail_after = fail_after
    self.stop_after = stop_after
    self.start_count = 0

  async def _start_receiver(self) -> None:
    self.start_count += 1

  async def _stop_receiver(self) -> None:
    pass

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    return None

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    pass

  async def serve_forever(self) -> None:
    """Simulate timed failure or clean stop."""
    if not self._running:
      await self.start()
    try:
      if self.fail_after is not None:
        await asyncio.sleep(self.fail_after)
        raise RuntimeError(f"{self.config.platform} simulated crash")
      if self.stop_after is not None:
        await asyncio.sleep(self.stop_after)
      else:
        # Run until cancelled
        while self._running:
          await asyncio.sleep(0.05)
    except asyncio.CancelledError:
      pass
    finally:
      await self.stop()


# --- Fixtures ---


@pytest.fixture
def stub_agent():
  return Agent(model=MockModel(responses=["ok"]), instructions="test")


# --- Tests ---


class TestServe:
  @pytest.mark.asyncio
  async def test_no_interfaces_raises(self):
    with pytest.raises(ValueError, match="at least one interface"):
      await serve()

  @pytest.mark.asyncio
  async def test_single_clean_stop(self, stub_agent):
    iface = StubInterface(stub_agent, stop_after=0.1, platform="alpha")
    await serve(iface)
    # Should return normally after the interface stops
    assert iface.start_count == 1

  @pytest.mark.asyncio
  async def test_all_clean_stop(self, stub_agent):
    a = StubInterface(stub_agent, stop_after=0.1, platform="alpha")
    b = StubInterface(stub_agent, stop_after=0.15, platform="beta")
    await serve(a, b)
    assert a.start_count == 1
    assert b.start_count == 1

  @pytest.mark.asyncio
  async def test_failed_interface_restarts(self, stub_agent):
    """An interface that crashes should be restarted by the supervisor."""
    # Crasher fails at 0.05s, backoff is 1.0s, restart at ~1.05s.
    # Cancel at 2.0s to give time for at least one restart.
    fail_iface = StubInterface(stub_agent, fail_after=0.05, platform="crasher")

    task = asyncio.current_task()
    assert task is not None

    async def _cancel_after():
      await asyncio.sleep(2.0)
      task.cancel()

    cancel_task = asyncio.create_task(_cancel_after())

    with contextlib.suppress(asyncio.CancelledError):
      await serve(fail_iface)

    cancel_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await cancel_task

    # Should have been started more than once (initial + at least one restart)
    assert fail_iface.start_count >= 2

  @pytest.mark.asyncio
  async def test_clean_stop_not_restarted(self, stub_agent):
    """An interface that stops cleanly should NOT be restarted."""
    stopper = StubInterface(stub_agent, stop_after=0.05, platform="stopper")
    runner = StubInterface(stub_agent, stop_after=0.2, platform="runner")

    await serve(stopper, runner)

    assert stopper.start_count == 1  # Not restarted
    assert runner.start_count == 1

  @pytest.mark.asyncio
  async def test_cancellation_cleans_up(self, stub_agent):
    """External cancellation should cancel all interfaces and return cleanly."""
    a = StubInterface(stub_agent, platform="alpha")
    b = StubInterface(stub_agent, platform="beta")

    async def _cancel_after():
      await asyncio.sleep(0.15)
      serve_task.cancel()

    serve_task = asyncio.ensure_future(serve(a, b))
    cancel_task = asyncio.create_task(_cancel_after())

    # serve() should return cleanly after cancellation — no CancelledError
    await serve_task

    cancel_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await cancel_task

  @pytest.mark.asyncio
  async def test_name_parameter(self, stub_agent):
    """The name parameter should not cause errors."""
    iface = StubInterface(stub_agent, stop_after=0.05, platform="test")
    await serve(iface, name="my-supervisor")

  @pytest.mark.asyncio
  async def test_agent_aserve_delegates(self, stub_agent):
    """Agent.aserve() should delegate to the runtime."""
    iface = StubInterface(stub_agent, stop_after=0.1, platform="via-agent")
    await stub_agent.aserve(iface)
    assert iface.start_count == 1

  @pytest.mark.asyncio
  async def test_crash_and_clean_stop_coexist(self, stub_agent):
    """A crashing interface and a clean-stopping one should coexist."""
    crasher = StubInterface(stub_agent, fail_after=0.05, platform="crasher")
    stopper = StubInterface(stub_agent, stop_after=2.5, platform="stopper")

    async def _cancel_after():
      await asyncio.sleep(3.0)
      serve_task.cancel()

    serve_task = asyncio.ensure_future(serve(crasher, stopper))
    cancel_task = asyncio.create_task(_cancel_after())

    with contextlib.suppress(asyncio.CancelledError):
      await serve_task

    cancel_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await cancel_task

    # Crasher should have restarted multiple times
    assert crasher.start_count >= 2
    # Stopper should have run once
    assert stopper.start_count == 1
