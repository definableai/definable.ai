"""E2E tests for the Signal interface (requires real signal-cli-rest-api)."""

import os

import pytest

from definable.agents.agent import Agent
from definable.interfaces.signal.config import SignalConfig
from definable.interfaces.signal.interface import SignalInterface

requires_signal = pytest.mark.skipif(
  not (os.getenv("SIGNAL_PHONE_NUMBER") and os.getenv("SIGNAL_API_URL")),
  reason="SIGNAL_PHONE_NUMBER and SIGNAL_API_URL environment variables not set",
)


@pytest.mark.e2e
@pytest.mark.signal
@requires_signal
class TestSignalE2E:
  @pytest.fixture
  def phone_number(self):
    return os.environ["SIGNAL_PHONE_NUMBER"]

  @pytest.fixture
  def api_url(self):
    return os.environ.get("SIGNAL_API_URL", "http://localhost:8080")

  @pytest.fixture
  def agent(self):
    from definable.models.openai import OpenAIChat

    return Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      instructions="You are a test bot. Reply briefly.",
    )

  @pytest.mark.asyncio
  async def test_connects_to_rest_api(self, agent, phone_number, api_url):
    """Verify the interface can connect to signal-cli-rest-api."""
    interface = SignalInterface(
      agent=agent,
      config=SignalConfig(
        phone_number=phone_number,
        api_base_url=api_url,
      ),
    )
    await interface.start()
    assert interface._running is True

    # Graceful shutdown
    await interface.stop()
    assert interface._running is False
    assert interface._client is None  # type: ignore[unreachable]

  @pytest.mark.asyncio
  async def test_graceful_shutdown(self, agent, phone_number, api_url):
    """Verify the interface shuts down cleanly via context manager."""
    interface = SignalInterface(
      agent=agent,
      config=SignalConfig(
        phone_number=phone_number,
        api_base_url=api_url,
      ),
    )
    async with interface:
      assert interface._running is True
    assert interface._running is False
