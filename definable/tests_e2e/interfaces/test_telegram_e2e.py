"""E2E tests for the Telegram interface (requires real credentials)."""

import os

import pytest

from definable.agents.agent import Agent
from definable.interfaces.telegram.config import TelegramConfig
from definable.interfaces.telegram.interface import TelegramInterface

requires_telegram = pytest.mark.skipif(
  not os.getenv("TELEGRAM_BOT_TOKEN"),
  reason="TELEGRAM_BOT_TOKEN environment variable not set",
)


@pytest.mark.e2e
@pytest.mark.telegram
@requires_telegram
class TestTelegramE2E:
  @pytest.fixture
  def bot_token(self):
    return os.environ["TELEGRAM_BOT_TOKEN"]

  @pytest.fixture
  def agent(self):
    from definable.models.openai import OpenAIChat

    return Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      instructions="You are a test bot. Reply briefly.",
    )

  @pytest.mark.asyncio
  async def test_bot_connects_via_getme(self, agent, bot_token):
    """Verify the bot can connect and retrieve its identity."""
    interface = TelegramInterface(
      agent=agent,
      config=TelegramConfig(bot_token=bot_token),
    )
    # Start the interface (verifies token via getMe)
    await interface.start()
    assert interface._running is True

    # Graceful shutdown
    await interface.stop()
    assert interface._running is False
    assert interface._client is None  # type: ignore[unreachable]

  @pytest.mark.asyncio
  async def test_graceful_shutdown(self, agent, bot_token):
    """Verify the interface shuts down cleanly via context manager."""
    interface = TelegramInterface(
      agent=agent,
      config=TelegramConfig(bot_token=bot_token),
    )
    async with interface:
      assert interface._running is True
    assert interface._running is False
