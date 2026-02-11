"""E2E tests for the Discord interface (requires real credentials)."""

import os

import pytest

from definable.agents.agent import Agent
from definable.interfaces.discord.config import DiscordConfig
from definable.interfaces.discord.interface import DiscordInterface

requires_discord = pytest.mark.skipif(
  not os.getenv("DISCORD_BOT_TOKEN"),
  reason="DISCORD_BOT_TOKEN environment variable not set",
)


@pytest.mark.e2e
@pytest.mark.discord
@requires_discord
class TestDiscordE2E:
  @pytest.fixture
  def bot_token(self):
    return os.environ["DISCORD_BOT_TOKEN"]

  @pytest.fixture
  def agent(self):
    from definable.models.openai import OpenAIChat

    return Agent(
      model=OpenAIChat(id="gpt-4o-mini"),
      instructions="You are a test bot. Reply briefly.",
    )

  @pytest.mark.asyncio
  async def test_bot_connects_to_gateway(self, agent, bot_token):
    """Verify the bot can connect to the Discord gateway."""
    interface = DiscordInterface(
      agent=agent,
      config=DiscordConfig(bot_token=bot_token),
    )
    await interface.start()
    assert interface._running is True
    assert interface._client is not None

    # Graceful shutdown
    await interface.stop()
    assert interface._running is False

  @pytest.mark.asyncio
  async def test_graceful_shutdown(self, agent, bot_token):
    """Verify the interface shuts down cleanly via context manager."""
    interface = DiscordInterface(
      agent=agent,
      config=DiscordConfig(bot_token=bot_token),
    )
    async with interface:
      assert interface._running is True
    assert interface._running is False
