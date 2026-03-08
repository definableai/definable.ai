"""Phase 18 Test: Location Messages.

Tests that the bot receives location and venue messages.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_18.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a location-aware bot. When you receive a location like
[Location: lat, lng], tell the user something interesting about that area.
When you receive a venue like [Venue: name, address (lat, lng)], acknowledge the place.
Keep responses short.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 18 bot running — send locations to test")
    print()
    print("Test plan:")
    print("  1. Send a location (tap paperclip → Location) → bot sees [Location: lat, lng]")
    print("  2. Send a venue (search for a place) → bot sees [Venue: name, address ...]")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
