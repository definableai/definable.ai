"""Phase 14 Test: Active Rate Limiting.

Tests that the bot rate-limits users who send too many messages.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_14.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses very short (1 sentence).",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  rate_limit_messages_per_minute=5,  # Low limit for testing
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 14 bot running — spam messages to test rate limiting")
    print()
    print("Test plan:")
    print("  1. Send 5 messages quickly — all should get responses")
    print("  2. Send a 6th message — should get 'too fast' cooldown message")
    print("  3. Wait 60 seconds, try again — should work normally")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
