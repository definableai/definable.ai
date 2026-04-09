"""Phase 13 Test: Typing Indicator Circuit Breaker.

Tests that the bot gracefully handles typing indicator failures
without crashing. The circuit breaker suspends typing after
repeated failures.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_13.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses short.",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  typing_indicator=True,
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 13 bot running — send messages rapidly to test typing resilience")
    print()
    print("Test plan:")
    print("  1. Send several messages quickly — bot should respond to all")
    print("  2. No crashes even if typing indicator fails internally")
    print("  3. The typing indicator ('typing...') should appear in the chat")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
