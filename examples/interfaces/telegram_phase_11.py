"""Phase 11 Test: Forward Context.

Tests that forwarded messages include [Forwarded from X] context.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_11.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a message context bot. When you see [Forwarded from X],
acknowledge who the message was originally from and respond to the content.
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
    print("Phase 11 bot running — forward messages to test")
    print()
    print("Test plan:")
    print("  1. Forward a message from another user → bot says 'Forwarded from @user'")
    print("  2. Forward a message from a channel → bot says 'Forwarded from ChannelName'")
    print("  3. Send a normal message → no forward context")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
