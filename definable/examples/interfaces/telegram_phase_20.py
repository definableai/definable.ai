"""Phase 20 Test: Reactions.

Tests that the bot processes emoji reactions on messages.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_20.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a reaction-aware bot. When you see [Reaction: emoji on message N],
acknowledge the reaction and respond with a fun comment about the emoji.
For regular messages, respond normally. Keep responses short.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  handle_reactions=True,  # Phase 20: opt-in
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 20 bot running — react to messages to test")
    print()
    print("Test plan:")
    print("  1. Send a message to the bot")
    print("  2. React to the bot's reply with an emoji (long-press → react)")
    print("  3. Bot should respond acknowledging the reaction")
    print("  4. Try different emoji reactions")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
