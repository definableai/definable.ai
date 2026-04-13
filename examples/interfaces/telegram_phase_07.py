"""Phase 7 Test: Group Chat Intelligence.

Tests that the bot only responds in groups when @mentioned or
replied to (mention mode). In DMs it responds to everything.

Setup:
  1. Add the bot to a group chat
  2. Make sure the bot has permission to read messages (disable privacy mode via @BotFather)

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_07.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a helpful group assistant. Keep responses short (1-2 sentences).
Always mention that you were triggered by a mention or reply so the tester can confirm the behavior.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  group_mode="mention",  # Phase 7: only respond when mentioned or replied to
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 7 bot running — test in both DM and group chat")
    print()
    print("In DM: bot should respond to every message")
    print("In group: bot should ONLY respond when:")
    print("  - @mentioned (e.g. '@yourbot what time is it?')")
    print("  - Replied to (reply to a bot message)")
    print("  - A /command is sent")
    print("  Bot should IGNORE regular group messages without mention")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
