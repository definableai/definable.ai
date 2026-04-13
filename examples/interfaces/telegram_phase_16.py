"""Phase 16 Test: Command Menu Sync.

Tests that the bot syncs its command menu on startup.
After running, tap the "/" button in Telegram to see the menu.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_16.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a helpful assistant with commands.
If the user sends /help, list available commands.
If the user sends /about, describe yourself.
If the user sends /joke, tell a joke.
Otherwise respond normally.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  commands={
    "help": "Show available commands",
    "about": "Learn about this bot",
    "joke": "Get a random joke",
  },
  sync_commands_on_startup=True,
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 16 bot running — check the command menu in Telegram")
    print()
    print("Test plan:")
    print("  1. Open the bot chat in Telegram")
    print('  2. Tap the "/" button (or type /) → should see help, about, joke')
    print("  3. Select each command → bot should respond appropriately")
    print("  4. Check terminal for 'Synced 3 bot commands' log")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
