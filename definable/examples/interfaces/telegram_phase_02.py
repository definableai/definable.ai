"""Phase 2 Test: Smart HTML Chunking.

Tests that long messages split across multiple Telegram messages
without breaking HTML tags. The agent is instructed to produce
a very long response with formatting throughout.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_02.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a verbose formatting test bot. When the user sends any message,
respond with an EXTREMELY long message (at least 5000 characters) that includes:

1. Multiple sections with **bold headers**
2. Paragraphs with *italic text* mixed in
3. Several fenced code blocks (Python, JavaScript, etc.) — make them realistic and long
4. Numbered lists with `inline code` in items
5. Multiple [hyperlinks](https://example.com) scattered throughout
6. ~~Strikethrough~~ and ||spoiler|| text in various places
7. > Blockquotes between sections

The key requirement: make it LONG enough that Telegram must split it into multiple messages
(over 4096 characters). Use real-looking technical content — like a tutorial on async Python
or a guide to REST API design. The formatting must be consistent throughout the entire response.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 2 bot running — send any message to test long HTML chunking")
    print("The bot will respond with a very long formatted message split across multiple messages")
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
