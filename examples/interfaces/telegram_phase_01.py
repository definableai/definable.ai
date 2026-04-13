"""Phase 1 Test: Markdown → HTML Conversion.

Tests that the bot auto-converts markdown in agent responses to
Telegram HTML. Send any message — the agent is instructed to respond
with rich markdown formatting so you can verify rendering.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_01.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a formatting test bot. For EVERY message the user sends,
respond with a message that demonstrates ALL of these markdown formats:

1. **Bold text** using double asterisks
2. *Italic text* using single asterisks
3. `inline code` using backticks
4. A fenced code block with a language tag
5. A [hyperlink](https://example.com)
6. ~~Strikethrough text~~
7. ||Spoiler text||
8. > A blockquote

Make the response natural and conversational while including all formats.
Keep it short — one example of each is enough.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  auto_format=True,  # Phase 1: enable markdown→HTML conversion
  parse_mode="HTML",  # Telegram parse mode
)


async def main():
  async with interface:
    print("Phase 1 bot running — send any message to test markdown→HTML conversion")
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
