"""Phase 10 Test: Sticker Support.

Tests that the bot converts stickers to text descriptions
and optionally extracts static stickers as images.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_10.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a sticker interpreter bot. When you receive a sticker description
like [Sticker: emoji from 'SetName'], respond by acknowledging the sticker and
interpreting the emotion or meaning behind it. Keep responses short and fun.""",
)


class StickerDebugInterface(TelegramInterface):
  async def _convert_inbound(self, raw_message):
    msg = await super()._convert_inbound(raw_message)
    if msg and msg.text:
      print(f"  [TEXT] {msg.text}")
    if msg and msg.images:
      print(f"  [IMAGES] {len(msg.images)} image(s)")
    return msg


interface = StickerDebugInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 10 bot running — send stickers to test")
    print()
    print("Test plan:")
    print("  1. Send a static sticker → should show [Sticker: emoji from 'set'] + image")
    print("  2. Send an animated sticker → should show description, NO image")
    print("  3. Send same sticker twice → second should hit cache (check terminal)")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
