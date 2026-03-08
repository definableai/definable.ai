"""Phase 19 Test: Media Groups.

Tests that multiple photos/videos sent as an album are combined
into a single message to the agent instead of separate messages.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_19.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a photo album bot. When you receive images, tell the user
how many images you received in this batch. Keep responses short.""",
)


class MediaGroupDebugInterface(TelegramInterface):
  async def _convert_inbound(self, raw_message):
    msg = await super()._convert_inbound(raw_message)
    if msg:
      n_images = len(msg.images) if msg.images else 0
      n_videos = len(msg.videos) if msg.videos else 0
      mg_id = msg.metadata.get("media_group_id", "none")
      print(f"  [MEDIA GROUP] id={mg_id} images={n_images} videos={n_videos}")
    return msg


interface = MediaGroupDebugInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  media_group_timeout=0.5,
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 19 bot running — send photo albums to test")
    print()
    print("Test plan:")
    print("  1. Select multiple photos and send as album")
    print("  2. Bot should receive ONE message with multiple images")
    print("  3. Check terminal: should show media_group_id with image count > 1")
    print("  4. Send a single photo — should work normally (no media group)")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
