"""Phase 9 Test: Video/VideoNote/Animation Support.

Tests that the bot correctly receives and acknowledges video,
video note (circular), and animation (GIF) messages.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_09.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a media analysis bot. When you receive media, describe what you know about it.
If you receive video content, acknowledge it and mention any metadata available.
If there's no media, just chat normally. Keep responses short.""",
)


class MediaDebugInterface(TelegramInterface):
  """Subclass that logs media details for verification."""

  async def _convert_inbound(self, raw_message):
    msg = await super()._convert_inbound(raw_message)
    if msg is None:
      return None

    # Log media details for debugging
    if msg.videos:
      for v in msg.videos:
        print(f"  [VIDEO] url={v.url}, mime={v.mime_type}, duration={v.duration}, w={v.width}, h={v.height}")
    if msg.images:
      for img in msg.images:
        print(f"  [IMAGE] url={img.url}")
    if msg.audio:
      for a in msg.audio:
        print(f"  [AUDIO] url={a.url}, mime={a.mime_type}")

    return msg


interface = MediaDebugInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 9 bot running — send media to test extraction")
    print()
    print("Test plan:")
    print("  1. Send a video file → bot should acknowledge video")
    print("  2. Record a video note (circular) → bot should acknowledge it")
    print("  3. Send a GIF/animation → bot should acknowledge it")
    print("  4. Send a video with a caption → caption should be the text")
    print("  5. Check terminal for [VIDEO] logs with metadata")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
