"""Phase 12 Test: Edited Message Processing.

Tests that edited messages are tagged with is_edit metadata.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_12.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="""You are a message tracking bot. Respond normally to messages.
If the message metadata indicates it was edited, acknowledge that explicitly
by saying something like 'I see you edited your message!'""",
)


class EditDebugInterface(TelegramInterface):
  async def _convert_inbound(self, raw_message):
    msg = await super()._convert_inbound(raw_message)
    if msg:
      is_edit = msg.metadata.get("is_edit", False)
      print(f"  [MSG] text='{msg.text}' is_edit={is_edit}")
    return msg


interface = EditDebugInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 12 bot running — edit messages to test")
    print()
    print("Test plan:")
    print("  1. Send a message → terminal shows is_edit=False")
    print("  2. Edit that message → terminal shows is_edit=True, bot re-processes it")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
