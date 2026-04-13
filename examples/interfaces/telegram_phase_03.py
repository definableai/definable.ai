"""Phase 3 Test: Message Editing.

Tests that the bot can send a message and then edit it.
Send any message — the bot sends a "Thinking..." placeholder,
waits 2 seconds, then edits it with the actual response.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_03.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


class EditingTelegramInterface(TelegramInterface):
  """Custom interface that demonstrates message editing."""

  async def _send_response(self, original_msg, response, raw_message):
    chat_id = original_msg.platform_chat_id
    api_chat_id = chat_id.split(":topic:")[0] if ":topic:" in chat_id else chat_id

    # Step 1: Send placeholder
    msg_id = await self._send_message(api_chat_id, "Thinking... 🤔")

    # Step 2: Wait
    await asyncio.sleep(2)

    # Step 3: Edit with actual response
    if response.content:
      pm = self._tg_config.parse_mode
      from definable.agent.interface.telegram.formatting import markdown_to_telegram_html

      text = markdown_to_telegram_html(response.content) if self._tg_config.auto_format and pm == "HTML" else response.content
      success = await self._edit_message(api_chat_id, msg_id, text, parse_mode=pm)
      if not success:
        # Fallback: send as new message
        await self._send_text(api_chat_id, response.content)


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses short (1-2 sentences).",
)

interface = EditingTelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  auto_format=True,
  parse_mode="HTML",
  streaming=False,  # Disable streaming so our custom _send_response is used
)


async def main():
  async with interface:
    print("Phase 3 bot running — send any message to test message editing")
    print('You should see "Thinking..." that gets replaced after 2 seconds')
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
