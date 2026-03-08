"""Phase 5 Test: Callback Query Handling.

Tests that the bot can handle inline keyboard button presses.
Send any message — the bot responds with buttons. Pressing a
button triggers a registered callback handler.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_05.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface, InlineKeyboard, InlineButton


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses short.",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  handle_callback_queries=True,
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


# Register callback handlers
async def handle_greeting(callback_query):
  """Handle the 'greet' callback."""
  user = callback_query.get("from", {})
  name = user.get("first_name", "friend")
  return f"Hello {name}! 👋"


async def handle_joke(callback_query):
  """Handle the 'joke' callback."""
  return "Why do programmers prefer dark mode? Because light attracts bugs! 🐛"


async def handle_time(callback_query):
  """Handle the 'time' callback."""
  import datetime

  now = datetime.datetime.now().strftime("%H:%M:%S")
  return f"Current time: {now} ⏰"


interface.register_callback(r"^greet$", handle_greeting)
interface.register_callback(r"^joke$", handle_joke)
interface.register_callback(r"^time$", handle_time)


# Override _send_response to always include buttons
class ButtonTelegramInterface(TelegramInterface):
  async def _send_response(self, original_msg, response, raw_message):
    chat_id = original_msg.platform_chat_id
    api_chat_id = chat_id.split(":topic:")[0] if ":topic:" in chat_id else chat_id

    # Send the agent's text response first
    if response.content:
      await self._send_text(api_chat_id, response.content)

    # Then send a message with buttons
    kb = InlineKeyboard()
    kb.row(
      InlineButton("👋 Greet me", callback_data="greet"),
      InlineButton("😂 Tell a joke", callback_data="joke"),
    )
    kb.button("⏰ What time?", callback_data="time")
    kb.button("🔗 Visit Example", url="https://example.com")

    await self.send_with_buttons(api_chat_id, "Pick an action:", kb)


# Rebuild with the subclass
button_interface = ButtonTelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  handle_callback_queries=True,
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)
button_interface.register_callback(r"^greet$", handle_greeting)
button_interface.register_callback(r"^joke$", handle_joke)
button_interface.register_callback(r"^time$", handle_time)


async def main():
  async with button_interface:
    print("Phase 5 bot running — send any message to get buttons")
    print("Press the buttons to test callback handling")
    print("Press Ctrl+C to stop")
    await button_interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
