"""Phase 21 Test: TelegramOutputSkill + Thinking Indicator.

Tests that:
  1. The agent can send inline buttons by itself (via the skill tool)
  2. The thinking indicator shows "Thinking..." when agent has thinking enabled
  3. Streaming doesn't double-send responses

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_21_skill.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface
from definable.agent.interface.telegram.skill import TelegramOutputSkill
from definable.agent.reasoning.thinking import Thinking


agent = Agent(
  model="openai/gpt-4o-mini",
  skills=[TelegramOutputSkill()],
  thinking=Thinking(enabled=True),
  instructions="""You are a helpful assistant connected to Telegram.
When the user asks you to make a choice or decision, present options as inline buttons.
For example, if they say "pick a color", use telegram_reply_buttons to show color options.
Always think before answering complex questions. Keep responses short.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=True,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 21 bot running — test skill + thinking")
    print()
    print("Test plan:")
    print("  1. Send 'pick a color' -> bot should show buttons (Red, Blue, Green)")
    print("  2. Send a complex question -> should see 'Thinking...' placeholder")
    print("  3. Verify no double messages (streaming fix)")
    print("  4. Click a button -> bot should respond to the choice")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
