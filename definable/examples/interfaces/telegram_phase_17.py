"""Phase 17 Test: DM vs Group Policies.

Tests access policies for DM and group chats independently.
This example disables DMs and only allows a specific group.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  export TEST_GROUP_CHAT_ID="your-group-chat-id"  # negative number for groups
  .venv/bin/python definable/examples/interfaces/telegram_phase_17.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface


agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses short.",
)

group_id = os.environ.get("TEST_GROUP_CHAT_ID")
group_allowlist = [int(group_id)] if group_id else None

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  dm_policy="open",  # DMs are open
  group_policy="allowlist" if group_allowlist else "open",
  group_allowlist=group_allowlist,
  group_mode="always",  # Respond to all group messages for testing
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 17 bot running — test DM and group policies")
    print()
    print("Test plan:")
    print("  1. DM the bot → should respond (dm_policy=open)")
    if group_allowlist:
      print(f"  2. Message in allowed group ({group_id}) → should respond")
      print("  3. Message in any other group → should be ignored")
    else:
      print("  2. No TEST_GROUP_CHAT_ID set — group_policy=open, all groups allowed")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
