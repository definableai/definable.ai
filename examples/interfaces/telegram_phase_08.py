"""Phase 8 Test: Forum/Topic Support.

Tests that the bot maintains separate conversation sessions per
forum topic. Messages in different topics get independent context.

Setup:
  1. Create a supergroup with forum topics enabled
  2. Add the bot to that group
  3. Create at least 2 topics

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_08.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramInterface
from definable.memory import Memory


agent = Agent(
  model="openai/gpt-4o-mini",
  memory=Memory(),  # In-memory session history
  instructions="""You are a helpful assistant. You remember what was said in this conversation.
When the user asks what you remember or what was discussed, list the topics from this session.
Always start your response with the session context: "In this thread, we've discussed: ..."
If nothing was discussed yet, say "This is our first message in this thread!"
Keep responses short.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  enable_forum_topics=True,  # Phase 8: topic-based session isolation
  group_mode="always",  # Respond to all messages in group for easier testing
  streaming=False,
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 8 bot running — test in a forum-enabled supergroup")
    print()
    print("Test plan:")
    print("  1. In Topic A: say 'My name is Alice'")
    print("  2. In Topic B: say 'My name is Bob'")
    print("  3. In Topic A: ask 'What is my name?' → should say Alice")
    print("  4. In Topic B: ask 'What is my name?' → should say Bob")
    print("  5. In General topic: should be a separate session too")
    print()
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
