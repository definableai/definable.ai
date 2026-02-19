"""Multi-interface example — run Telegram + Discord with a single agent.

This example shows how to use ``serve()`` to run multiple messaging
platform interfaces concurrently.  If one interface crashes it is
automatically restarted with exponential backoff.

It also demonstrates cross-platform identity resolution using
``SQLiteIdentityResolver``, so a user recognized on both Telegram
and Discord shares a single unified memory.

Prerequisites:
  pip install 'definable[telegram,discord]'

Usage:
  export OPENAI_API_KEY="your-openai-key"
  export TELEGRAM_BOT_TOKEN="your-telegram-token"
  export DISCORD_BOT_TOKEN="your-discord-token"
  python definable/examples/interfaces/03_multi_interface.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.tracing import Tracing, JSONLExporter
from definable.agent.interface import serve
from definable.agent.interface.discord import DiscordConfig, DiscordInterface
from definable.agent.interface.identity import SQLiteIdentityResolver
from definable.agent.interface.telegram import TelegramConfig, TelegramInterface
from definable.memory import Memory, SQLiteStore
from definable.model.openai import OpenAIChat

# Set these environment variables before running:
#   export OPENAI_API_KEY="sk-proj-..."
#   export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
#   export DISCORD_BOT_TOKEN="your-discord-bot-token"
#   export VOYAGEAI_API_KEY="pa-..."


async def main():
  memory = Memory(store=SQLiteStore("./example_memory.db"))
  agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="You are a helpful assistant. Keep responses concise.",
    name="multi-bot",
    memory=memory,
    tracing=Tracing(
      exporters=[JSONLExporter("./traces")],
    ),
  )

  # Identity resolver — maps platform user IDs to canonical IDs so that
  # the same person on Telegram and Discord shares a single memory store.
  identity = SQLiteIdentityResolver("./identity.db")

  # Pre-link known users (in production, use a /link command or admin tool).
  async with identity:
    await identity.link("telegram", "12345678", "alice", username="Alice")
    await identity.link("discord", "987654321", "alice", username="Alice")

  telegram = TelegramInterface(
    agent=agent,
    config=TelegramConfig(bot_token=os.environ["TELEGRAM_BOT_TOKEN"]),
  )

  discord = DiscordInterface(
    agent=agent,
    config=DiscordConfig(bot_token=os.environ["DISCORD_BOT_TOKEN"]),
  )

  # serve() runs both interfaces concurrently and propagates the shared
  # identity resolver. If one crashes it auto-restarts; Ctrl+C stops everything.
  await serve(telegram, discord, name="multi-bot")


if __name__ == "__main__":
  asyncio.run(main())
