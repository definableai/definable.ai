"""Multi-interface example — run Telegram + Discord with a single agent.

Demonstrates the canonical pattern for multi-channel agents:
  Agent(interfaces=[...]).serve()

When 2+ interfaces are registered, the agent automatically creates an
InterfaceGateway for production-grade supervision (auto-restart with
exponential backoff).

Prerequisites:
  pip install 'definable[telegram,discord]'

Usage:
  export OPENAI_API_KEY="your-openai-key"
  export TELEGRAM_BOT_TOKEN="your-telegram-token"
  export DISCORD_BOT_TOKEN="your-discord-token"
  python definable/examples/interfaces/02_multi_interface.py
"""

import os

from definable.agent import Agent
from definable.agent.interface.discord import DiscordInterface
from definable.agent.interface.telegram import TelegramInterface
from definable.agent.tracing import JSONLExporter, Tracing
from definable.memory import Memory, SQLiteStore

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant. Keep responses concise.",
  name="multi-bot",
  memory=Memory(store=SQLiteStore()),
  audio_transcriber=True,
  tracing=Tracing(exporters=[JSONLExporter("./traces")]),
  interfaces=[
    TelegramInterface(bot_token=os.environ["TELEGRAM_BOT_TOKEN"]),
    DiscordInterface(bot_token=os.environ["DISCORD_BOT_TOKEN"]),
  ],
)

if __name__ == "__main__":
  agent.serve()
