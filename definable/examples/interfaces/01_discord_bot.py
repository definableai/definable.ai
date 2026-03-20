"""Discord bot example using Definable interfaces.

Prerequisites:
  1. Create a Discord bot at https://discord.com/developers/applications
  2. Enable the MESSAGE_CONTENT privileged intent in Bot settings
  3. Invite the bot to your server with the "Send Messages" and "Read Messages" permissions
  4. Install the discord.py dependency:
       pip install 'definable[discord]'

Usage:
  export DISCORD_BOT_TOKEN="your-bot-token"
  export OPENAI_API_KEY="your-openai-key"
  python definable/examples/interfaces/01_discord_bot.py
"""

import os

from definable.agent import Agent
from definable.agent.interface.discord import DiscordInterface
from definable.memory import Memory, SQLiteStore

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant on Discord. Keep responses concise.",
  memory=Memory(store=SQLiteStore("./example_memory.db")),
  interfaces=DiscordInterface(
    bot_token=os.environ["DISCORD_BOT_TOKEN"],
    # Optional: restrict to specific channels or guilds
    # allowed_guild_ids=[123456789],
    # allowed_channel_ids=[987654321],
  ),
)

if __name__ == "__main__":
  agent.serve()
