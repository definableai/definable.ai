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

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.discord import DiscordConfig, DiscordInterface
from definable.memory import Memory, SQLiteStore
from definable.model.openai import OpenAIChat

# Set these environment variables before running:
#   export DISCORD_BOT_TOKEN="your-discord-bot-token"
#   export OPENAI_API_KEY="sk-proj-..."
#   export VOYAGEAI_API_KEY="pa-..."


class ContentFilterHook:
  """Example hook that filters forbidden content from responses."""

  async def on_after_respond(self, message, response, session):
    if response.content and "forbidden" in response.content.lower():
      response.content = "I can't help with that."
    return response


async def main(user_id: str):
  memory = Memory(store=SQLiteStore("./example_memory.db"))
  agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
    instructions="You are a helpful assistant on Discord. Keep responses concise.",
    memory=memory,
  )

  interface = DiscordInterface(
    agent=agent,
    config=DiscordConfig(
      bot_token=os.environ["DISCORD_BOT_TOKEN"],
      # Optional: restrict to specific channels or guilds
      # allowed_guild_ids=[123456789],
      # allowed_channel_ids=[987654321],
      # Optional: only respond to messages starting with !ask
      # command_prefix="!ask",
    ),
  )

  interface.add_hook(ContentFilterHook())  # type: ignore[arg-type]

  async with interface:
    print("Discord bot is running! Press Ctrl+C to stop.")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main(user_id="example-user-id"))
