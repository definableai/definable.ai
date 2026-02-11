"""Signal bot example using Definable interfaces.

Prerequisites:
  1. Register/link your phone number with signal-cli-rest-api
  2. Install the package:
        pip install definable[signal]

Usage:
  export SIGNAL_PHONE_NUMBER="+1234567890"
  export OPENAI_API_KEY="your-openai-key"
  python 02_signal_bot.py
"""

import asyncio
import os

from definable.agents import Agent
from definable.interfaces.signal import SignalConfig, SignalInterface
from definable.models.openai import OpenAIChat


async def main():
  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are a helpful assistant on Signal. Keep responses concise.",
  )

  # Option A: Auto-managed Docker container (recommended for local dev)
  # The container is started automatically and stopped when the interface exits.
  # Use docker_data_dir to persist Signal registration across restarts.
  config = SignalConfig(
    phone_number=os.environ["SIGNAL_PHONE_NUMBER"],
    manage_container=True,
    docker_data_dir="./signal-data",
  )

  # Option B: Bring your own container
  # Start the container yourself:
  #   docker run -d -p 8080:8080 \
  #     -v ./signal-data:/home/.local/share/signal-cli \
  #     bbernhard/signal-cli-rest-api
  #
  # config = SignalConfig(
  #   phone_number=os.environ["SIGNAL_PHONE_NUMBER"],
  #   api_base_url="http://localhost:8080",
  # )

  interface = SignalInterface(
    agent=agent,
    config=config,
  )

  async with interface:
    print("Signal bot is running! Press Ctrl+C to stop.")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
