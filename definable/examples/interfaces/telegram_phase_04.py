"""Phase 4 Test: Response Streaming.

Tests that the bot streams its response by editing the message
in real-time as tokens arrive from the model. You should see
text appearing progressively in a single message.

Usage:
  export TELEGRAM_BOT_TOKEN="your-bot-token"
  .venv/bin/python definable/examples/interfaces/telegram_phase_04.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.tool.decorator import tool
from definable.agent.interface.telegram import TelegramInterface


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  return f"The weather in {city} is 22°C, partly cloudy with a gentle breeze."


agent = Agent(
  model="openai/gpt-4o-mini",
  tools=[get_weather],
  instructions="""You are a helpful assistant. When asked about weather, use the get_weather tool.
For other questions, give detailed responses (3-5 paragraphs) so the streaming effect is visible.
Use **bold**, *italic*, and `code` formatting in your responses.""",
)

interface = TelegramInterface(
  agent=agent,
  bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  streaming=True,  # Phase 4: enable streaming
  stream_edit_interval=1.0,  # Edit every 1 second
  stream_min_chars=30,  # Send first message after 30 chars
  stream_tool_indicator=True,  # Show "Using tool: X..." during tool calls
  auto_format=True,
  parse_mode="HTML",
)


async def main():
  async with interface:
    print("Phase 4 bot running — send any message to test response streaming")
    print("You should see text appearing progressively in a single message")
    print('Try: "What is the weather in Tokyo?" to test tool indicators')
    print("Press Ctrl+C to stop")
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
