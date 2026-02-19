"""Remote macOS control via Telegram.

Allows anyone with access to your Telegram bot to control your Mac remotely.
The MacOS skill bridges between Telegram messages and macOS system actions.

Architecture:
  Telegram message → Definable Agent → MacOS skill → Desktop Bridge → macOS

Prerequisites:
  1. Build and run the Desktop Bridge:
       cd definable/desktop-bridge
       swift build -c release
       .build/release/DesktopBridge
  2. Create a Telegram bot via @BotFather and get the token.
  3. Install dependencies:
       pip install 'definable[telegram,desktop]'
  4. Set environment variables:
       export TELEGRAM_BOT_TOKEN="your-bot-token"
       export OPENAI_API_KEY="sk-proj-..."
       # Optional: restrict to your Telegram user ID for security
       export ALLOWED_TELEGRAM_USER_ID="123456789"

Security recommendations:
  - Set ALLOWED_TELEGRAM_USER_ID to allow only yourself
  - Use MacOS(enable_input=False) for read-only access from untrusted users
  - Use allowed_apps={"Safari", "TextEdit"} to limit which apps can be controlled
  - The bridge binds to 127.0.0.1 only — not accessible from the network

Usage:
  python definable/examples/interfaces/04_desktop_control_via_telegram.py
"""

import asyncio
import os

from definable.agent import Agent
from definable.agent.interface.telegram import TelegramConfig, TelegramInterface
from definable.model.openai import OpenAIChat
from definable.skill.builtin.macos import MacOS


async def main() -> None:
  model = OpenAIChat(id="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

  # Read optional security config
  allowed_user_id = os.environ.get("ALLOWED_TELEGRAM_USER_ID")
  allowed_user_ids = {int(allowed_user_id)} if allowed_user_id else None

  # --- MacOS skill: configure safety controls ---
  # For personal use with your own Telegram user ID, full access is fine.
  # For shared bots, restrict apps and disable write operations.
  macos_skill = MacOS(
    # Uncomment to restrict which apps can be controlled:
    # allowed_apps={"Safari", "TextEdit", "Finder"},
    # Uncomment for read-only mode (no mouse/keyboard, no file writes):
    # enable_input=False,
    # enable_file_write=False,
  )

  agent = Agent(
    model=model,
    skills=[macos_skill],
    instructions=(
      "You control a Mac remotely via Telegram. "
      "Before interacting with anything, take a screenshot to understand the current state. "
      "Describe what you see and what you're doing in each step. "
      "Always confirm successful actions with a follow-up screenshot. "
      "If the user asks you to do something potentially destructive, ask for confirmation first."
    ),
  )

  # Build the Telegram config
  telegram_config = TelegramConfig(
    bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
    # Restrict to specific user IDs for security
    allowed_user_ids=allowed_user_ids,  # type: ignore[arg-type]
    # Auto-parse markdown so screenshots render correctly
    parse_mode="Markdown",
  )

  interface = TelegramInterface(agent=agent, config=telegram_config)

  print("Desktop control Telegram bot is running!")
  print("  Bridge: http://127.0.0.1:7777")
  if allowed_user_ids:
    print(f"  Allowed user IDs: {allowed_user_ids}")
  print("  Send any message to your bot to start controlling your Mac.")
  print("  Press Ctrl+C to stop.\n")

  async with interface:
    await interface.serve_forever()


if __name__ == "__main__":
  asyncio.run(main())
