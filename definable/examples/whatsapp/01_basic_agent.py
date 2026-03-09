"""Example 1: Basic Agent + Baileys WhatsApp.

The simplest possible setup: an Agent connected to WhatsApp via Baileys.
On first run it shows a QR code in terminal — scan it with your phone.
After that, anyone who messages you gets a response from the agent.

Requirements:
  - Node.js >= 18 (check: node --version)
  - pip install websockets  (if not already installed)
  - An OpenAI API key in OPENAI_API_KEY env var

Run:
  .venv/bin/python definable/examples/whatsapp/01_basic_agent.py
"""

import asyncio
import signal

from definable.agent import Agent
from definable.agent.interface.whatsapp import WhatsAppInterface


async def main():
  # 1. Create an agent
  agent = Agent(
    model="openai/gpt-4o-mini",
    instructions="You are a very nice guy how want friends and chat with them",
  )

  # 2. Create the WhatsApp interface with Baileys
  whatsapp = WhatsAppInterface(
    provider="baileys",
    auth_dir="./whatsapp-auth-anandesh",  # credentials stored here after QR scan
    markdown_conversion=True,  # convert markdown to WhatsApp formatting
    # verbose=True,               # enable to see bridge-level debug logs (noisy)
  )

  # 3. Bind and start
  whatsapp.bind(agent)
  await whatsapp.start()

  print("\n[ready] WhatsApp agent is running.")
  print("[ready] If this is your first time, scan the QR code above with WhatsApp.")
  print("[ready] Send a message to your WhatsApp number to test.")
  print("[ready] Press Ctrl+C to stop.\n")

  # 4. Wait for shutdown signal (Ctrl+C or SIGTERM)
  stop = asyncio.Event()
  loop = asyncio.get_running_loop()
  for sig in (signal.SIGINT, signal.SIGTERM):
    loop.add_signal_handler(sig, stop.set)

  await stop.wait()

  # 5. Clean shutdown
  print("\n[stopping] Shutting down...")
  await whatsapp.stop()
  print("[stopped] Done.")


if __name__ == "__main__":
  asyncio.run(main())
