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

from definable.agent import Agent
from definable.agent.interface.whatsapp import WhatsAppInterface

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="You are a helpful assistant on WhatsApp. Keep responses concise.",
  interfaces=WhatsAppInterface(
    provider="baileys",
    auth_dir="./whatsapp-auth",  # credentials stored here after QR scan
    markdown_conversion=True,  # convert markdown to WhatsApp formatting
  ),
)

if __name__ == "__main__":
  agent.serve()
