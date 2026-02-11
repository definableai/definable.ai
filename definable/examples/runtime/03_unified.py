"""Unified runtime: interface + webhook + cron + auth + hooks.

Demonstrates the full agent-centric runtime with all features combined.

Prerequisites:
  pip install 'definable-ai[runtime]'
  export OPENAI_API_KEY=sk-...
  export TELEGRAM_BOT_TOKEN=...  # Optional, for Telegram interface

Usage:
  python examples/runtime/03_unified.py
"""

import os

from definable.agents import Agent
from definable.auth import APIKeyAuth
from definable.models.openai import OpenAIChat
from definable.triggers import Cron, EventTrigger, Webhook

agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  name="UnifiedBot",
  instructions="You are a helpful assistant.",
)

# --- Auth (protects /run and authenticated webhooks) ---
agent.auth = APIKeyAuth(keys={"demo-key-123"})


# --- Agent-level hooks ---
@agent.before_request
async def log_before(context):
  print(f"  -> Starting run {context.run_id[:8]}...")


@agent.after_response
async def log_after(output):
  content_preview = (output.content or "")[:60]
  print(f"  <- Run {output.run_id[:8]} done: {content_preview}")


# --- Webhook trigger ---
@agent.on(Webhook("/github", auth=False))  # Public webhook (no auth)
async def handle_github(event):
  repo = event.body.get("repository", {}).get("name", "unknown") if event.body else "unknown"
  return f"Summarize this GitHub event for repo {repo}"


# --- Cron trigger ---
@agent.on(Cron("0 */6 * * *"))  # Every 6 hours
async def periodic_check(event):
  return "Give me a one-sentence status update."


# --- Event trigger ---
@agent.on(EventTrigger("user_signup"))
async def on_signup(event):
  name = event.body.get("name", "someone") if event.body else "someone"
  return f"Write a short welcome message for {name}."


# --- Optional: Telegram interface ---
def maybe_add_telegram():
  bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
  if not bot_token:
    return
  from definable.interfaces.telegram import TelegramConfig, TelegramInterface

  telegram = TelegramInterface(
    config=TelegramConfig(bot_token=bot_token),
  )
  agent.add_interface(telegram)
  print("Telegram interface registered")


if __name__ == "__main__":
  maybe_add_telegram()
  agent.serve(port=8000, dev=True)
