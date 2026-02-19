"""Unified runtime: interface + webhook + cron + auth + hooks.

Demonstrates the full agent-centric runtime with all features combined.

Prerequisites:
  pip install 'definable[runtime]'
  export OPENAI_API_KEY=sk-...
  export TELEGRAM_BOT_TOKEN=...  # Optional, for Telegram interface

Usage:
  python examples/runtime/03_unified.py
"""

import os

from definable.agent import Agent
from definable.agent.auth import APIKeyAuth
from definable.model.openai import OpenAIChat
from definable.agent.trigger import Cron, EventTrigger, Webhook

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


# --- Optional: Telegram interface with AllowlistAuth ---
def maybe_add_telegram():
  bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
  if not bot_token:
    return
  from definable.agent.auth import AllowlistAuth
  from definable.agent.interface.telegram import TelegramConfig, TelegramInterface

  # Only allow specific Telegram users (set via env var or hardcode)
  allowed = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
  allowed_ids = {uid.strip() for uid in allowed.split(",") if uid.strip()}

  telegram = TelegramInterface(
    config=TelegramConfig(bot_token=bot_token),
    auth=AllowlistAuth(user_ids=allowed_ids) if allowed_ids else None,
  )
  agent.add_interface(telegram)
  print("Telegram interface registered")


if __name__ == "__main__":
  maybe_add_telegram()
  agent.serve(port=8000, dev=True)
