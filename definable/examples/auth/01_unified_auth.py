"""Unified authentication across HTTP, Telegram, and Discord.

Demonstrates every auth feature in a single agent:
  1. APIKeyAuth     — protects the HTTP /run endpoint
  2. AllowlistAuth  — per-interface user-ID allowlists (Telegram, Discord)
  3. CompositeAuth  — combines providers into a single auth object
  4. auth=False     — public webhook endpoint (no auth)
  5. Agent hooks    — before_request / after_response with auth context
  6. Tools          — agent has real tools so it does useful work

Prerequisites:
  pip install 'definable[runtime]'
  export OPENAI_API_KEY=sk-...

  # Optional — add one or both messaging interfaces:
  pip install 'definable[telegram]'
  export TELEGRAM_BOT_TOKEN=...
  export TELEGRAM_ALLOWED_USERS=12345,67890   # comma-separated user IDs

  pip install 'definable[discord]'
  export DISCORD_BOT_TOKEN=...
  export DISCORD_ALLOWED_USERS=111222,333444   # comma-separated user IDs

Usage:
  python definable/examples/auth/01_unified_auth.py
"""

import os
from datetime import datetime, timezone

from definable.agents import Agent
from definable.auth import AllowlistAuth, APIKeyAuth, CompositeAuth
from definable.knowledge.embedders.voyageai import VoyageAIEmbedder
from definable.memory import CognitiveMemory
from definable.memory.store import SQLiteMemoryStore
from definable.models.openai import OpenAIChat
from definable.tools.decorator import tool
from definable.triggers import Webhook

# ---------------------------------------------------------------------------
# Tools — give the agent something real to do
# ---------------------------------------------------------------------------


@tool
def get_server_time() -> str:
  """Get the current server time in UTC."""
  return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def check_system_status(service: str) -> str:
  """Check the status of a system service (mock)."""
  statuses = {
    "api": "healthy — 12ms p99 latency",
    "database": "healthy — 3 active connections",
    "cache": "healthy — 94% hit rate",
    "queue": "degraded — 1200 messages pending",
  }
  return statuses.get(service.lower(), f"Unknown service: {service}")


@tool
def list_services() -> str:
  """List all available services to check."""
  return "Available services: api, database, cache, queue"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------
store = SQLiteMemoryStore("./test_memory.db")
memory = CognitiveMemory(
  store=store,
  token_budget=500,
  distillation_model=OpenAIChat(id="gpt-5.2", api_key=os.environ["OPENAI_API_KEY"]),
  embedder=VoyageAIEmbedder(id="voyage-4-lite", api_key=os.environ["VOYAGEAI_API_KEY"]),
)

agent = Agent(
  model=OpenAIChat(id="gpt-4o-mini"),
  name="AuthDemoBot",
  instructions=("You are a system-operations assistant. Use your tools to answer questions about server time and service health."),
  tools=[get_server_time, check_system_status, list_services],
  memory=memory,
)


# ---------------------------------------------------------------------------
# Auth configuration — CompositeAuth covers HTTP + messaging
# ---------------------------------------------------------------------------

# Build per-platform AllowlistAuth providers from env vars
_telegram_users = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
_discord_users = os.environ.get("DISCORD_ALLOWED_USERS", "")

_telegram_ids = {uid.strip() for uid in _telegram_users.split(",") if uid.strip()}
_discord_ids = {uid.strip() for uid in _discord_users.split(",") if uid.strip()}

# Assemble the composite: APIKeyAuth handles HTTP, AllowlistAuth handles each platform
_providers = [APIKeyAuth(keys={"demo-key-123"})]
if _telegram_ids:
  _providers.append(AllowlistAuth(user_ids=_telegram_ids, platforms={"telegram"}))
if _discord_ids:
  _providers.append(AllowlistAuth(user_ids=_discord_ids, platforms={"discord"}))

agent.auth = CompositeAuth(*_providers)


# ---------------------------------------------------------------------------
# Agent hooks — log auth context on every run
# ---------------------------------------------------------------------------


@agent.before_request
async def log_before(context):
  auth_info = ""
  if hasattr(context, "auth") and context.auth:
    auth_info = f" [user={context.auth.user_id}, method={context.auth.metadata.get('auth_method', 'api_key')}]"
  print(f"  -> Run {context.run_id[:8]} starting{auth_info}")


@agent.after_response
async def log_after(output):
  preview = (output.content or "")[:80]
  print(f"  <- Run {output.run_id[:8]} done: {preview}")


# ---------------------------------------------------------------------------
# Public webhook — auth=False bypasses all auth providers
# ---------------------------------------------------------------------------


@agent.on(Webhook("/health", auth=False))
async def health_check(event):
  return "Report current server time and the status of all services."


# ---------------------------------------------------------------------------
# Optional interfaces — each guarded by env vars
# ---------------------------------------------------------------------------


def maybe_add_telegram():
  """Register Telegram interface if TELEGRAM_BOT_TOKEN is set."""
  bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
  if not bot_token:
    return
  from definable.interfaces.telegram import TelegramConfig, TelegramInterface

  telegram = TelegramInterface(
    config=TelegramConfig(bot_token=bot_token),
    auth=AllowlistAuth(user_ids=_telegram_ids) if _telegram_ids else None,
  )
  agent.add_interface(telegram)
  print(f"Telegram interface registered (allowed users: {_telegram_ids or 'all'})")


def maybe_add_discord():
  """Register Discord interface if DISCORD_BOT_TOKEN is set."""
  bot_token = os.environ.get("DISCORD_BOT_TOKEN")
  if not bot_token:
    return
  from definable.interfaces.discord import DiscordConfig, DiscordInterface

  discord = DiscordInterface(
    config=DiscordConfig(bot_token=bot_token),
    auth=AllowlistAuth(user_ids=_discord_ids) if _discord_ids else None,
  )
  agent.add_interface(discord)
  print(f"Discord interface registered (allowed users: {_discord_ids or 'all'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  maybe_add_telegram()
  maybe_add_discord()
  agent.serve(port=8000, dev=True)
