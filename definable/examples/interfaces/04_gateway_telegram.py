"""InterfaceGateway example — Telegram + Discord with shared hooks and identity linking.

Demonstrates the InterfaceGateway as a central coordination layer:
  - Gateway-level hooks that apply to ALL interfaces (logging, rate limiting)
  - Per-interface status tracking and lifecycle events
  - Cross-platform identity linking via the /link command
  - Shared sessions (optional, off by default)

How identity linking works:
  1. A user types /link on Telegram — the gateway generates a 6-character code
  2. The same user types /link ABC123 on Discord — the gateway validates the code
     and links both platform accounts to a single canonical user ID
  3. From that point on, memory and context are unified across platforms

Prerequisites:
  pip install 'definable[telegram,discord]'

Usage:
  export OPENAI_API_KEY="your-openai-key"
  export TELEGRAM_BOT_TOKEN="your-telegram-token"
  export DISCORD_BOT_TOKEN="your-discord-token"
  python definable/examples/interfaces/04_gateway_telegram.py
"""

import asyncio
import os
from typing import Optional

from definable.agent import Agent
from definable.agent.interface import SQLiteIdentityResolver
from definable.agent.interface.discord import DiscordInterface
from definable.agent.interface.gateway import (
  InterfaceErrorEvent,
  InterfaceGateway,
  InterfaceRestartedEvent,
  InterfaceStartedEvent,
  InterfaceStoppedEvent,
)
from definable.agent.interface.telegram import TelegramInterface
from definable.memory import FileStore, Memory
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool
from definable.utils.log import log_info

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
  """Get current weather for a city."""
  return f"It's 22°C and sunny in {city}."


# ---------------------------------------------------------------------------
# Gateway-level hooks (apply to ALL interfaces)
# ---------------------------------------------------------------------------


class RateLimitHook:
  """Simple per-user rate limiter — 1 message per second."""

  def __init__(self) -> None:
    self._last_seen: dict[str, float] = {}

  async def on_message_received(self, message) -> Optional[bool]:
    import time

    key = f"{message.platform}:{message.platform_user_id}"
    now = time.time()
    last = self._last_seen.get(key, 0)
    if now - last < 1.0:
      log_info(f"[rate-limit] Throttled {key}")
      return False  # veto — drop the message
    self._last_seen[key] = now
    return None  # pass through


class AuditHook:
  """Logs every message across all platforms for audit."""

  async def on_message_received(self, message) -> Optional[bool]:
    log_info(f"[audit] [{message.platform}] user={message.platform_user_id}: {message.text!r}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  # -- Agent setup --
  memory = Memory(store=FileStore("./memory"))
  agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[get_weather],
    instructions=(
      "You are a helpful assistant available on multiple platforms. "
      "Keep responses concise. If the user asks about linking accounts, "
      "tell them to type /link on one platform and then /link <CODE> on the other."
    ),
    name="gateway-bot",
    memory=memory,
  )

  # -- Identity resolver (cross-platform user unification) --
  resolver = SQLiteIdentityResolver("./gateway_identity.db")

  # Pre-link known users (admin setup).
  # In production, users link themselves via the /link command.
  # async with resolver:
  #   await resolver.link("telegram", os.environ.get("MY_TELEGRAM_ID", "12345"), "alice")
  #   await resolver.link("discord", os.environ.get("MY_DISCORD_ID", "67890"), "alice")

  # -- Create gateway --
  gateway = InterfaceGateway(
    agent,
    identity_resolver=resolver,
    enable_identity_linking=True,  # enables /link self-service flow
    link_code_ttl=300,  # codes expire after 5 minutes
  )

  # -- Add gateway-level hooks (apply to all interfaces) --
  gateway.add_hook(AuditHook())  # type: ignore[arg-type]
  gateway.add_hook(RateLimitHook())  # type: ignore[arg-type]

  # -- Register interfaces --
  telegram = TelegramInterface(
    bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
  )
  discord = DiscordInterface(
    bot_token=os.environ["DISCORD_BOT_TOKEN"],
  )

  gateway.add(telegram)
  gateway.add(discord)

  # -- Subscribe to lifecycle events --
  @agent._event_bus.on(InterfaceStartedEvent)
  def on_started(event: InterfaceStartedEvent) -> None:
    print(f"  [gateway] {event.platform} started ({event.interface_class})")

  @agent._event_bus.on(InterfaceStoppedEvent)
  def on_stopped(event: InterfaceStoppedEvent) -> None:
    print(f"  [gateway] {event.platform} stopped")

  @agent._event_bus.on(InterfaceErrorEvent)
  def on_error(event: InterfaceErrorEvent) -> None:
    print(f"  [gateway] {event.platform} ERROR: {event.error_message}")

  @agent._event_bus.on(InterfaceRestartedEvent)
  def on_restarted(event: InterfaceRestartedEvent) -> None:
    print(f"  [gateway] {event.platform} restarted (attempt #{event.restart_count}, backoff {event.backoff_seconds}s)")

  # -- Check statuses --
  print("Interface statuses before start:", gateway.statuses)
  print(f"Healthy: {gateway.is_healthy}")

  # -- Serve (blocks until Ctrl+C) --
  await gateway.aserve(name="gateway-bot")


if __name__ == "__main__":
  asyncio.run(main())
