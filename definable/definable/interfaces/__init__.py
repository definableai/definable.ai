"""
Definable Interfaces — Connect agents to messaging platforms.

This module provides the BaseInterface ABC and platform implementations
for exposing agents through messaging platforms like Telegram, Discord,
Slack, WhatsApp, and custom web UIs.

Quick Start (Telegram):
  from definable.agents import Agent
  from definable.models.openai import OpenAIChat
  from definable.interfaces.telegram import TelegramInterface, TelegramConfig

  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are a helpful assistant.",
  )
  interface = TelegramInterface(
    agent=agent,
    config=TelegramConfig(bot_token="YOUR_BOT_TOKEN"),
  )
  async with interface:
    await interface.serve_forever()

With Hooks:
  from definable.interfaces.hooks import LoggingHook, AllowlistHook

  interface.add_hook(LoggingHook())
  interface.add_hook(AllowlistHook(allowed_user_ids={"123456"}))
"""

from definable.interfaces.base import BaseInterface
from definable.interfaces.serve import serve
from definable.interfaces.config import InterfaceConfig
from definable.interfaces.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.interfaces.hooks import AllowlistHook, InterfaceHook, LoggingHook
from definable.interfaces.message import InterfaceMessage, InterfaceResponse
from definable.interfaces.session import InterfaceSession, SessionManager


# Lazy imports for platform implementations to avoid requiring their dependencies
def __getattr__(name: str):
  if name == "TelegramInterface":
    from definable.interfaces.telegram.interface import TelegramInterface

    return TelegramInterface
  if name == "TelegramConfig":
    from definable.interfaces.telegram.config import TelegramConfig

    return TelegramConfig
  if name == "DiscordInterface":
    from definable.interfaces.discord.interface import DiscordInterface

    return DiscordInterface
  if name == "DiscordConfig":
    from definable.interfaces.discord.config import DiscordConfig

    return DiscordConfig
  if name == "SignalInterface":
    from definable.interfaces.signal.interface import SignalInterface

    return SignalInterface
  if name == "SignalConfig":
    from definable.interfaces.signal.config import SignalConfig

    return SignalConfig
  if name in ("IdentityResolver", "SQLiteIdentityResolver", "PlatformIdentity"):
    from definable.interfaces import identity as _identity_mod

    return getattr(_identity_mod, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define lazy-loaded types for static analysis
TelegramInterface: type
TelegramConfig: type
DiscordInterface: type
DiscordConfig: type
SignalInterface: type
SignalConfig: type
IdentityResolver: type
SQLiteIdentityResolver: type
PlatformIdentity: type

__all__ = [
  # Core
  "BaseInterface",
  "InterfaceConfig",
  "serve",
  "InterfaceMessage",
  "InterfaceResponse",
  "InterfaceSession",
  "SessionManager",
  # Hooks
  "InterfaceHook",
  "LoggingHook",
  "AllowlistHook",
  # Errors
  "InterfaceError",
  "InterfaceConnectionError",
  "InterfaceAuthenticationError",
  "InterfaceRateLimitError",
  "InterfaceMessageError",
  # Platform implementations (lazy-loaded)
  "TelegramInterface",
  "TelegramConfig",
  "DiscordInterface",
  "DiscordConfig",
  "SignalInterface",
  "SignalConfig",
  # Identity resolution (lazy-loaded)
  "IdentityResolver",
  "SQLiteIdentityResolver",
  "PlatformIdentity",
]
