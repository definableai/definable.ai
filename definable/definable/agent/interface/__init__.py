"""
Definable Interfaces — Connect agents to messaging platforms.

This module provides the BaseInterface ABC and platform implementations
for exposing agents through messaging platforms like Telegram, Discord,
Slack, WhatsApp, and custom web UIs.

Quick Start (Telegram):
  from definable.agent import Agent
  from definable.model.openai import OpenAIChat
  from definable.agent.interface.telegram import TelegramInterface, TelegramConfig

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
  from definable.agent.interface.hooks import LoggingHook, AllowlistHook

  interface.add_hook(LoggingHook())
  interface.add_hook(AllowlistHook(allowed_user_ids={"123456"}))
"""

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.serve import serve
from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.agent.interface.hooks import AllowlistHook, InterfaceHook, LoggingHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import InterfaceSession, SessionManager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.interface.desktop.config import DesktopConfig
  from definable.agent.interface.desktop.interface import DesktopInterface
  from definable.agent.interface.discord.config import DiscordConfig
  from definable.agent.interface.discord.interface import DiscordInterface
  from definable.agent.interface.identity import IdentityResolver, PlatformIdentity, SQLiteIdentityResolver
  from definable.agent.interface.signal.config import SignalConfig
  from definable.agent.interface.signal.interface import SignalInterface
  from definable.agent.interface.telegram.config import TelegramConfig
  from definable.agent.interface.telegram.interface import TelegramInterface


# Lazy imports for platform implementations to avoid requiring their dependencies
def __getattr__(name: str):
  if name == "DesktopInterface":
    from definable.agent.interface.desktop.interface import DesktopInterface

    return DesktopInterface
  if name == "DesktopConfig":
    from definable.agent.interface.desktop.config import DesktopConfig

    return DesktopConfig
  if name == "TelegramInterface":
    from definable.agent.interface.telegram.interface import TelegramInterface

    return TelegramInterface
  if name == "TelegramConfig":
    from definable.agent.interface.telegram.config import TelegramConfig

    return TelegramConfig
  if name == "DiscordInterface":
    from definable.agent.interface.discord.interface import DiscordInterface

    return DiscordInterface
  if name == "DiscordConfig":
    from definable.agent.interface.discord.config import DiscordConfig

    return DiscordConfig
  if name == "SignalInterface":
    from definable.agent.interface.signal.interface import SignalInterface

    return SignalInterface
  if name == "SignalConfig":
    from definable.agent.interface.signal.config import SignalConfig

    return SignalConfig
  if name in ("IdentityResolver", "SQLiteIdentityResolver", "PlatformIdentity"):
    from definable.agent.interface import identity as _identity_mod

    return getattr(_identity_mod, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
  "DesktopInterface",
  "DesktopConfig",
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
