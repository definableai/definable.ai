"""
Definable Interfaces — Connect agents to messaging platforms.

This module provides the BaseInterface ABC and platform implementations
for exposing agents through messaging platforms like Telegram, Discord,
Slack, WhatsApp, and custom web UIs.

Quick Start (Telegram):
  from definable.agent import Agent
  from definable.model.openai import OpenAIChat
  from definable.agent.interface.telegram import TelegramInterface

  agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are a helpful assistant.",
  )
  interface = TelegramInterface(
    agent=agent,
    bot_token="YOUR_BOT_TOKEN",
  )
  async with interface:
    await interface.serve_forever()

With Hooks:
  from definable.agent.interface.hooks import LoggingHook, AllowlistHook

  interface.add_hook(LoggingHook())
  interface.add_hook(AllowlistHook(allowed_user_ids={"123456"}))
"""

from definable.agent.interface.base import BaseInterface
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
  from definable.agent.interface.call.config import CallConfig
  from definable.agent.interface.call.interface import CallInterface
  from definable.agent.interface.cli.config import CLIConfig
  from definable.agent.interface.cli.interface import CLIInterface
  from definable.agent.interface.desktop.config import DesktopConfig
  from definable.agent.interface.desktop.interface import DesktopInterface
  from definable.agent.interface.discord.config import DiscordConfig
  from definable.agent.interface.discord.interface import DiscordInterface
  from definable.agent.interface.gateway import (
    InterfaceErrorEvent,
    InterfaceGateway,
    InterfaceRestartedEvent,
    InterfaceStartedEvent,
    InterfaceStatus,
    InterfaceStoppedEvent,
  )
  from definable.agent.interface.identity import IdentityResolver, PlatformIdentity, SQLiteIdentityResolver
  from definable.agent.interface.slack.config import SlackConfig
  from definable.agent.interface.slack.interface import SlackInterface
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
  if name == "CLIInterface":
    from definable.agent.interface.cli.interface import CLIInterface

    return CLIInterface
  if name == "CLIConfig":
    from definable.agent.interface.cli.config import CLIConfig

    return CLIConfig
  if name in ("IdentityResolver", "SQLiteIdentityResolver", "PlatformIdentity"):
    from definable.agent.interface import identity as _identity_mod

    return getattr(_identity_mod, name)
  if name in (
    "InterfaceGateway",
    "InterfaceStatus",
    "InterfaceStartedEvent",
    "InterfaceStoppedEvent",
    "InterfaceRestartedEvent",
    "InterfaceErrorEvent",
  ):
    from definable.agent.interface import gateway as _gateway_mod

    return getattr(_gateway_mod, name)
  if name == "CallInterface":
    from definable.agent.interface.call.interface import CallInterface

    return CallInterface
  if name == "CallConfig":
    from definable.agent.interface.call.config import CallConfig

    return CallConfig
  if name == "SlackInterface":
    from definable.agent.interface.slack.interface import SlackInterface

    return SlackInterface
  if name == "SlackConfig":
    from definable.agent.interface.slack.config import SlackConfig

    return SlackConfig
  if name == "WebSocketInterface":
    from definable.agent.interface.websocket.interface import WebSocketInterface

    return WebSocketInterface
  if name == "WebSocketConfig":
    from definable.agent.interface.websocket.config import WebSocketConfig

    return WebSocketConfig
  if name == "WhatsAppInterface":
    from definable.agent.interface.whatsapp.interface import WhatsAppInterface

    return WhatsAppInterface
  if name == "WhatsAppConfig":
    from definable.agent.interface.whatsapp.config import WhatsAppConfig

    return WhatsAppConfig
  if name == "EmailInterface":
    from definable.agent.interface.email.interface import EmailInterface

    return EmailInterface
  if name == "EmailConfig":
    from definable.agent.interface.email.config import EmailConfig

    return EmailConfig
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "BaseInterface",
  "InterfaceConfig",
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
  "CallInterface",
  "CallConfig",
  "CLIInterface",
  "CLIConfig",
  "DesktopInterface",
  "DesktopConfig",
  "SlackInterface",
  "SlackConfig",
  "TelegramInterface",
  "TelegramConfig",
  "DiscordInterface",
  "DiscordConfig",
  # Identity resolution (lazy-loaded)
  "IdentityResolver",
  "SQLiteIdentityResolver",
  "PlatformIdentity",
  # Gateway (lazy-loaded)
  "InterfaceGateway",
  "InterfaceStatus",
  "InterfaceStartedEvent",
  "InterfaceStoppedEvent",
  "InterfaceRestartedEvent",
  "InterfaceErrorEvent",
  # WebSocket (lazy-loaded)
  "WebSocketInterface",  # noqa: F822
  "WebSocketConfig",  # noqa: F822
  # WhatsApp (lazy-loaded)
  "WhatsAppInterface",  # noqa: F822
  "WhatsAppConfig",  # noqa: F822
  # Email (lazy-loaded)
  "EmailInterface",  # noqa: F822
  "EmailConfig",  # noqa: F822
]
