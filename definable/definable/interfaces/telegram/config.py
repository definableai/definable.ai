"""Telegram-specific configuration."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from definable.interfaces.config import InterfaceConfig
from definable.interfaces.errors import InterfaceError


@dataclass(frozen=True)
class TelegramConfig(InterfaceConfig):
  """Configuration for the Telegram interface.

  Extends InterfaceConfig with Telegram-specific settings.

  Attributes:
    bot_token: Telegram Bot API token (required).
    mode: Operation mode — "polling" for development, "webhook" for production.
    webhook_url: Public URL for webhook mode (required when mode="webhook").
    webhook_path: URL path the webhook server listens on.
    webhook_port: Port for the webhook HTTP server.
    webhook_secret: Secret token for webhook verification.
    allowed_user_ids: Restrict access to these Telegram user IDs.
    allowed_chat_ids: Restrict access to these Telegram chat IDs.
    parse_mode: Telegram message parse mode.
    polling_interval: Seconds between polling requests.
    polling_timeout: Long-polling timeout in seconds.
    connect_timeout: HTTP connection timeout in seconds.
    request_timeout: HTTP request timeout in seconds.
  """

  platform: str = "telegram"
  bot_token: str = ""

  mode: Literal["polling", "webhook"] = "polling"

  # Webhook settings
  webhook_url: Optional[str] = None
  webhook_path: str = "/webhook/telegram"
  webhook_port: int = 8443
  webhook_secret: Optional[str] = None

  # Access control
  allowed_user_ids: Optional[List[int]] = field(default=None, hash=False)
  allowed_chat_ids: Optional[List[int]] = field(default=None, hash=False)

  # Message formatting
  parse_mode: Literal["HTML", "MarkdownV2", "Markdown", None] = "HTML"

  # Polling settings
  polling_interval: float = 0.5
  polling_timeout: int = 30

  # HTTP settings
  connect_timeout: float = 10.0
  request_timeout: float = 60.0

  max_message_length: int = 4096

  def __post_init__(self) -> None:
    if not self.bot_token:
      raise InterfaceError("bot_token is required for TelegramConfig", platform="telegram")
    if self.mode == "webhook" and not self.webhook_url:
      raise InterfaceError(
        "webhook_url is required when mode='webhook'",
        platform="telegram",
      )
