"""Telegram-specific configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import InterfaceError


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
    auto_format: Convert Markdown to Telegram HTML automatically (Phase 1).
    streaming: Enable response streaming via message edits (Phase 4).
    stream_edit_interval: Minimum seconds between message edits during streaming (Phase 4).
    stream_min_chars: Minimum characters before first message send during streaming (Phase 4).
    stream_tool_indicator: Show tool usage during streaming (Phase 4).
    handle_callback_queries: Handle inline keyboard callback queries (Phase 5).
    group_mode: Bot behavior in group chats (Phase 7).
    enable_forum_topics: Support forum topic-based session isolation (Phase 8).
    handle_reactions: Handle message reaction updates (Phase 20).
    commands: Bot command menu to sync on startup (Phase 16).
    sync_commands_on_startup: Automatically sync commands on startup (Phase 16).
    dm_policy: Access policy for direct messages (Phase 17).
    group_policy: Access policy for group chats (Phase 17).
    dm_allowlist: User IDs allowed in DM when dm_policy="allowlist" (Phase 17).
    group_allowlist: Chat IDs allowed when group_policy="allowlist" (Phase 17).
    media_group_timeout: Seconds to wait for additional media group items (Phase 19).
    outbound_rate_limit: Max outbound API calls per second (Phase 14).
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
  auto_format: bool = True  # Phase 1: Markdown→HTML conversion

  # Polling settings
  polling_interval: float = 0.5
  polling_timeout: int = 30

  # HTTP settings
  connect_timeout: float = 10.0
  request_timeout: float = 60.0

  max_message_length: int = 4096

  # Phase 4: Streaming
  streaming: bool = True
  stream_edit_interval: float = 1.0
  stream_min_chars: int = 30
  stream_tool_indicator: bool = True

  # Phase 5: Callback queries
  handle_callback_queries: bool = True

  # Phase 7: Group chat
  group_mode: Literal["mention", "always", "disabled"] = "mention"

  # Phase 8: Forum topics
  enable_forum_topics: bool = True

  # Phase 14: Rate limiting
  outbound_rate_limit: float = 30.0  # calls per second

  # Phase 16: Command menu
  commands: Optional[Dict[str, str]] = field(default=None, hash=False)
  sync_commands_on_startup: bool = True

  # Phase 17: DM vs Group policies
  dm_policy: Literal["open", "allowlist", "disabled"] = "open"
  group_policy: Literal["open", "allowlist", "disabled"] = "open"
  dm_allowlist: Optional[List[int]] = field(default=None, hash=False)
  group_allowlist: Optional[List[int]] = field(default=None, hash=False)

  # Phase 19: Media groups
  media_group_timeout: float = 0.5

  # Phase 20: Reactions
  handle_reactions: bool = False

  def __post_init__(self) -> None:
    if not self.bot_token:
      raise InterfaceError("bot_token is required for TelegramConfig", platform="telegram")
    if self.mode == "webhook" and not self.webhook_url:
      raise InterfaceError(
        "webhook_url is required when mode='webhook'",
        platform="telegram",
      )
