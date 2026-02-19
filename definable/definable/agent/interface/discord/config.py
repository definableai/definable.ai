"""Discord-specific configuration."""

from dataclasses import dataclass, field
from typing import List, Optional

from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import InterfaceError


@dataclass(frozen=True)
class DiscordConfig(InterfaceConfig):
  """Configuration for the Discord interface.

  Extends InterfaceConfig with Discord-specific settings.

  Attributes:
    bot_token: Discord bot token from the Developer Portal (required).
    intents_message_content: Enable MESSAGE_CONTENT privileged intent.
    allowed_guild_ids: Restrict to specific guild (server) IDs.
    allowed_channel_ids: Restrict to specific channel IDs.
    respond_to_bots: Whether to respond to messages from other bots.
    command_prefix: If set, only respond to messages starting with this prefix.
    connect_timeout: Timeout for initial gateway connection in seconds.
  """

  platform: str = "discord"
  bot_token: str = ""

  # Intent settings
  intents_message_content: bool = True

  # Access control
  allowed_guild_ids: Optional[List[int]] = field(default=None, hash=False)
  allowed_channel_ids: Optional[List[int]] = field(default=None, hash=False)

  # Behavior
  respond_to_bots: bool = False
  command_prefix: Optional[str] = None

  # HTTP settings
  connect_timeout: float = 30.0

  max_message_length: int = 2000

  def __post_init__(self) -> None:
    if not self.bot_token:
      raise InterfaceError("bot_token is required for DiscordConfig", platform="discord")
