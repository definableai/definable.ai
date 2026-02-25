"""Slack-specific configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import InterfaceError


@dataclass(frozen=True)
class SlackConfig(InterfaceConfig):
  """Configuration for the Slack interface.

  Extends InterfaceConfig with Slack-specific settings.

  Attributes:
    bot_token: Slack Bot token (xoxb-..., required).
    app_token: Slack App-level token (xapp-..., required for socket mode).
    signing_secret: Signing secret for HTTP mode request verification.
    mode: Receiver mode — "socket" for development, "http" for production.
    events_path: URL path for Events API in HTTP mode.
    interactions_path: URL path for interactive components in HTTP mode.
    respond_to_mentions: Whether to respond to @bot mentions in channels.
    respond_to_dms: Whether to respond to direct messages.
    respond_to_thread_replies: Whether to respond when a user replies in a bot thread.
    thread_replies_in_channel: Always reply in a thread when in channels.
    thread_replies_in_dm: Whether to reply in threads in DMs.
    typing_reaction: Emoji name to react with while processing (empty to disable).
    done_reaction: Emoji name to react with on completion (empty to disable).
    convert_markdown: Whether to convert Markdown to Slack mrkdwn format.
    allowed_user_ids: Restrict access to these Slack user IDs.
    allowed_channel_ids: Restrict access to these Slack channel IDs.
    max_retries: Maximum retries for Slack API calls on rate limit.
    connect_timeout: HTTP connection timeout in seconds.
    request_timeout: HTTP request timeout in seconds.
    slash_commands: Slash commands to register. Maps command name (e.g. "/ask")
      to a description string. Commands are routed through the agent pipeline.
    route_commands_to_agent: Whether slash commands are routed to the agent.
      If False, commands are only dispatched to registered command callbacks.
  """

  platform: str = "slack"
  bot_token: str = ""
  app_token: str = ""
  signing_secret: str = ""

  mode: Literal["socket", "http"] = "socket"

  # HTTP Events API settings
  events_path: str = "/slack/events"
  interactions_path: str = "/slack/interactions"

  # Channel behavior
  respond_to_mentions: bool = True
  respond_to_dms: bool = True
  respond_to_thread_replies: bool = True
  thread_replies_in_channel: bool = True

  # DM threading
  thread_replies_in_dm: bool = False

  # Typing indicators
  typing_reaction: str = "hourglass_flowing_sand"
  done_reaction: str = ""

  # Text formatting
  convert_markdown: bool = True

  # Access control
  allowed_user_ids: Optional[List[str]] = field(default=None, hash=False)
  allowed_channel_ids: Optional[List[str]] = field(default=None, hash=False)

  # Rate limiting
  max_retries: int = 3

  # HTTP settings
  connect_timeout: float = 10.0
  request_timeout: float = 60.0

  max_message_length: int = 40000

  # Slash commands: {"/command_name": "description"}
  slash_commands: Optional[Dict[str, str]] = field(default=None, hash=False)

  # Whether slash commands are routed through the agent pipeline
  route_commands_to_agent: bool = True

  def __post_init__(self) -> None:
    if not self.bot_token:
      raise InterfaceError("bot_token is required for SlackConfig", platform="slack")
    if self.mode == "socket" and not self.app_token:
      raise InterfaceError(
        "app_token is required when mode='socket'",
        platform="slack",
      )
    if self.mode == "http" and not self.signing_secret:
      raise InterfaceError(
        "signing_secret is required when mode='http'",
        platform="slack",
      )
