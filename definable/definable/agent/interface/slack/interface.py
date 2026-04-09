"""Slack interface implementation using slack-bolt and slack-sdk."""

import contextlib
import io
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.errors import (
  InterfaceAuthenticationError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.agent.interface.slack.config import SlackConfig
from definable.agent.interface.slack.formatter import markdown_to_mrkdwn, split_text
from definable.media import Audio, File, Image, Video
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from slack_bolt.async_app import AsyncApp
  from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
  from slack_sdk.web.async_client import AsyncWebClient

  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver


def _ensure_slack_deps() -> None:
  """Verify that slack-bolt is installed."""
  try:
    import slack_bolt  # noqa: F401
  except ImportError:
    raise ImportError("Slack dependencies not found. Install them with: pip install 'definable[slack]'") from None


class SlackInterface(BaseInterface):
  """Interface connecting an agent to Slack via the Bolt framework.

  Supports both Socket Mode (for development — no public URL needed) and
  HTTP Events API (for production — mounts on AgentServer's FastAPI app).

  Example (Socket Mode)::

      interface = SlackInterface(
        agent=agent,
        bot_token="xoxb-...",
        app_token="xapp-...",
      )
      async with interface:
        await interface.serve_forever()

  Example (HTTP Events API)::

      interface = SlackInterface(
        agent=agent,
        bot_token="xoxb-...",
        signing_secret="...",
        mode="http",
      )
      runtime = AgentRuntime(agent, interfaces=[interface], port=3000)
      await runtime.start()
  """

  def __init__(
    self,
    *,
    # Slack-specific
    bot_token: str = "",
    app_token: str = "",
    signing_secret: str = "",
    mode: Literal["socket", "http"] = "socket",
    events_path: str = "/slack/events",
    interactions_path: str = "/slack/interactions",
    respond_to_mentions: bool = True,
    respond_to_dms: bool = True,
    respond_to_thread_replies: bool = True,
    thread_replies_in_channel: bool = True,
    thread_replies_in_dm: bool = False,
    typing_reaction: str = "hourglass_flowing_sand",
    done_reaction: str = "",
    convert_markdown: bool = True,
    allowed_user_ids: Optional[List[str]] = None,
    allowed_channel_ids: Optional[List[str]] = None,
    max_retries: int = 3,
    connect_timeout: float = 10.0,
    request_timeout: float = 60.0,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 40000,
    rate_limit_messages_per_minute: int = 30,
    # Slash commands
    slash_commands: Optional[Dict[str, str]] = None,
    route_commands_to_agent: bool = True,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
  ) -> None:
    resolved_config = SlackConfig(
      bot_token=bot_token,
      app_token=app_token,
      signing_secret=signing_secret,
      mode=mode,
      events_path=events_path,
      interactions_path=interactions_path,
      respond_to_mentions=respond_to_mentions,
      respond_to_dms=respond_to_dms,
      respond_to_thread_replies=respond_to_thread_replies,
      thread_replies_in_channel=thread_replies_in_channel,
      thread_replies_in_dm=thread_replies_in_dm,
      typing_reaction=typing_reaction,
      done_reaction=done_reaction,
      convert_markdown=convert_markdown,
      allowed_user_ids=allowed_user_ids,
      allowed_channel_ids=allowed_channel_ids,
      max_retries=max_retries,
      connect_timeout=connect_timeout,
      request_timeout=request_timeout,
      slash_commands=slash_commands,
      route_commands_to_agent=route_commands_to_agent,
      max_session_history=max_session_history,
      session_ttl_seconds=session_ttl_seconds,
      max_concurrent_requests=max_concurrent_requests,
      error_message=error_message,
      typing_indicator=typing_indicator,
      max_message_length=max_message_length,
      rate_limit_messages_per_minute=rate_limit_messages_per_minute,
    )
    super().__init__(
      agent=agent,
      config=resolved_config,
      session_manager=session_manager,
      hooks=hooks,
      identity_resolver=identity_resolver,
      auth=auth,
    )
    self._slack_config: SlackConfig = self.config  # type: ignore[assignment]
    self._bolt_app: Optional["AsyncApp"] = None
    self._socket_handler: Optional["AsyncSocketModeHandler"] = None
    self._client: Optional["AsyncWebClient"] = None
    self._bot_user_id: Optional[str] = None
    self._bot_thread_parents: Set[str] = set()  # Track threads the bot has participated in
    self._command_callbacks: Dict[str, Any] = {}  # {"/name": async_callback}
    self._action_callbacks: Dict[str, Any] = {}  # {"action_id": async_callback}
    self._view_callbacks: Dict[str, Any] = {}  # {"callback_id": async_callback}
    self._shortcut_callbacks: Dict[str, Any] = {}  # {"callback_id": async_callback}
    self._reaction_added_callbacks: Dict[str, Any] = {}  # {"emoji" or "*": async_callback}
    self._reaction_removed_callbacks: Dict[str, Any] = {}  # {"emoji" or "*": async_callback}
    self._home_opened_callback: Optional[Any] = None
    self._event_callbacks: Dict[str, Any] = {}  # {"event_type": async_callback}

  # --- Lifecycle ---

  async def _start_receiver(self) -> None:
    _ensure_slack_deps()
    from slack_bolt.async_app import AsyncApp

    if self._slack_config.mode == "socket":
      self._bolt_app = AsyncApp(token=self._slack_config.bot_token)
    else:
      self._bolt_app = AsyncApp(
        token=self._slack_config.bot_token,
        signing_secret=self._slack_config.signing_secret,
      )

    self._client = self._bolt_app.client

    # Identify the bot user so we can filter self-messages
    await self._identify_bot()

    # Register event listeners
    self._register_listeners()

    if self._slack_config.mode == "socket":
      from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

      self._socket_handler = AsyncSocketModeHandler(self._bolt_app, self._slack_config.app_token)
      await self._socket_handler.connect_async()
      log_info("[slack] Socket Mode connected")
    else:
      log_info(f"[slack] HTTP mode ready (events: {self._slack_config.events_path})")

  async def _stop_receiver(self) -> None:
    if self._socket_handler is not None:
      await self._socket_handler.close_async()
      self._socket_handler = None

    self._bolt_app = None
    self._client = None

  # --- Bot identification ---

  async def _identify_bot(self) -> None:
    """Call auth.test to get the bot's own user ID."""
    assert self._client is not None
    try:
      result = await self._client.auth_test()
      self._bot_user_id = result.get("user_id")
      bot_name = result.get("user")
      log_info(f"[slack] Connected as @{bot_name} (user_id={self._bot_user_id})")
    except Exception as e:
      raise InterfaceAuthenticationError(
        f"Failed to authenticate with Slack: {e}",
        platform="slack",
      ) from e

  # --- Event listeners ---

  def _register_listeners(self) -> None:
    """Register Bolt event listeners for messages, mentions, commands, and interactions."""
    assert self._bolt_app is not None

    @self._bolt_app.event("message")
    async def handle_message(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      await self._on_message_event(event)

    @self._bolt_app.event("app_mention")
    async def handle_mention(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      await self._on_mention_event(event)

    # Register configured slash commands
    if self._slack_config.slash_commands:
      for cmd_name in self._slack_config.slash_commands:
        self._register_command_listener(cmd_name)

    # Catch-all handler for Block Kit interactive components
    @self._bolt_app.action(re.compile(".*"))
    async def handle_action(ack: Any, body: Dict[str, Any], action: Dict[str, Any], respond: Any) -> None:  # noqa: ARG001
      await ack()
      await self._on_action(body, action)

    # Catch-all handler for modal view submissions
    @self._bolt_app.view(re.compile(".*"))
    async def handle_view_submission(ack: Any, body: Dict[str, Any], view: Dict[str, Any]) -> None:
      await ack()
      await self._on_view_submission(body, view)

    # Catch-all handler for shortcuts (message & global)
    @self._bolt_app.shortcut(re.compile(".*"))
    async def handle_shortcut(ack: Any, shortcut: Dict[str, Any], body: Dict[str, Any]) -> None:
      await ack()
      await self._on_shortcut(shortcut, body)

    # Reaction events
    @self._bolt_app.event("reaction_added")
    async def handle_reaction_added(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      await self._on_reaction_added(event)

    @self._bolt_app.event("reaction_removed")
    async def handle_reaction_removed(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      await self._on_reaction_removed(event)

    # App Home tab
    @self._bolt_app.event("app_home_opened")
    async def handle_home_opened(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      await self._on_home_opened(event)

    # Register generic event callbacks
    for evt_type in self._event_callbacks:
      self._register_event_listener(evt_type)

  def _register_command_listener(self, cmd_name: str) -> None:
    """Register a Bolt listener for a single slash command."""
    assert self._bolt_app is not None

    @self._bolt_app.command(cmd_name)
    async def handle_command(ack: Any, command: Dict[str, Any], respond: Any) -> None:  # noqa: ARG001
      await ack()
      await self._on_command(command)

  def _register_event_listener(self, event_type: str) -> None:
    """Register a Bolt listener for a generic event type."""
    assert self._bolt_app is not None

    @self._bolt_app.event(event_type)
    async def handle_generic_event(event: Dict[str, Any], say: Any) -> None:  # noqa: ARG001
      callback = self._event_callbacks.get(event_type)
      if callback:
        try:
          await callback(event)
        except Exception as e:
          log_error(f"[slack] Event callback error for {event_type}: {e}")

  async def _on_message_event(self, event: Dict[str, Any]) -> None:
    """Handle an incoming message event."""
    # Skip bot messages (including our own)
    if event.get("bot_id") or event.get("subtype") == "bot_message":
      return
    if event.get("user") == self._bot_user_id:
      return

    # Skip message subtypes we don't handle
    subtype = event.get("subtype")
    if subtype and subtype not in ("file_share", "thread_broadcast"):
      return

    channel_type = event.get("channel_type", "")

    # DMs
    if channel_type == "im":
      if not self._slack_config.respond_to_dms:
        return
      await self.handle_platform_message(event)
      return

    # Thread replies in channels
    thread_ts = event.get("thread_ts")
    if thread_ts and self._slack_config.respond_to_thread_replies:
      # Only respond if the bot is part of this thread
      if thread_ts in self._bot_thread_parents:
        await self.handle_platform_message(event)
        return

    # Channel messages without mention — ignore (mentions handled by app_mention)

  async def _on_mention_event(self, event: Dict[str, Any]) -> None:
    """Handle an @mention event."""
    if not self._slack_config.respond_to_mentions:
      return
    # Skip our own messages
    if event.get("user") == self._bot_user_id:
      return

    # Track this thread so we respond to follow-up replies
    thread_ts = event.get("thread_ts") or event.get("ts") or ""
    if thread_ts:
      self._bot_thread_parents.add(thread_ts)

    await self.handle_platform_message(event)

  async def _on_command(self, command: Dict[str, Any]) -> None:
    """Handle a slash command invocation."""
    cmd_name = command.get("command", "")

    # Custom callback takes priority
    if cmd_name in self._command_callbacks:
      try:
        await self._command_callbacks[cmd_name](command)
      except Exception as e:
        log_error(f"[slack] Command callback error for {cmd_name}: {e}")
      return

    # Route through agent pipeline
    if self._slack_config.route_commands_to_agent:
      synthetic = {
        "user": command.get("user_id", ""),
        "channel": command.get("channel_id", ""),
        "ts": command.get("trigger_id", ""),
        "text": command.get("text", ""),
        "channel_type": "channel",
      }
      await self.handle_platform_message(synthetic)
    else:
      log_warning(f"[slack] Unhandled command {cmd_name} — no callback and route_commands_to_agent=False")

  async def _on_action(self, body: Dict[str, Any], action: Dict[str, Any]) -> None:
    """Handle a Block Kit interactive component action."""
    action_id = action.get("action_id", "")
    callback = self._action_callbacks.get(action_id)
    if callback:
      try:
        await callback(action, body)
      except Exception as e:
        log_error(f"[slack] Action callback error for {action_id}: {e}")
    else:
      log_debug(f"[slack] Unhandled action: {action_id}")

  async def _on_view_submission(self, body: Dict[str, Any], view: Dict[str, Any]) -> None:
    """Handle a modal view submission."""
    callback_id = view.get("callback_id", "")
    callback = self._view_callbacks.get(callback_id)
    if callback:
      try:
        await callback(view, body)
      except Exception as e:
        log_error(f"[slack] View callback error for {callback_id}: {e}")
    else:
      log_debug(f"[slack] Unhandled view submission: {callback_id}")

  async def _on_shortcut(self, shortcut: Dict[str, Any], body: Dict[str, Any]) -> None:
    """Handle a message or global shortcut."""
    callback_id = shortcut.get("callback_id", "")
    callback = self._shortcut_callbacks.get(callback_id)
    if callback:
      try:
        await callback(shortcut, body)
      except Exception as e:
        log_error(f"[slack] Shortcut callback error for {callback_id}: {e}")
    else:
      log_debug(f"[slack] Unhandled shortcut: {callback_id}")

  async def _on_reaction_added(self, event: Dict[str, Any]) -> None:
    """Handle a reaction_added event."""
    # Skip our own reactions (typing/done indicators)
    if event.get("user") == self._bot_user_id:
      return
    reaction = event.get("reaction", "")
    callback = self._reaction_added_callbacks.get(reaction) or self._reaction_added_callbacks.get("*")
    if callback:
      try:
        await callback(event)
      except Exception as e:
        log_error(f"[slack] Reaction added callback error for {reaction}: {e}")

  async def _on_reaction_removed(self, event: Dict[str, Any]) -> None:
    """Handle a reaction_removed event."""
    if event.get("user") == self._bot_user_id:
      return
    reaction = event.get("reaction", "")
    callback = self._reaction_removed_callbacks.get(reaction) or self._reaction_removed_callbacks.get("*")
    if callback:
      try:
        await callback(event)
      except Exception as e:
        log_error(f"[slack] Reaction removed callback error for {reaction}: {e}")

  async def _on_home_opened(self, event: Dict[str, Any]) -> None:
    """Handle the app_home_opened event."""
    if event.get("tab") != "home":
      return
    if self._home_opened_callback:
      try:
        await self._home_opened_callback(event)
      except Exception as e:
        log_error(f"[slack] Home opened callback error: {e}")

  # --- Callback registration ---

  def on_command(self, name: str, callback: Any) -> "SlackInterface":
    """Register a callback for a slash command.

    The command must also be listed in ``slash_commands`` config for Bolt
    to register the listener. The callback receives the command dict::

        async def handle_ask(command: dict) -> None:
            print(command["text"])

        interface.on_command("/ask", handle_ask)

    Args:
      name: Command name including the leading slash (e.g. "/ask").
      callback: Async callable receiving the command dict.

    Returns:
      Self for chaining.
    """
    if not name.startswith("/"):
      name = f"/{name}"
    self._command_callbacks[name] = callback
    return self

  def on_action(self, action_id: str, callback: Any) -> "SlackInterface":
    """Register a callback for a Block Kit action (button click, menu selection, etc.).

    The callback receives the action dict and full body dict::

        async def handle_click(action: dict, body: dict) -> None:
            print(f"Button {action['action_id']} clicked by {body['user']['id']}")

        interface.on_action("approve_button", handle_click)

    Args:
      action_id: The ``action_id`` of the Block Kit element.
      callback: Async callable ``(action, body) -> None``.

    Returns:
      Self for chaining.
    """
    self._action_callbacks[action_id] = callback
    return self

  def on_view(self, callback_id: str, callback: Any) -> "SlackInterface":
    """Register a callback for a modal view submission.

    The callback receives the view dict and full body dict::

        async def handle_form(view: dict, body: dict) -> None:
            values = view["state"]["values"]
            print(values)

        interface.on_view("feedback_form", handle_form)

    Args:
      callback_id: The ``callback_id`` of the modal view.
      callback: Async callable ``(view, body) -> None``.

    Returns:
      Self for chaining.
    """
    self._view_callbacks[callback_id] = callback
    return self

  def on_shortcut(self, callback_id: str, callback: Any) -> "SlackInterface":
    """Register a callback for a message or global shortcut.

    The callback receives the shortcut dict and full body dict::

        async def handle_summarize(shortcut: dict, body: dict) -> None:
            message = shortcut.get("message", {})
            trigger_id = shortcut["trigger_id"]
            # Open a modal or process the message

        interface.on_shortcut("summarize_message", handle_summarize)

    Args:
      callback_id: The ``callback_id`` configured in the Slack app.
      callback: Async callable ``(shortcut, body) -> None``.

    Returns:
      Self for chaining.
    """
    self._shortcut_callbacks[callback_id] = callback
    return self

  def on_reaction_added(self, reaction: str, callback: Any) -> "SlackInterface":
    """Register a callback for when a reaction emoji is added to a message.

    The bot's own reactions (typing/done indicators) are automatically filtered.

    Use ``"*"`` as the reaction name to catch all reactions::

        async def on_thumbsup(event: dict) -> None:
            channel = event["item"]["channel"]
            ts = event["item"]["ts"]
            print(f"Thumbsup on message {ts} in {channel}")

        interface.on_reaction_added("thumbsup", on_thumbsup)
        interface.on_reaction_added("*", on_any_reaction)  # catch-all

    Args:
      reaction: Emoji name (e.g. ``"thumbsup"``) or ``"*"`` for all.
      callback: Async callable ``(event) -> None``.

    Returns:
      Self for chaining.
    """
    self._reaction_added_callbacks[reaction] = callback
    return self

  def on_reaction_removed(self, reaction: str, callback: Any) -> "SlackInterface":
    """Register a callback for when a reaction emoji is removed from a message.

    Args:
      reaction: Emoji name or ``"*"`` for all.
      callback: Async callable ``(event) -> None``.

    Returns:
      Self for chaining.
    """
    self._reaction_removed_callbacks[reaction] = callback
    return self

  def on_home_opened(self, callback: Any) -> "SlackInterface":
    """Register a callback for the App Home tab being opened.

    Called when a user opens the bot's Home tab. Use this to dynamically
    build the home view with ``publish_home()``::

        async def build_home(event: dict) -> None:
            user_id = event["user"]
            view = home_tab_view([section_block(f"Welcome, <@{user_id}>!")])
            await interface.publish_home(user_id, view)

        interface.on_home_opened(build_home)

    Args:
      callback: Async callable ``(event) -> None``.

    Returns:
      Self for chaining.
    """
    self._home_opened_callback = callback
    return self

  def on_event(self, event_type: str, callback: Any) -> "SlackInterface":
    """Register a callback for any Slack event type.

    This is an escape hatch for events not covered by the dedicated handlers.
    Useful for ``member_joined_channel``, ``channel_created``, ``team_join``, etc.

    The callback receives the event dict::

        async def on_join(event: dict) -> None:
            print(f"User {event['user']} joined {event['channel']}")

        interface.on_event("member_joined_channel", on_join)

    Args:
      event_type: Slack event type string.
      callback: Async callable ``(event) -> None``.

    Returns:
      Self for chaining.

    Raises:
      ValueError: If the event type is handled by a dedicated method.
    """
    reserved = {"message", "app_mention", "reaction_added", "reaction_removed", "app_home_opened"}
    if event_type in reserved:
      raise ValueError(
        f"Event '{event_type}' is handled internally. "
        f"Use the dedicated registration method instead "
        f"(e.g., on_reaction_added for reaction_added events)."
      )
    self._event_callbacks[event_type] = callback
    if self._bolt_app is not None:
      self._register_event_listener(event_type)
    return self

  # --- Inbound conversion ---

  async def _convert_inbound(self, raw_message: Dict[str, Any]) -> Optional[InterfaceMessage]:
    """Convert a Slack event dict to InterfaceMessage."""
    user_id = raw_message.get("user", "")
    channel = raw_message.get("channel", "")
    ts = raw_message.get("ts", "")
    thread_ts = raw_message.get("thread_ts")
    text = raw_message.get("text", "")
    channel_type = raw_message.get("channel_type", "")

    # Access control
    if self._slack_config.allowed_user_ids is not None:
      if user_id not in self._slack_config.allowed_user_ids:
        log_debug(f"[slack] Ignoring message from unauthorized user {user_id}")
        return None

    if self._slack_config.allowed_channel_ids is not None:
      if channel not in self._slack_config.allowed_channel_ids:
        log_debug(f"[slack] Ignoring message from unauthorized channel {channel}")
        return None

    # Strip @mention of our bot from the text
    if self._bot_user_id:
      text = text.replace(f"<@{self._bot_user_id}>", "").strip()

    # Resolve username
    username = await self._get_username(user_id)

    # Determine chat_id for session mapping:
    # Thread-based: use thread_ts as conversation identifier
    # Non-threaded DM: use channel as conversation identifier
    if thread_ts:
      chat_id = thread_ts
    elif channel_type == "im" and not self._slack_config.thread_replies_in_dm:
      chat_id = channel
    else:
      # For new channel messages (mentions), the reply will start a thread
      # Use the message ts as the thread root; fall back to channel for commands (no ts)
      chat_id = ts or channel

    # Extract media from file attachments
    images: Optional[List[Image]] = None
    audio_list: Optional[List[Audio]] = None
    video_list: Optional[List[Video]] = None
    files_list: Optional[List[File]] = None

    slack_files = raw_message.get("files", [])
    if slack_files:
      images, audio_list, video_list, files_list = await self._extract_media(slack_files)

    return InterfaceMessage(
      text=text or None,
      platform="slack",
      platform_user_id=user_id,
      platform_chat_id=chat_id,
      platform_message_id=ts,
      username=username,
      images=images,
      audio=audio_list,
      videos=video_list,
      files=files_list,
      reply_to_message_id=thread_ts,
      metadata={
        "channel": channel,
        "thread_ts": thread_ts,
        "channel_type": channel_type,
        "ts": ts,
      },
    )

  # --- Response sending ---

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send response back to Slack."""
    assert self._client is not None
    channel = original_msg.metadata.get("channel", "")
    event_ts = original_msg.metadata.get("ts", "")
    channel_type = original_msg.metadata.get("channel_type", "")

    # Determine threading
    existing_thread = original_msg.metadata.get("thread_ts")
    thread_ts = self._resolve_thread_ts(original_msg, channel_type)
    is_new_thread = not existing_thread and thread_ts is not None

    # Remove typing reaction
    if self.config.typing_indicator and self._slack_config.typing_reaction:
      await self._remove_reaction(channel, event_ts, self._slack_config.typing_reaction)

    # Send text content
    if response.content:
      text = response.content
      if self._slack_config.convert_markdown:
        text = markdown_to_mrkdwn(text)

      chunks = split_text(text, self._slack_config.max_message_length)
      for chunk in chunks:
        result = await self._post_message(channel, chunk, thread_ts=thread_ts)
        # Track new threads so we respond to follow-up replies
        if result and is_new_thread:
          new_ts = result.get("ts") or thread_ts
          if new_ts:
            self._bot_thread_parents.add(new_ts)
          is_new_thread = False  # Only track once

    # Send images
    if response.images:
      for image in response.images:
        await self._send_image(channel, image, thread_ts=thread_ts)

    # Send audio
    if response.audio:
      for audio in response.audio:
        await self._send_file_content(channel, audio, thread_ts=thread_ts)

    # Send videos
    if response.videos:
      for video in response.videos:
        await self._send_file_content(channel, video, thread_ts=thread_ts)

    # Send files
    if response.files:
      for file in response.files:
        await self._send_file_media(channel, file, thread_ts=thread_ts)

    # Add done reaction
    if self._slack_config.done_reaction:
      await self._add_reaction(channel, event_ts, self._slack_config.done_reaction)

  def _resolve_thread_ts(self, message: InterfaceMessage, channel_type: str) -> Optional[str]:
    """Determine the thread_ts for a response."""
    existing_thread = message.metadata.get("thread_ts")
    if existing_thread:
      return existing_thread

    # In channels, always thread (unless disabled)
    if channel_type != "im" and self._slack_config.thread_replies_in_channel:
      return message.metadata.get("ts")

    # In DMs, thread only if enabled
    if channel_type == "im" and self._slack_config.thread_replies_in_dm:
      return message.metadata.get("ts")

    return None

  # --- Typing indicator ---

  async def _add_typing_indicator(self, channel: str, ts: str) -> None:
    """Add a reaction to indicate the bot is processing."""
    if self.config.typing_indicator and self._slack_config.typing_reaction:
      await self._add_reaction(channel, ts, self._slack_config.typing_reaction)

  # Override handle_platform_message to inject typing indicator
  async def handle_platform_message(self, raw_message: Any) -> None:
    """Process a platform message with typing indicator support."""
    # Add typing reaction before the pipeline runs
    channel = raw_message.get("channel", "")
    ts = raw_message.get("ts", "")
    await self._add_typing_indicator(channel, ts)

    # Delegate to the base pipeline
    await super().handle_platform_message(raw_message)

  # --- Slack API helpers ---

  async def _post_message(
    self,
    channel: str,
    text: str,
    *,
    thread_ts: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Post a message to a Slack channel."""
    assert self._client is not None
    try:
      kwargs: Dict[str, Any] = {"channel": channel, "text": text}
      if thread_ts:
        kwargs["thread_ts"] = thread_ts
      if blocks:
        kwargs["blocks"] = blocks
      result = await self._client.chat_postMessage(**kwargs)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_postMessage")
      return None

  async def update_message(
    self,
    channel: str,
    ts: str,
    *,
    text: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Edit an existing message via ``chat.update``.

    Args:
      channel: Channel containing the message.
      ts: Timestamp of the message to update.
      text: New text content (also used as fallback for blocks).
      blocks: New Block Kit blocks (replaces existing blocks).

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      kwargs: Dict[str, Any] = {"channel": channel, "ts": ts}
      if text is not None:
        kwargs["text"] = text
      if blocks is not None:
        kwargs["blocks"] = blocks
      result = await self._client.chat_update(**kwargs)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_update")
      return None

  async def send_ephemeral(
    self,
    channel: str,
    user: str,
    text: str,
    *,
    blocks: Optional[List[Dict[str, Any]]] = None,
    thread_ts: Optional[str] = None,
  ) -> Optional[Dict[str, Any]]:
    """Send an ephemeral message visible only to a specific user.

    Args:
      channel: Channel to send to.
      user: User ID who will see the message.
      text: Message text (also used as fallback for blocks).
      blocks: Optional Block Kit blocks.
      thread_ts: Optional thread to send in.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      kwargs: Dict[str, Any] = {"channel": channel, "user": user, "text": text}
      if blocks:
        kwargs["blocks"] = blocks
      if thread_ts:
        kwargs["thread_ts"] = thread_ts
      result = await self._client.chat_postEphemeral(**kwargs)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_postEphemeral")
      return None

  async def send_blocks(
    self,
    channel: str,
    blocks: List[Dict[str, Any]],
    *,
    text: str = "",
    thread_ts: Optional[str] = None,
  ) -> Optional[Dict[str, Any]]:
    """Send a Block Kit message to a channel.

    Args:
      channel: Channel to send to.
      blocks: Block Kit block list.
      text: Fallback text for notifications (recommended by Slack).
      thread_ts: Optional thread to send in.

    Returns:
      The API response dict, or None on failure.
    """
    return await self._post_message(channel, text, thread_ts=thread_ts, blocks=blocks)

  async def open_modal(
    self,
    trigger_id: str,
    view: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    """Open a modal dialog using ``views.open``.

    Args:
      trigger_id: Trigger ID from a slash command or interaction.
      view: Modal view definition (use ``modal_view()`` from formatter).

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.views_open(trigger_id=trigger_id, view=view)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "views_open")
      return None

  async def update_modal(
    self,
    view_id: str,
    view: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    """Update an existing modal view using ``views.update``.

    Args:
      view_id: ID of the existing view to update.
      view: Updated view definition.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.views_update(view_id=view_id, view=view)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "views_update")
      return None

  async def push_modal(
    self,
    trigger_id: str,
    view: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    """Push a new view onto the modal stack using ``views.push``.

    Args:
      trigger_id: Trigger ID from an interaction within a modal.
      view: View definition to push.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.views_push(trigger_id=trigger_id, view=view)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "views_push")
      return None

  async def publish_home(
    self,
    user_id: str,
    view: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    """Publish or update the App Home tab for a user via ``views.publish``.

    Args:
      user_id: The user whose Home tab to update.
      view: Home tab view definition (use ``home_tab_view()`` from formatter).

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.views_publish(user_id=user_id, view=view)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "views_publish")
      return None

  async def schedule_message(
    self,
    channel: str,
    text: str,
    post_at: int,
    *,
    thread_ts: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Schedule a message for future delivery via ``chat.scheduleMessage``.

    Args:
      channel: Channel to post to.
      text: Message text (also used as fallback for blocks).
      post_at: Unix timestamp (seconds) for when to send.
      thread_ts: Optional thread to post in.
      blocks: Optional Block Kit blocks.

    Returns:
      The API response dict (contains ``scheduled_message_id``), or None on failure.
    """
    assert self._client is not None
    try:
      kwargs: Dict[str, Any] = {"channel": channel, "text": text, "post_at": post_at}
      if thread_ts:
        kwargs["thread_ts"] = thread_ts
      if blocks:
        kwargs["blocks"] = blocks
      result = await self._client.chat_scheduleMessage(**kwargs)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_scheduleMessage")
      return None

  async def delete_scheduled_message(
    self,
    channel: str,
    scheduled_message_id: str,
  ) -> Optional[Dict[str, Any]]:
    """Cancel a scheduled message via ``chat.deleteScheduledMessage``.

    Args:
      channel: Channel the message was scheduled for.
      scheduled_message_id: ID from ``schedule_message()`` response.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.chat_deleteScheduledMessage(
        channel=channel,
        scheduled_message_id=scheduled_message_id,
      )
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_deleteScheduledMessage")
      return None

  async def delete_message(
    self,
    channel: str,
    ts: str,
  ) -> Optional[Dict[str, Any]]:
    """Delete a bot message via ``chat.delete``.

    Note: Bots can only delete messages they posted.

    Args:
      channel: Channel containing the message.
      ts: Timestamp of the message to delete.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.chat_delete(channel=channel, ts=ts)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "chat_delete")
      return None

  async def get_permalink(
    self,
    channel: str,
    message_ts: str,
  ) -> Optional[str]:
    """Get a permanent URL for a message via ``chat.getPermalink``.

    Args:
      channel: Channel containing the message.
      message_ts: Timestamp of the message.

    Returns:
      The permalink URL string, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.chat_getPermalink(channel=channel, message_ts=message_ts)
      data = result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
      return data.get("permalink")  # type: ignore[union-attr]
    except Exception as e:
      self._handle_api_error(e, "chat_getPermalink")
      return None

  async def set_topic(
    self,
    channel: str,
    topic: str,
  ) -> Optional[Dict[str, Any]]:
    """Set the topic of a channel via ``conversations.setTopic``.

    Args:
      channel: Channel to update.
      topic: New topic text.

    Returns:
      The API response dict, or None on failure.
    """
    assert self._client is not None
    try:
      result = await self._client.conversations_setTopic(channel=channel, topic=topic)
      return result.data if hasattr(result, "data") else dict(result)  # type: ignore[arg-type]
    except Exception as e:
      self._handle_api_error(e, "conversations_setTopic")
      return None

  async def pin_message(self, channel: str, timestamp: str) -> None:
    """Pin a message in a channel via ``pins.add``.

    Args:
      channel: Channel containing the message.
      timestamp: Timestamp of the message to pin.
    """
    assert self._client is not None
    with contextlib.suppress(Exception):
      await self._client.pins_add(channel=channel, timestamp=timestamp)

  async def unpin_message(self, channel: str, timestamp: str) -> None:
    """Unpin a message in a channel via ``pins.remove``.

    Args:
      channel: Channel containing the message.
      timestamp: Timestamp of the message to unpin.
    """
    assert self._client is not None
    with contextlib.suppress(Exception):
      await self._client.pins_remove(channel=channel, timestamp=timestamp)

  async def _send_image(
    self,
    channel: str,
    image: Image,
    *,
    thread_ts: Optional[str] = None,
  ) -> None:
    """Send an image to a Slack channel."""
    if image.url:
      # Use an image block for URL-based images
      blocks = [
        {
          "type": "image",
          "image_url": image.url,
          "alt_text": image.alt_text or "Image",
        }
      ]
      await self._post_message(channel, image.alt_text or "Image", thread_ts=thread_ts, blocks=blocks)
    elif image.content or image.filepath:
      content = image.content
      if image.filepath and not content:
        with open(str(image.filepath), "rb") as f:
          content = f.read()
      if content:
        ext = image.format or "png"
        await self._upload_file(
          channel,
          content,
          filename=f"image.{ext}",
          thread_ts=thread_ts,
        )

  async def _send_file_content(
    self,
    channel: str,
    media: Any,
    *,
    thread_ts: Optional[str] = None,
  ) -> None:
    """Send an Audio or Video as a file upload."""
    content = None
    filename = "file"
    if hasattr(media, "content") and media.content:
      content = media.content
    elif hasattr(media, "filepath") and media.filepath:
      with open(str(media.filepath), "rb") as f:
        content = f.read()
    elif hasattr(media, "url") and media.url:
      # Download the content first
      content = await self._download_url(media.url)

    if content:
      ext = getattr(media, "format", None) or "bin"
      if hasattr(media, "mime_type") and media.mime_type:
        # Derive extension from MIME type
        parts = media.mime_type.split("/")
        if len(parts) == 2:
          ext = parts[1].split(";")[0]
      filename = f"file.{ext}"
      await self._upload_file(channel, content, filename=filename, thread_ts=thread_ts)

  async def _send_file_media(
    self,
    channel: str,
    file: File,
    *,
    thread_ts: Optional[str] = None,
  ) -> None:
    """Send a File media object."""
    content = None
    filename = file.filename or file.name or "file"

    if file.content:
      content = file.content if isinstance(file.content, bytes) else str(file.content).encode("utf-8")
    elif file.filepath:
      with open(str(file.filepath), "rb") as f:
        content = f.read()
    elif file.url:
      content = await self._download_url(file.url)

    if content:
      await self._upload_file(channel, content, filename=filename, thread_ts=thread_ts)

  async def _upload_file(
    self,
    channel: str,
    content: bytes,
    *,
    filename: str = "file",
    title: Optional[str] = None,
    thread_ts: Optional[str] = None,
    initial_comment: Optional[str] = None,
  ) -> None:
    """Upload a file to Slack using files_upload_v2."""
    assert self._client is not None
    try:
      kwargs: Dict[str, Any] = {
        "channel": channel,
        "file": io.BytesIO(content),
        "filename": filename,
      }
      if title:
        kwargs["title"] = title
      if thread_ts:
        kwargs["thread_ts"] = thread_ts
      if initial_comment:
        kwargs["initial_comment"] = initial_comment
      await self._client.files_upload_v2(**kwargs)
    except Exception as e:
      log_error(f"[slack] File upload failed: {e}")

  async def _add_reaction(self, channel: str, timestamp: str, name: str) -> None:
    """Add an emoji reaction to a message."""
    assert self._client is not None
    with contextlib.suppress(Exception):
      await self._client.reactions_add(channel=channel, timestamp=timestamp, name=name)

  async def _remove_reaction(self, channel: str, timestamp: str, name: str) -> None:
    """Remove an emoji reaction from a message."""
    assert self._client is not None
    with contextlib.suppress(Exception):
      await self._client.reactions_remove(channel=channel, timestamp=timestamp, name=name)

  async def _get_username(self, user_id: str) -> Optional[str]:
    """Resolve a Slack user ID to a display name."""
    if not user_id:
      return None
    assert self._client is not None
    try:
      result = await self._client.users_info(user=user_id)
      user_info = result.get("user", {})
      return user_info.get("real_name") or user_info.get("profile", {}).get("display_name") or user_info.get("name")
    except Exception as e:
      log_debug(f"[slack] Failed to resolve username for {user_id}: {e}")
      return None

  # --- Media extraction ---

  async def _extract_media(
    self,
    slack_files: List[Dict[str, Any]],
  ) -> tuple[
    Optional[List[Image]],
    Optional[List[Audio]],
    Optional[List[Video]],
    Optional[List[File]],
  ]:
    """Extract and categorize media from Slack file objects."""
    images: List[Image] = []
    audio_list: List[Audio] = []
    video_list: List[Video] = []
    files_list: List[File] = []

    for file_info in slack_files:
      mimetype = file_info.get("mimetype", "")
      url_private = file_info.get("url_private", "")
      filename = file_info.get("name", "file")
      size = file_info.get("size")

      if not url_private:
        continue

      content = await self._download_file(url_private)
      if content is None:
        continue

      if mimetype.startswith("image/"):
        fmt = mimetype.split("/")[1] if "/" in mimetype else None
        images.append(Image(content=content, mime_type=mimetype, format=fmt))
      elif mimetype.startswith("audio/"):
        fmt = mimetype.split("/")[1] if "/" in mimetype else None
        duration = file_info.get("duration_ms")
        audio_list.append(
          Audio(
            content=content,
            mime_type=mimetype,
            format=fmt,
            duration=duration / 1000.0 if duration else None,
          )
        )
      elif mimetype.startswith("video/"):
        fmt = mimetype.split("/")[1] if "/" in mimetype else None
        video_list.append(Video(content=content, mime_type=mimetype, format=fmt))
      else:
        # General file — only set mime_type if it's in the allowed list
        file_kwargs: Dict[str, Any] = {
          "content": content,
          "filename": filename,
          "size": size,
        }
        if mimetype in File.valid_mime_types():
          file_kwargs["mime_type"] = mimetype
        files_list.append(File(**file_kwargs))

    return (
      images or None,
      audio_list or None,
      video_list or None,
      files_list or None,
    )

  async def _download_file(self, url_private: str) -> Optional[bytes]:
    """Download a file from Slack using the bot token for auth."""
    try:
      import httpx

      async with httpx.AsyncClient(
        timeout=httpx.Timeout(
          connect=self._slack_config.connect_timeout,
          read=self._slack_config.request_timeout,
          write=self._slack_config.request_timeout,
          pool=self._slack_config.connect_timeout,
        ),
      ) as client:
        response = await client.get(
          url_private,
          headers={"Authorization": f"Bearer {self._slack_config.bot_token}"},
        )
        if response.status_code == 200:
          return response.content
        log_warning(f"[slack] File download returned status {response.status_code}")
    except Exception as e:
      log_warning(f"[slack] File download failed: {e}")
    return None

  async def _download_url(self, url: str) -> Optional[bytes]:
    """Download content from a generic URL."""
    try:
      import httpx

      async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)) as client:
        response = await client.get(url)
        if response.status_code == 200:
          return response.content
    except Exception as e:
      log_warning(f"[slack] URL download failed: {e}")
    return None

  # --- Error handling ---

  def _handle_api_error(self, error: Exception, method: str) -> None:
    """Map Slack SDK errors to interface error types and log them."""
    try:
      from slack_sdk.errors import SlackApiError

      if isinstance(error, SlackApiError):
        resp = error.response
        status = resp.status_code if hasattr(resp, "status_code") else 0
        slack_error = resp.get("error", str(error)) if hasattr(resp, "get") else str(error)

        if status == 401 or slack_error == "invalid_auth":
          raise InterfaceAuthenticationError(
            f"Slack auth failed on {method}: {slack_error}",
            platform="slack",
          ) from error
        if status == 429 or slack_error == "rate_limited":
          retry_after = None
          if hasattr(resp, "headers"):
            ra = resp.headers.get("Retry-After")
            if ra:
              retry_after = float(ra)
          raise InterfaceRateLimitError(
            f"Slack rate limited on {method}: {slack_error}",
            platform="slack",
            retry_after=retry_after,
          ) from error
        if status == 400:
          raise InterfaceMessageError(
            f"Slack bad request on {method}: {slack_error}",
            platform="slack",
          ) from error

        log_error(f"[slack] API error on {method}: {slack_error}")
        return
    except (InterfaceAuthenticationError, InterfaceRateLimitError, InterfaceMessageError):
      raise
    except ImportError:
      pass

    log_error(f"[slack] Error calling {method}: {error}")

  # --- HTTP mode helpers ---

  def get_bolt_app(self) -> "AsyncApp":
    """Return the underlying Bolt AsyncApp for HTTP mode mounting.

    Use this to mount Slack routes on an existing FastAPI app::

        from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

        handler = AsyncSlackRequestHandler(interface.get_bolt_app())
        app.post("/slack/events")(handler.handle)

    Returns:
      The Bolt AsyncApp instance.

    Raises:
      RuntimeError: If the interface has not been started yet.
    """
    if self._bolt_app is None:
      raise RuntimeError("Interface not started. Call start() first or use async context manager.")
    return self._bolt_app
