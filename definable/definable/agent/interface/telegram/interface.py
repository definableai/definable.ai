"""Telegram interface implementation using the Telegram Bot API."""

import asyncio
import contextlib
import contextvars
import hmac
import re
import time as _time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Literal, Optional

import httpx

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import InterfaceSession, SessionManager
from definable.agent.interface.telegram.config import TelegramConfig
from definable.agent.interface.telegram.formatting import markdown_to_telegram_html, split_html
from definable.agent.interface.telegram.stickers import StickerCache
from definable.media import Audio, File, Image, Video
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from aiohttp import web

  from definable.agent.agent import Agent
  from definable.agent.events import RunOutput
  from definable.agent.interface.identity import IdentityResolver
  from definable.agent.interface.telegram.keyboards import InlineKeyboard

# Type alias for callback handlers
CallbackHandler = Callable[[Dict[str, Any]], Awaitable[Optional[str]]]

# Context variable for agent-controlled buttons (set by TelegramOutputSkill tools, read by _send_response)
_pending_buttons_var: contextvars.ContextVar[Optional[List[List[Any]]]] = contextvars.ContextVar("_pending_buttons_var", default=None)


class _TypingCircuitBreaker:
  """Circuit breaker for per-chat typing indicators.

  Tracks consecutive failures per chat and applies exponential
  backoff, suspending indicators after too many failures.

  Args:
    max_failures: Consecutive failures before suspension.
    base_backoff: Initial backoff in seconds.
    max_backoff: Maximum backoff in seconds.
  """

  def __init__(self, max_failures: int = 10, base_backoff: float = 1.0, max_backoff: float = 300.0) -> None:
    self._max_failures = max_failures
    self._base_backoff = base_backoff
    self._max_backoff = max_backoff
    self._failures: Dict[str, int] = {}
    self._suspended_until: Dict[str, float] = {}

  def should_send(self, chat_id: str) -> bool:
    """Check whether typing indicators should be sent for this chat."""
    if chat_id in self._suspended_until:
      if _time.monotonic() < self._suspended_until[chat_id]:
        return False
      # Suspension expired — reset
      del self._suspended_until[chat_id]
      self._failures.pop(chat_id, None)
    return True

  def record_success(self, chat_id: str) -> None:
    """Record a successful typing indicator send."""
    self._failures.pop(chat_id, None)
    self._suspended_until.pop(chat_id, None)

  def record_failure(self, chat_id: str) -> None:
    """Record a failed typing indicator send."""
    count = self._failures.get(chat_id, 0) + 1
    self._failures[chat_id] = count
    if count >= self._max_failures:
      backoff = min(self._base_backoff * (2 ** (count - self._max_failures)), self._max_backoff)
      self._suspended_until[chat_id] = _time.monotonic() + backoff


class _SlidingWindowRateLimiter:
  """Sliding-window rate limiter for per-user message throttling.

  Args:
    max_requests: Maximum requests allowed in the window.
    window_seconds: Window duration in seconds.
  """

  def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
    self._max_requests = max_requests
    self._window = window_seconds
    self._requests: Dict[str, List[float]] = {}

  def is_allowed(self, key: str) -> bool:
    """Check if a request is within the rate limit.

    Args:
      key: Unique identifier (e.g. user_id).

    Returns:
      True if allowed, False if rate-limited.
    """
    now = _time.monotonic()
    window_start = now - self._window

    if key not in self._requests:
      self._requests[key] = [now]
      return True

    # Prune old entries
    timestamps = self._requests[key]
    self._requests[key] = [t for t in timestamps if t > window_start]
    timestamps = self._requests[key]

    if len(timestamps) >= self._max_requests:
      return False

    timestamps.append(now)
    return True


class _OutboundRateLimiter:
  """Token-bucket rate limiter for outbound API calls.

  Args:
    calls_per_second: Maximum API calls per second.
  """

  def __init__(self, calls_per_second: float = 30.0) -> None:
    self._interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
    self._last_call: float = 0

  async def acquire(self) -> None:
    """Wait until an API call is allowed."""
    if self._interval <= 0:
      return
    now = _time.monotonic()
    elapsed = now - self._last_call
    if elapsed < self._interval:
      await asyncio.sleep(self._interval - elapsed)
    self._last_call = _time.monotonic()


class TelegramInterface(BaseInterface):
  """Interface connecting an agent to Telegram via the Bot API.

  Supports both polling (for development) and webhook (for production)
  modes. Uses httpx for async HTTP calls.

  Features:
    - Markdown→HTML auto-conversion (Phase 1)
    - Smart HTML chunking preserving tags (Phase 2)
    - Response streaming via message editing (Phase 4)
    - Inline keyboard support (Phase 6)
    - Group chat mention-only mode (Phase 7)
    - Forum topic session isolation (Phase 8)
    - Video, sticker, forward, edit support (Phase 9-12)
    - Typing indicator circuit breaker (Phase 13)
    - Rate limiting (Phase 14)
    - Update deduplication (Phase 15)
    - Bot command menu sync (Phase 16)
    - DM vs group access policies (Phase 17)
    - Location messages (Phase 18)
    - Media group batching (Phase 19)
    - Reaction handling (Phase 20)

  Example (polling)::

      interface = TelegramInterface(
        agent=agent,
        bot_token="BOT_TOKEN",
      )
      async with interface:
        await interface.serve_forever()

  Example (webhook)::

      interface = TelegramInterface(
        agent=agent,
        bot_token="BOT_TOKEN",
        mode="webhook",
        webhook_url="https://example.com/webhook/telegram",
      )
      async with interface:
        await interface.serve_forever()
  """

  def __init__(
    self,
    *,
    # Telegram-specific
    bot_token: str = "",
    mode: Literal["polling", "webhook"] = "polling",
    webhook_url: Optional[str] = None,
    webhook_path: str = "/webhook/telegram",
    webhook_port: int = 8443,
    webhook_secret: Optional[str] = None,
    allowed_user_ids: Optional[List[int]] = None,
    allowed_chat_ids: Optional[List[int]] = None,
    parse_mode: Literal["HTML", "MarkdownV2", "Markdown", None] = "HTML",
    auto_format: bool = True,
    polling_interval: float = 0.5,
    polling_timeout: int = 30,
    connect_timeout: float = 10.0,
    request_timeout: float = 60.0,
    # Streaming (Phase 4)
    streaming: bool = True,
    stream_edit_interval: float = 1.0,
    stream_min_chars: int = 30,
    stream_tool_indicator: bool = True,
    # Callback queries (Phase 5)
    handle_callback_queries: bool = True,
    # Group chat (Phase 7)
    group_mode: Literal["mention", "always", "disabled"] = "mention",
    # Forum topics (Phase 8)
    enable_forum_topics: bool = True,
    # Rate limiting (Phase 14)
    outbound_rate_limit: float = 30.0,
    # Commands (Phase 16)
    commands: Optional[Dict[str, str]] = None,
    sync_commands_on_startup: bool = True,
    # DM/Group policies (Phase 17)
    dm_policy: Literal["open", "allowlist", "disabled"] = "open",
    group_policy: Literal["open", "allowlist", "disabled"] = "open",
    dm_allowlist: Optional[List[int]] = None,
    group_allowlist: Optional[List[int]] = None,
    # Media groups (Phase 19)
    media_group_timeout: float = 0.5,
    # Reactions (Phase 20)
    handle_reactions: bool = False,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 4096,
    rate_limit_messages_per_minute: int = 30,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
  ) -> None:
    resolved_config = TelegramConfig(
      bot_token=bot_token,
      mode=mode,
      webhook_url=webhook_url,
      webhook_path=webhook_path,
      webhook_port=webhook_port,
      webhook_secret=webhook_secret,
      allowed_user_ids=allowed_user_ids,
      allowed_chat_ids=allowed_chat_ids,
      parse_mode=parse_mode,
      auto_format=auto_format,
      polling_interval=polling_interval,
      polling_timeout=polling_timeout,
      connect_timeout=connect_timeout,
      request_timeout=request_timeout,
      streaming=streaming,
      stream_edit_interval=stream_edit_interval,
      stream_min_chars=stream_min_chars,
      stream_tool_indicator=stream_tool_indicator,
      handle_callback_queries=handle_callback_queries,
      group_mode=group_mode,
      enable_forum_topics=enable_forum_topics,
      outbound_rate_limit=outbound_rate_limit,
      commands=commands,
      sync_commands_on_startup=sync_commands_on_startup,
      dm_policy=dm_policy,
      group_policy=group_policy,
      dm_allowlist=dm_allowlist,
      group_allowlist=group_allowlist,
      media_group_timeout=media_group_timeout,
      handle_reactions=handle_reactions,
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
    self._tg_config: TelegramConfig = self.config  # type: ignore[assignment]
    self._base_url = f"https://api.telegram.org/bot{self._tg_config.bot_token}"
    self._client: Optional[httpx.AsyncClient] = None
    self._poll_task: Optional[asyncio.Task[None]] = None
    self._webhook_runner: Optional["web.AppRunner"] = None
    self._webhook_site: Optional["web.TCPSite"] = None
    self._offset: int = 0

    # Phase 5: Callback query handlers
    self._callback_handlers: List[tuple[re.Pattern[str], CallbackHandler]] = []

    # Phase 7: Bot identity (set in _verify_bot)
    self._bot_username: Optional[str] = None
    self._bot_id: Optional[int] = None

    # Phase 10: Sticker cache
    self._sticker_cache = StickerCache()

    # Phase 13: Typing circuit breaker
    self._typing_cb = _TypingCircuitBreaker()

    # Phase 14: Rate limiters
    self._inbound_rate_limiter = _SlidingWindowRateLimiter(
      max_requests=self._tg_config.rate_limit_messages_per_minute,
      window_seconds=60.0,
    )
    self._outbound_rate_limiter = _OutboundRateLimiter(
      calls_per_second=self._tg_config.outbound_rate_limit,
    )

    # Phase 15: Update deduplication
    self._seen_updates: OrderedDict[int, None] = OrderedDict()
    self._max_seen_updates = 1000

    # Phase 19: Media group buffering
    self._media_group_buffers: Dict[str, Dict[str, Any]] = {}
    self._media_group_tasks: Dict[str, asyncio.Task[None]] = {}

  # --- Lifecycle ---

  async def _start_receiver(self) -> None:
    self._client = httpx.AsyncClient(
      timeout=httpx.Timeout(
        connect=self._tg_config.connect_timeout,
        read=self._tg_config.request_timeout,
        write=self._tg_config.request_timeout,
        pool=self._tg_config.connect_timeout,
      ),
    )

    # Verify bot token and store identity
    await self._verify_bot()

    # Phase 16: Sync commands on startup
    if self._tg_config.sync_commands_on_startup and self._tg_config.commands:
      await self._sync_commands()

    if self._tg_config.mode == "polling":
      # Delete any existing webhook before polling
      await self._api_call("deleteWebhook")
      self._poll_task = asyncio.create_task(self._poll_loop())
      log_info("[telegram] Polling started")
    else:
      await self._setup_webhook()
      log_info("[telegram] Webhook started")

  async def _stop_receiver(self) -> None:
    if self._poll_task is not None:
      self._poll_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._poll_task
      self._poll_task = None

    # Cancel any pending media group tasks
    for task in self._media_group_tasks.values():
      task.cancel()
    self._media_group_tasks.clear()
    self._media_group_buffers.clear()

    if self._webhook_site is not None:
      await self._teardown_webhook()

    if self._client is not None:
      await self._client.aclose()
      self._client = None

  # --- Bot verification ---

  async def _verify_bot(self) -> None:
    """Verify the bot token by calling getMe and store bot identity."""
    result = await self._api_call("getMe")
    self._bot_username = result.get("username", "unknown")
    self._bot_id = result.get("id")
    log_info(f"[telegram] Connected as @{self._bot_username}")

  # --- Phase 16: Command menu sync ---

  async def _sync_commands(self) -> None:
    """Sync bot command menu with Telegram."""
    commands = self._tg_config.commands
    if not commands:
      return
    cmd_list = [{"command": cmd, "description": desc} for cmd, desc in commands.items()]
    try:
      await self._api_call("setMyCommands", {"commands": cmd_list})
      log_info(f"[telegram] Synced {len(cmd_list)} bot commands")
    except Exception as e:
      log_warning(f"[telegram] Failed to sync commands: {e}")

  # --- Polling ---

  async def _poll_loop(self) -> None:
    """Long-polling loop that fetches updates from Telegram."""
    while self._running:
      try:
        updates = await self._get_updates()
        for update in updates:
          update_id = update.get("update_id", 0)
          if update_id >= self._offset:
            self._offset = update_id + 1

          # Phase 15: Deduplication
          if self._is_duplicate_update(update_id):
            continue

          # Process each update concurrently
          asyncio.create_task(self._process_update(update))
      except asyncio.CancelledError:
        break
      except httpx.TimeoutException:
        # Long-polling timeout is normal
        continue
      except Exception as e:
        log_error(f"[telegram] Polling error: {e}")
        await asyncio.sleep(self._tg_config.polling_interval)

  async def _get_updates(self) -> List[Dict[str, Any]]:
    """Fetch updates via long polling."""
    data: Dict[str, Any] = {
      "offset": self._offset,
      "timeout": self._tg_config.polling_timeout,
      "allowed_updates": self._get_allowed_updates(),
    }
    assert self._client is not None
    response = await self._client.post(
      f"{self._base_url}/getUpdates",
      json=data,
      timeout=httpx.Timeout(
        connect=self._tg_config.connect_timeout,
        read=self._tg_config.polling_timeout + 5.0,
        write=self._tg_config.request_timeout,
        pool=self._tg_config.connect_timeout,
      ),
    )
    result = response.json()
    if not result.get("ok"):
      raise InterfaceConnectionError(
        f"getUpdates failed: {result.get('description', 'Unknown error')}",
        platform="telegram",
      )
    return result.get("result", [])

  def _get_allowed_updates(self) -> List[str]:
    """Build the list of update types to receive."""
    updates = ["message", "edited_message"]
    if self._tg_config.handle_callback_queries:
      updates.append("callback_query")
    if self._tg_config.handle_reactions:
      updates.append("message_reaction")
    return updates

  # --- Phase 15: Update deduplication ---

  def _is_duplicate_update(self, update_id: int) -> bool:
    """Check and record an update_id for deduplication."""
    if update_id in self._seen_updates:
      return True
    self._seen_updates[update_id] = None
    while len(self._seen_updates) > self._max_seen_updates:
      self._seen_updates.popitem(last=False)
    return False

  # --- Update processing ---

  async def _process_update(self, update: Dict[str, Any]) -> None:
    """Process a single Telegram update."""
    # Phase 5: Callback queries
    if "callback_query" in update and self._tg_config.handle_callback_queries:
      await self._handle_callback_query(update["callback_query"])
      return

    # Phase 20: Reactions
    if "message_reaction" in update and self._tg_config.handle_reactions:
      await self._handle_reaction(update["message_reaction"])
      return

    # Phase 12: Differentiate edited messages
    is_edit = "edited_message" in update
    message = update.get("message") or update.get("edited_message")
    if message is None:
      return

    # Phase 19: Media groups
    media_group_id = message.get("media_group_id")
    if media_group_id:
      await self._buffer_media_group(media_group_id, message, is_edit)
      return

    await self._dispatch_message(message, is_edit)

  async def _dispatch_message(self, message: Dict[str, Any], is_edit: bool = False) -> None:
    """Dispatch a single message (or merged media group) to the pipeline."""
    if is_edit:
      message["_is_edit"] = True
    await self.handle_platform_message(message)

  # --- Phase 5: Callback queries ---

  def register_callback(self, pattern: str, handler: CallbackHandler) -> None:
    """Register a callback handler for inline keyboard button presses.

    Args:
      pattern: Regex pattern to match against callback_data.
      handler: Async function receiving the callback_query dict.
        Returns optional text response.
    """
    self._callback_handlers.append((re.compile(pattern), handler))

  async def _handle_callback_query(self, callback_query: Dict[str, Any]) -> None:
    """Process an incoming callback query."""
    query_id = callback_query.get("id", "")
    data = callback_query.get("data", "")

    # Try registered handlers
    for pattern, handler in self._callback_handlers:
      if pattern.search(data):
        try:
          response_text = await handler(callback_query)
          await self._answer_callback_query(query_id, text=response_text)
        except Exception as e:
          log_error(f"[telegram] Callback handler error: {e}")
          await self._answer_callback_query(query_id, text="Error processing action")
        return

    # Fallback: treat as text message to agent
    message = callback_query.get("message", {})
    if message:
      chat = message.get("chat", {})
      from_user = callback_query.get("from", {})
      synthetic = {
        "chat": chat,
        "from": from_user,
        "message_id": message.get("message_id", 0),
        "text": data,
      }
      await self._answer_callback_query(query_id)
      await self.handle_platform_message(synthetic)

  async def _answer_callback_query(
    self,
    callback_query_id: str,
    text: Optional[str] = None,
    show_alert: bool = False,
  ) -> None:
    """Answer a callback query (required by Telegram API)."""
    data: Dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
      data["text"] = text
    if show_alert:
      data["show_alert"] = True
    with contextlib.suppress(Exception):
      await self._api_call("answerCallbackQuery", data)

  # --- Phase 20: Reactions ---

  async def _handle_reaction(self, reaction_update: Dict[str, Any]) -> None:
    """Process a message_reaction update as synthetic text."""
    chat = reaction_update.get("chat", {})
    user = reaction_update.get("user") or reaction_update.get("actor_chat", {})
    message_id = reaction_update.get("message_id", 0)

    new_reactions = reaction_update.get("new_reaction", [])
    if not new_reactions:
      return

    emojis = [r.get("emoji", "") for r in new_reactions if r.get("type") == "emoji"]
    if not emojis:
      return

    emoji_str = " ".join(emojis)
    synthetic = {
      "chat": chat,
      "from": user,
      "message_id": message_id,
      "text": f"[Reaction: {emoji_str} on message {message_id}]",
    }
    await self.handle_platform_message(synthetic)

  # --- Phase 19: Media group buffering ---

  async def _buffer_media_group(self, media_group_id: str, message: Dict[str, Any], is_edit: bool) -> None:
    """Buffer a message that's part of a media group."""
    if media_group_id not in self._media_group_buffers:
      self._media_group_buffers[media_group_id] = {
        "messages": [],
        "is_edit": is_edit,
        "first_message": message,
      }

    self._media_group_buffers[media_group_id]["messages"].append(message)

    # Cancel existing timer and reset
    if media_group_id in self._media_group_tasks:
      self._media_group_tasks[media_group_id].cancel()

    self._media_group_tasks[media_group_id] = asyncio.create_task(self._flush_media_group(media_group_id))

  async def _flush_media_group(self, media_group_id: str) -> None:
    """Wait for timeout then dispatch combined media group."""
    await asyncio.sleep(self._tg_config.media_group_timeout)

    buf = self._media_group_buffers.pop(media_group_id, None)
    self._media_group_tasks.pop(media_group_id, None)
    if not buf:
      return

    # Merge: use first message as base, attach media from all messages
    first = buf["first_message"]
    is_edit = buf["is_edit"]

    # Collect all media into the first message
    all_photos: List[Any] = []
    all_documents: List[Any] = []
    all_videos: List[Any] = []
    captions: List[str] = []

    for msg in buf["messages"]:
      if msg.get("photo"):
        all_photos.append(max(msg["photo"], key=lambda p: p.get("file_size", 0)))
      if msg.get("document"):
        all_documents.append(msg["document"])
      if msg.get("video"):
        all_videos.append(msg["video"])
      cap = msg.get("caption")
      if cap:
        captions.append(cap)

    # Build merged message
    merged = dict(first)
    if all_photos:
      merged["_media_group_photos"] = all_photos
    if all_documents:
      merged["_media_group_documents"] = all_documents
    if all_videos:
      merged["_media_group_videos"] = all_videos
    if captions:
      merged["text"] = merged.get("text") or "\n".join(captions)
    merged["_media_group_id"] = media_group_id

    await self._dispatch_message(merged, is_edit)

  # --- Webhook ---

  async def _setup_webhook(self) -> None:
    """Set up webhook mode with an aiohttp server."""
    try:
      from aiohttp import web
    except ImportError:
      raise InterfaceConnectionError(
        "aiohttp is required for webhook mode. Install it with: pip install aiohttp",
        platform="telegram",
      )

    # Set the webhook on Telegram
    set_data: Dict[str, Any] = {
      "url": f"{self._tg_config.webhook_url}{self._tg_config.webhook_path}",
      "allowed_updates": self._get_allowed_updates(),
    }
    if self._tg_config.webhook_secret:
      set_data["secret_token"] = self._tg_config.webhook_secret

    await self._api_call("setWebhook", set_data)

    # Create aiohttp application
    app = web.Application()
    app.router.add_post(self._tg_config.webhook_path, self._webhook_handler)

    self._webhook_runner = web.AppRunner(app)
    await self._webhook_runner.setup()
    self._webhook_site = web.TCPSite(
      self._webhook_runner,
      "0.0.0.0",  # noqa: S104
      self._tg_config.webhook_port,
    )
    await self._webhook_site.start()
    log_info(f"[telegram] Webhook server listening on port {self._tg_config.webhook_port}")

  async def _teardown_webhook(self) -> None:
    """Tear down the webhook server."""
    if self._webhook_site is not None:
      await self._webhook_site.stop()
      self._webhook_site = None
    if self._webhook_runner is not None:
      await self._webhook_runner.cleanup()
      self._webhook_runner = None
    # Remove webhook from Telegram
    with contextlib.suppress(Exception):
      await self._api_call("deleteWebhook")

  async def _webhook_handler(self, request: "web.Request") -> "web.Response":
    """Handle an incoming webhook request."""
    from aiohttp import web

    # Verify secret token if configured
    if self._tg_config.webhook_secret:
      token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
      if not hmac.compare_digest(token, self._tg_config.webhook_secret):
        return web.Response(status=403, text="Forbidden")

    try:
      data = await request.json()
    except Exception:
      return web.Response(status=400, text="Bad Request")

    update_id = data.get("update_id")
    if update_id is not None and self._is_duplicate_update(update_id):
      return web.Response(status=200, text="OK")

    # Process the update
    asyncio.create_task(self._process_update(data))
    return web.Response(status=200, text="OK")

  # --- Inbound conversion ---

  async def _convert_inbound(self, raw_message: Dict[str, Any]) -> Optional[InterfaceMessage]:
    """Convert a Telegram message dict to InterfaceMessage."""
    chat = raw_message.get("chat", {})
    from_user = raw_message.get("from", {})

    user_id = str(from_user.get("id", ""))
    chat_id = str(chat.get("id", ""))
    message_id = str(raw_message.get("message_id", ""))
    chat_type = chat.get("type", "private")  # private, group, supergroup, channel

    # Phase 17: DM vs group policies
    if not self._check_chat_policy(chat_type, user_id, chat_id):
      return None

    # Legacy access control
    if self._tg_config.allowed_user_ids is not None:
      if int(user_id) not in self._tg_config.allowed_user_ids:
        log_debug(f"[telegram] Ignoring message from unauthorized user {user_id}")
        return None

    if self._tg_config.allowed_chat_ids is not None:
      if int(chat_id) not in self._tg_config.allowed_chat_ids:
        log_debug(f"[telegram] Ignoring message from unauthorized chat {chat_id}")
        return None

    # Phase 14: Inbound rate limiting
    if not self._inbound_rate_limiter.is_allowed(user_id):
      log_debug(f"[telegram] Rate-limited user {user_id}")
      with contextlib.suppress(Exception):
        await self._api_call(
          "sendMessage",
          {
            "chat_id": chat_id,
            "text": "You're sending messages too fast. Please wait a moment.",
          },
        )
      return None

    # Phase 7: Group chat filtering
    if chat_type in ("group", "supergroup") and not self._should_respond_in_group(raw_message):
      return None

    # Extract text (message text or caption)
    text = raw_message.get("text") or raw_message.get("caption")

    # Phase 7: Strip bot mention from text
    if text and self._bot_username:
      text = self._strip_bot_mention(text)

    # Phase 12: Mark edited messages
    is_edit = raw_message.get("_is_edit", False)

    # Extract username
    username = from_user.get("username") or from_user.get("first_name")

    # Extract media
    images: Optional[List[Image]] = None
    audio_list: Optional[List[Audio]] = None
    video_list: Optional[List[Video]] = None
    files: Optional[List[File]] = None

    # Photos — Telegram sends multiple sizes, pick the largest
    photo_list = raw_message.get("photo")
    if photo_list:
      largest_photo = max(photo_list, key=lambda p: p.get("file_size", 0))
      file_id = largest_photo.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        images = [Image(url=file_url)]

    # Phase 19: Media group photos
    mg_photos = raw_message.get("_media_group_photos", [])
    for photo_obj in mg_photos:
      file_id = photo_obj.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        if images is None:
          images = []
        images.append(Image(url=file_url))

    # Voice messages
    voice = raw_message.get("voice")
    if voice:
      file_id = voice.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        audio_list = [
          Audio(
            url=file_url,
            mime_type=voice.get("mime_type", "audio/ogg"),
            duration=voice.get("duration"),
          )
        ]

    # Audio files
    audio_msg = raw_message.get("audio")
    if audio_msg:
      file_id = audio_msg.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        audio_list = [
          Audio(
            url=file_url,
            mime_type=audio_msg.get("mime_type", "audio/mpeg"),
            duration=audio_msg.get("duration"),
          )
        ]

    # Phase 9: Video
    video_text: Optional[str] = None
    video_msg = raw_message.get("video")
    if video_msg:
      file_id = video_msg.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        video_list = [
          Video(
            url=file_url,
            mime_type=video_msg.get("mime_type", "video/mp4"),
            duration=video_msg.get("duration"),
            width=video_msg.get("width"),
            height=video_msg.get("height"),
          )
        ]
      # Text fallback so the agent knows a video was sent
      video_text = self._describe_video(video_msg)

    # Phase 9: Video note (circular video)
    video_note = raw_message.get("video_note")
    if video_note:
      file_id = video_note.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        if video_list is None:
          video_list = []
        video_list.append(
          Video(
            url=file_url,
            mime_type="video/mp4",
            duration=video_note.get("duration"),
          )
        )
      video_text = self._describe_video(video_note, kind="Video note")

    # Phase 9: Animation (GIF)
    animation = raw_message.get("animation")
    if animation:
      file_id = animation.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        if video_list is None:
          video_list = []
        video_list.append(
          Video(
            url=file_url,
            mime_type=animation.get("mime_type", "video/mp4"),
            duration=animation.get("duration"),
            width=animation.get("width"),
            height=animation.get("height"),
          )
        )
      video_text = self._describe_video(animation, kind="GIF")

    # Phase 19: Media group videos
    mg_videos = raw_message.get("_media_group_videos", [])
    for vid_obj in mg_videos:
      file_id = vid_obj.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        if video_list is None:
          video_list = []
        video_list.append(
          Video(
            url=file_url,
            mime_type=vid_obj.get("mime_type", "video/mp4"),
            duration=vid_obj.get("duration"),
          )
        )

    # Documents
    document = raw_message.get("document")
    if document:
      file_id = document.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        files = [
          File(
            url=file_url,
            mime_type=document.get("mime_type"),
            filename=document.get("file_name"),
            size=document.get("file_size"),
          )
        ]

    # Phase 19: Media group documents
    mg_docs = raw_message.get("_media_group_documents", [])
    for doc_obj in mg_docs:
      file_id = doc_obj.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        if files is None:
          files = []
        files.append(
          File(
            url=file_url,
            mime_type=doc_obj.get("mime_type"),
            filename=doc_obj.get("file_name"),
            size=doc_obj.get("file_size"),
          )
        )

    # Phase 10: Sticker
    sticker = raw_message.get("sticker")
    sticker_text: Optional[str] = None
    sticker_image: Optional[Image] = None
    if sticker:
      sticker_text = self._sticker_cache.describe_sticker(sticker)
      # If sticker is non-animated, get as image for vision models
      if not sticker.get("is_animated") and not sticker.get("is_video"):
        file_id = sticker.get("file_id", "")
        file_url = await self._get_file_url(file_id)
        if file_url:
          sticker_image = Image(url=file_url)

    # Phase 18: Location
    location = raw_message.get("location")
    location_text: Optional[str] = None
    if location:
      lat = location.get("latitude", 0)
      lng = location.get("longitude", 0)
      location_text = f"[Location: {lat}, {lng}]"

    # Phase 18: Venue
    venue = raw_message.get("venue")
    if venue:
      venue_loc = venue.get("location", {})
      lat = venue_loc.get("latitude", 0)
      lng = venue_loc.get("longitude", 0)
      title = venue.get("title", "")
      address = venue.get("address", "")
      location_text = f"[Venue: {title}, {address} ({lat}, {lng})]"

    # Phase 11: Forward context
    forward_text: Optional[str] = None
    forward_from = raw_message.get("forward_from")
    forward_from_chat = raw_message.get("forward_from_chat")
    if forward_from:
      fwd_name = forward_from.get("first_name", "Unknown")
      fwd_username = forward_from.get("username")
      source = f"@{fwd_username}" if fwd_username else fwd_name
      forward_text = f"[Forwarded from {source}]"
    elif forward_from_chat:
      chat_title = forward_from_chat.get("title", "Unknown")
      forward_text = f"[Forwarded from {chat_title}]"

    # Compose final text
    text_parts: List[str] = []
    if forward_text:
      text_parts.append(forward_text)
    if sticker_text:
      text_parts.append(sticker_text)
    if video_text:
      text_parts.append(video_text)
    if location_text:
      text_parts.append(location_text)
    if text:
      text_parts.append(text)
    final_text = "\n".join(text_parts) if text_parts else text

    # Add sticker image
    if sticker_image:
      if images is None:
        images = []
      images.append(sticker_image)

    # Reply context
    reply_to = raw_message.get("reply_to_message")
    reply_to_message_id: Optional[str] = None
    if reply_to:
      reply_to_message_id = str(reply_to.get("message_id", ""))

    # Phase 8: Forum topic support — encode thread_id into chat_id for session isolation
    thread_id = raw_message.get("message_thread_id")
    effective_chat_id = chat_id
    if (
      self._tg_config.enable_forum_topics
      and thread_id is not None
      and thread_id != 1  # General topic is treated as no-topic
      and chat.get("is_forum")
    ):
      effective_chat_id = f"{chat_id}:topic:{thread_id}"

    # Build metadata
    metadata: Dict[str, Any] = {"raw": raw_message}
    if is_edit:
      metadata["is_edit"] = True
    if thread_id is not None:
      metadata["thread_id"] = thread_id
    if raw_message.get("_media_group_id"):
      metadata["media_group_id"] = raw_message["_media_group_id"]
    if forward_from or forward_from_chat:
      metadata["is_forward"] = True
    if location:
      metadata["location"] = location
    if venue:
      metadata["venue"] = venue

    return InterfaceMessage(
      text=final_text,
      platform="telegram",
      platform_user_id=user_id,
      platform_chat_id=effective_chat_id,
      platform_message_id=message_id,
      username=username,
      images=images,
      audio=audio_list,
      videos=video_list,
      files=files,
      reply_to_message_id=reply_to_message_id,
      metadata=metadata,
    )

  # --- Phase 7: Group chat intelligence ---

  def _should_respond_in_group(self, message: Dict[str, Any]) -> bool:
    """Check if the bot should respond to this group message."""
    group_mode = self._tg_config.group_mode

    if group_mode == "always":
      return True
    if group_mode == "disabled":
      return False

    # "mention" mode: respond only if mentioned or replied-to
    return self._is_bot_mentioned(message)

  def _is_bot_mentioned(self, message: Dict[str, Any]) -> bool:
    """Check if the bot is mentioned in this message."""
    # Check for @mention in entities
    text = message.get("text", "") or ""
    entities = message.get("entities", [])
    for entity in entities:
      if entity.get("type") == "mention":
        offset = entity.get("offset", 0)
        length = entity.get("length", 0)
        mention = text[offset : offset + length]
        if self._bot_username and mention.lower() == f"@{self._bot_username.lower()}":
          return True
      if entity.get("type") == "text_mention":
        user = entity.get("user", {})
        if user.get("id") == self._bot_id:
          return True

    # Check if bot command
    for entity in entities:
      if entity.get("type") == "bot_command":
        return True

    # Check if reply to bot's message
    reply_to = message.get("reply_to_message")
    if reply_to:
      reply_from = reply_to.get("from", {})
      if reply_from.get("id") == self._bot_id:
        return True

    return False

  def _strip_bot_mention(self, text: str) -> str:
    """Remove @bot_username from text."""
    if not self._bot_username:
      return text
    # Case-insensitive removal
    pattern = re.compile(re.escape(f"@{self._bot_username}"), re.IGNORECASE)
    return pattern.sub("", text).strip()

  @staticmethod
  def _describe_video(video_data: Dict[str, Any], kind: str = "Video") -> str:
    """Build a text description for a video/animation/video_note."""
    parts = [f"[{kind}"]
    duration = video_data.get("duration")
    if duration:
      parts.append(f": {duration}s")
    width = video_data.get("width")
    height = video_data.get("height")
    if width and height:
      parts.append(f", {width}x{height}")
    file_name = video_data.get("file_name")
    if file_name:
      parts.append(f", {file_name}")
    return "".join(parts) + "]"

  # --- Phase 17: DM vs group policies ---

  def _check_chat_policy(self, chat_type: str, user_id: str, chat_id: str) -> bool:
    """Check whether the message passes the DM/group policy."""
    if chat_type == "private":
      policy = self._tg_config.dm_policy
      if policy == "disabled":
        return False
      if policy == "allowlist":
        allowlist = self._tg_config.dm_allowlist
        if allowlist is not None and int(user_id) not in allowlist:
          return False
    elif chat_type in ("group", "supergroup"):
      policy = self._tg_config.group_policy
      if policy == "disabled":
        return False
      if policy == "allowlist":
        allowlist = self._tg_config.group_allowlist
        if allowlist is not None and int(chat_id) not in allowlist:
          return False
    return True

  # --- Response sending ---

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send response back to Telegram."""
    chat_id = original_msg.platform_chat_id
    # Extract the raw chat_id (strip topic suffix for API calls)
    api_chat_id = chat_id.split(":topic:")[0] if ":topic:" in chat_id else chat_id
    thread_id = original_msg.metadata.get("thread_id")

    # Check if text was already sent via streaming edits
    streamed = response.metadata.get("_tg_streamed", False) if response.metadata else False

    # Check if agent set pending buttons via TelegramOutputSkill
    pending_buttons = _pending_buttons_var.get(None)
    if pending_buttons is not None:
      _pending_buttons_var.set(None)  # consume

    # Phase 13: Typing indicator with circuit breaker
    if self.config.typing_indicator and not streamed:
      await self._send_typing(api_chat_id, thread_id)

    # Send text content (split if needed), skip if already streamed
    if response.content and not streamed:
      if pending_buttons:
        # Send last chunk with buttons
        from definable.agent.interface.telegram.keyboards import InlineKeyboard

        kb = InlineKeyboard()
        for row in pending_buttons:
          kb.row(*row)
        await self.send_with_buttons(api_chat_id, response.content, kb, thread_id)
      else:
        await self._send_text(api_chat_id, response.content, original_msg.platform_message_id, thread_id)

    # Send images
    if response.images:
      for image in response.images:
        await self._send_photo(api_chat_id, image, thread_id)

    # Send videos
    if response.videos:
      for video in response.videos:
        await self._send_video(api_chat_id, video, thread_id)

    # Send files
    if response.files:
      for file in response.files:
        await self._send_document(api_chat_id, file, thread_id)

  # --- Phase 13: Typing with circuit breaker ---

  async def _send_typing(self, chat_id: str, thread_id: Optional[int] = None) -> None:
    """Send typing indicator with circuit breaker protection."""
    if not self._typing_cb.should_send(chat_id):
      return
    try:
      data: Dict[str, Any] = {"chat_id": chat_id, "action": "typing"}
      if thread_id is not None:
        data["message_thread_id"] = thread_id
      await self._api_call("sendChatAction", data)
      self._typing_cb.record_success(chat_id)
    except Exception:
      self._typing_cb.record_failure(chat_id)

  # --- Phase 3: Message editing ---

  async def _send_message(
    self,
    chat_id: str,
    text: str,
    parse_mode: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
    thread_id: Optional[int] = None,
  ) -> str:
    """Send a message and return its message_id.

    Args:
      chat_id: Target chat.
      text: Message text.
      parse_mode: Telegram parse mode.
      reply_to_message_id: Message to reply to.
      reply_markup: Inline keyboard markup.
      thread_id: Forum topic thread ID.

    Returns:
      The message_id of the sent message as a string.
    """
    data: Dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode:
      data["parse_mode"] = parse_mode
    if reply_to_message_id:
      data["reply_to_message_id"] = reply_to_message_id
    if reply_markup:
      data["reply_markup"] = reply_markup
    if thread_id is not None:
      data["message_thread_id"] = thread_id

    try:
      result = await self._api_call("sendMessage", data)
    except InterfaceMessageError:
      # If parse_mode fails (e.g. invalid HTML), retry without it
      if parse_mode:
        data.pop("parse_mode", None)
        result = await self._api_call("sendMessage", data)
      else:
        raise

    return str(result.get("message_id", ""))

  async def _edit_message(
    self,
    chat_id: str,
    message_id: str,
    text: str,
    parse_mode: Optional[str] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
  ) -> bool:
    """Edit an existing message.

    Args:
      chat_id: Target chat.
      message_id: Message to edit.
      text: New text.
      parse_mode: Telegram parse mode.
      reply_markup: Inline keyboard markup.

    Returns:
      True if successful, False if the message was not modified or not found.
    """
    data: Dict[str, Any] = {
      "chat_id": chat_id,
      "message_id": message_id,
      "text": text,
    }
    if parse_mode:
      data["parse_mode"] = parse_mode
    if reply_markup:
      data["reply_markup"] = reply_markup

    try:
      await self._api_call("editMessageText", data)
      return True
    except InterfaceMessageError as e:
      err_msg = str(e).lower()
      if "message is not modified" in err_msg or "message to edit not found" in err_msg:
        return False
      # On parse_mode failure, retry without it
      if parse_mode:
        data.pop("parse_mode", None)
        try:
          await self._api_call("editMessageText", data)
          return True
        except InterfaceMessageError:
          return False
      return False

  async def _send_text(
    self,
    chat_id: str,
    text: str,
    reply_to_message_id: Optional[str] = None,
    thread_id: Optional[int] = None,
  ) -> None:
    """Send text message, splitting if it exceeds the max length."""
    max_len = self._tg_config.max_message_length
    pm = self._tg_config.parse_mode

    # Phase 1: Auto-convert markdown to HTML
    if self._tg_config.auto_format and pm == "HTML":
      text = markdown_to_telegram_html(text)

    # Phase 2: Use HTML-aware chunking when HTML parse_mode
    if pm == "HTML":
      chunks = split_html(text, max_len)
    else:
      chunks = self._split_text(text, max_len)

    for i, chunk in enumerate(chunks):
      reply_to = reply_to_message_id if i == 0 else None
      await self._send_message(chat_id, chunk, parse_mode=pm, reply_to_message_id=reply_to, thread_id=thread_id)

  # --- Phase 6: Inline keyboards ---

  async def send_with_buttons(
    self,
    chat_id: str,
    text: str,
    keyboard: "InlineKeyboard",
    thread_id: Optional[int] = None,
  ) -> str:
    """Send a message with an inline keyboard.

    Args:
      chat_id: Target chat.
      text: Message text.
      keyboard: InlineKeyboard instance.
      thread_id: Forum topic thread ID.

    Returns:
      The message_id of the sent message.
    """
    pm = self._tg_config.parse_mode
    if self._tg_config.auto_format and pm == "HTML":
      text = markdown_to_telegram_html(text)
    return await self._send_message(
      chat_id,
      text,
      parse_mode=pm,
      reply_markup=keyboard.to_dict(),
      thread_id=thread_id,
    )

  # --- Phase 4: Response streaming ---

  async def _run_agent(self, message: InterfaceMessage, session: InterfaceSession) -> "RunOutput":
    """Run the agent, using streaming if enabled."""
    if not self._tg_config.streaming:
      return await super()._run_agent(message, session)

    try:
      return await self._run_agent_streaming(message, session)
    except Exception as e:
      log_warning(f"[telegram] Streaming failed, falling back to non-streaming: {e}")
      return await super()._run_agent(message, session)

  async def _run_agent_streaming(self, message: InterfaceMessage, session: InterfaceSession) -> "RunOutput":
    """Run the agent with streaming, editing the message in real-time."""
    from definable.agent.events import RunOutput
    from definable.agent.run.agent import RunCompletedEvent, RunContentEvent, ToolCallStartedEvent

    user_id = message.platform_user_id
    if self._identity_resolver is not None:
      resolved = await self._safe_resolve_identity(message.platform, message.platform_user_id)
      if resolved is not None:
        user_id = resolved
    elif "auth_context" in message.metadata:
      user_id = message.metadata["auth_context"].user_id

    assert self.agent is not None
    run_session_id = session.session_id
    if getattr(self.agent, "_session_id_explicit", False):
      run_session_id = self.agent.session_id

    chat_id = message.platform_chat_id
    api_chat_id = chat_id.split(":topic:")[0] if ":topic:" in chat_id else chat_id
    thread_id = message.metadata.get("thread_id")
    pm = self._tg_config.parse_mode

    # Detect thinking and show placeholder
    thinking_active = False
    thinking = getattr(self.agent, "_thinking", None)
    if thinking is not None and getattr(thinking, "enabled", False):
      thinking_active = True

    # Streaming state machine
    buffer = ""
    sent_message_id: Optional[str] = None
    last_edit_time: float = 0
    min_chars = self._tg_config.stream_min_chars
    edit_interval = self._tg_config.stream_edit_interval
    completed_event: Optional[RunCompletedEvent] = None
    all_events: List[Any] = []

    # Send thinking placeholder if agent has thinking enabled
    if thinking_active:
      sent_message_id = await self._send_message(api_chat_id, "Thinking...", thread_id=thread_id)
      last_edit_time = _time.monotonic()

    async for event in self.agent.arun_stream(
      instruction=message.text or "",
      messages=session.messages,
      session_id=run_session_id,
      user_id=user_id,
      images=message.images,
    ):
      all_events.append(event)

      if isinstance(event, RunContentEvent) and event.content:
        buffer += str(event.content)

        # First send: wait for min_chars (or edit thinking placeholder)
        if sent_message_id is None:
          if len(buffer) >= min_chars:
            display = self._format_stream_text(buffer, pm)
            sent_message_id = await self._send_message(api_chat_id, display, parse_mode=pm, thread_id=thread_id)
            last_edit_time = _time.monotonic()
        else:
          # Throttled edits
          now = _time.monotonic()
          if now - last_edit_time >= edit_interval:
            display = self._format_stream_text(buffer, pm)
            await self._edit_message(api_chat_id, sent_message_id, display, parse_mode=pm)
            last_edit_time = now

      elif isinstance(event, ToolCallStartedEvent) and self._tg_config.stream_tool_indicator:
        tool_name = event.tool.tool_name if event.tool else "unknown"
        tool_text = f"Using tool: {tool_name}..."
        if sent_message_id is None:
          sent_message_id = await self._send_message(api_chat_id, tool_text, thread_id=thread_id)
          last_edit_time = _time.monotonic()
        else:
          display = self._format_stream_text(buffer + f"\n\n_{tool_text}_", pm) if buffer else tool_text
          await self._edit_message(api_chat_id, sent_message_id, display, parse_mode=pm)

      elif isinstance(event, RunCompletedEvent):
        completed_event = event

    # Final edit with complete content
    if sent_message_id and buffer:
      display = self._format_stream_text(buffer, pm)
      await self._edit_message(api_chat_id, sent_message_id, display, parse_mode=pm)

    # Build RunOutput from RunCompletedEvent or streamed buffer.
    # Mark metadata so _send_response skips text (already sent via streaming edits).
    streamed_meta = {"_tg_streamed": True}

    if completed_event is not None:
      meta = dict(completed_event.metadata) if completed_event.metadata else {}
      meta.update(streamed_meta)
      return RunOutput(
        content=completed_event.content or buffer,
        content_type=completed_event.content_type,
        parsed=completed_event.parsed,
        reasoning_content=completed_event.reasoning_content,
        images=completed_event.images,
        videos=completed_event.videos,
        audio=completed_event.audio,
        citations=completed_event.citations,
        references=completed_event.references,
        messages=session.messages,
        metrics=completed_event.metrics,
        metadata=meta,
        events=all_events,
      )

    # Fallback: construct from buffer if no RunCompletedEvent received
    if buffer:
      return RunOutput(
        content=buffer,
        messages=session.messages,
        metadata=streamed_meta,
        events=all_events,
      )

    # Last resort: no streaming output at all — re-run non-streaming
    return await super()._run_agent(message, session)

  def _format_stream_text(self, text: str, parse_mode: Optional[str]) -> str:
    """Format text for streaming display."""
    if self._tg_config.auto_format and parse_mode == "HTML":
      return markdown_to_telegram_html(text)
    return text

  # --- Media sending ---

  async def _send_photo(self, chat_id: str, image: Image, thread_id: Optional[int] = None) -> None:
    """Send a photo to a Telegram chat."""
    data: Dict[str, Any] = {"chat_id": chat_id}
    if thread_id is not None:
      data["message_thread_id"] = thread_id
    if image.url:
      data["photo"] = image.url
      await self._api_call("sendPhoto", data)
    elif image.filepath:
      await self._upload_file("sendPhoto", chat_id, "photo", str(image.filepath), thread_id)

  async def _send_video(self, chat_id: str, video: Video, thread_id: Optional[int] = None) -> None:
    """Send a video to a Telegram chat."""
    data: Dict[str, Any] = {"chat_id": chat_id}
    if thread_id is not None:
      data["message_thread_id"] = thread_id
    if video.url:
      data["video"] = video.url
      if video.duration:
        data["duration"] = int(video.duration)
      if video.width:
        data["width"] = video.width
      if video.height:
        data["height"] = video.height
      await self._api_call("sendVideo", data)
    elif video.filepath:
      await self._upload_file("sendVideo", chat_id, "video", str(video.filepath), thread_id)

  async def _send_animation(self, chat_id: str, video: Video, thread_id: Optional[int] = None) -> None:
    """Send an animation (GIF) to a Telegram chat."""
    data: Dict[str, Any] = {"chat_id": chat_id}
    if thread_id is not None:
      data["message_thread_id"] = thread_id
    if video.url:
      data["animation"] = video.url
      await self._api_call("sendAnimation", data)
    elif video.filepath:
      await self._upload_file("sendAnimation", chat_id, "animation", str(video.filepath), thread_id)

  async def _send_document(self, chat_id: str, file: File, thread_id: Optional[int] = None) -> None:
    """Send a document to a Telegram chat."""
    data: Dict[str, Any] = {"chat_id": chat_id}
    if thread_id is not None:
      data["message_thread_id"] = thread_id
    if file.url:
      data["document"] = file.url
      await self._api_call("sendDocument", data)
    elif file.filepath:
      await self._upload_file("sendDocument", chat_id, "document", str(file.filepath), thread_id)

  async def _upload_file(
    self,
    method: str,
    chat_id: str,
    field_name: str,
    filepath: str,
    thread_id: Optional[int] = None,
  ) -> None:
    """Upload a file to Telegram via multipart form."""
    assert self._client is not None
    form_data: Dict[str, str] = {"chat_id": chat_id}
    if thread_id is not None:
      form_data["message_thread_id"] = str(thread_id)
    with open(filepath, "rb") as f:
      response = await self._client.post(
        f"{self._base_url}/{method}",
        data=form_data,
        files={field_name: f},
      )
    result = response.json()
    if not result.get("ok"):
      raise InterfaceMessageError(
        f"{method} failed: {result.get('description', 'Unknown error')}",
        platform="telegram",
      )

  # --- Telegram Bot API ---

  async def _api_call(self, method: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a Telegram Bot API call.

    Args:
      method: API method name (e.g. "sendMessage").
      data: Request payload.

    Returns:
      The "result" field from the API response.

    Raises:
      InterfaceAuthenticationError: On 401 responses.
      InterfaceRateLimitError: On 429 responses.
      InterfaceMessageError: On 400 responses.
      InterfaceConnectionError: On other failures.
    """
    # Phase 14: Outbound rate limiting
    await self._outbound_rate_limiter.acquire()

    assert self._client is not None
    try:
      response = await self._client.post(
        f"{self._base_url}/{method}",
        json=data or {},
      )
    except httpx.ConnectError as e:
      raise InterfaceConnectionError(
        f"Failed to connect to Telegram API: {e}",
        platform="telegram",
      ) from e
    except httpx.TimeoutException as e:
      raise InterfaceConnectionError(
        f"Telegram API request timed out: {e}",
        platform="telegram",
      ) from e

    result = response.json()

    if result.get("ok"):
      return result.get("result", {})

    description = result.get("description", "Unknown error")
    error_code = result.get("error_code", response.status_code)

    if error_code == 401:
      raise InterfaceAuthenticationError(
        f"Invalid bot token: {description}",
        platform="telegram",
      )
    if error_code == 429:
      retry_after = result.get("parameters", {}).get("retry_after")
      raise InterfaceRateLimitError(
        f"Rate limited: {description}",
        platform="telegram",
        retry_after=float(retry_after) if retry_after else None,
      )
    if error_code == 400:
      raise InterfaceMessageError(
        f"Bad request: {description}",
        platform="telegram",
      )
    raise InterfaceConnectionError(
      f"Telegram API error ({error_code}): {description}",
      platform="telegram",
    )

  async def _get_file_url(self, file_id: str) -> Optional[str]:
    """Get a download URL for a Telegram file.

    Args:
      file_id: Telegram file ID.

    Returns:
      Download URL, or None if the file could not be resolved.
    """
    try:
      result = await self._api_call("getFile", {"file_id": file_id})
      file_path = result.get("file_path")
      if file_path:
        return f"https://api.telegram.org/file/bot{self._tg_config.bot_token}/{file_path}"
    except Exception as e:
      log_warning(f"[telegram] Failed to get file URL for {file_id}: {e}")
    return None

  # --- Utilities ---

  @staticmethod
  def _split_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks respecting max_length.

    Tries to split at newlines, then at spaces, falling back to
    hard splits if necessary.
    """
    if len(text) <= max_length:
      return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
      if len(remaining) <= max_length:
        chunks.append(remaining)
        break

      # Try to split at a newline
      split_pos = remaining.rfind("\n", 0, max_length)
      if split_pos == -1:
        # Try to split at a space
        split_pos = remaining.rfind(" ", 0, max_length)
      if split_pos == -1:
        # Hard split
        split_pos = max_length

      chunks.append(remaining[:split_pos])
      remaining = remaining[split_pos:].lstrip("\n")

    return chunks
