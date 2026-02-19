"""Telegram interface implementation using the Telegram Bot API."""

import asyncio
import contextlib
import hmac
from typing import Any, Dict, List, Optional

import httpx

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.errors import (
  InterfaceAuthenticationError,
  InterfaceConnectionError,
  InterfaceMessageError,
  InterfaceRateLimitError,
)
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.telegram.config import TelegramConfig
from definable.media import Audio, File, Image
from definable.utils.log import log_debug, log_error, log_info, log_warning


class TelegramInterface(BaseInterface):
  """Interface connecting an agent to Telegram via the Bot API.

  Supports both polling (for development) and webhook (for production)
  modes. Uses httpx for async HTTP calls.

  Args:
    agent: The Agent instance.
    config: TelegramConfig with bot token and settings.
    session_manager: Optional session manager.
    hooks: Optional list of hooks.

  Example (polling):
    interface = TelegramInterface(
      agent=agent,
      config=TelegramConfig(bot_token="BOT_TOKEN"),
    )
    async with interface:
      await interface.serve_forever()

  Example (webhook):
    interface = TelegramInterface(
      agent=agent,
      config=TelegramConfig(
        bot_token="BOT_TOKEN",
        mode="webhook",
        webhook_url="https://example.com/webhook/telegram",
      ),
    )
    async with interface:
      await interface.serve_forever()
  """

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._tg_config: TelegramConfig = self.config  # type: ignore[assignment]
    self._base_url = f"https://api.telegram.org/bot{self._tg_config.bot_token}"
    self._client: Optional[httpx.AsyncClient] = None
    self._poll_task: Optional[asyncio.Task[None]] = None
    self._webhook_runner: Any = None
    self._webhook_site: Any = None
    self._offset: int = 0

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

    # Verify bot token
    await self._verify_bot()

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

    if self._webhook_site is not None:
      await self._teardown_webhook()

    if self._client is not None:
      await self._client.aclose()
      self._client = None

  # --- Bot verification ---

  async def _verify_bot(self) -> None:
    """Verify the bot token by calling getMe."""
    result = await self._api_call("getMe")
    bot_username = result.get("username", "unknown")
    log_info(f"[telegram] Connected as @{bot_username}")

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

  async def _process_update(self, update: Dict[str, Any]) -> None:
    """Process a single Telegram update."""
    message = update.get("message") or update.get("edited_message")
    if message is None:
      return
    await self.handle_platform_message(message)

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

  async def _webhook_handler(self, request: Any) -> Any:
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

    # Process the update
    message = data.get("message") or data.get("edited_message")
    if message is not None:
      asyncio.create_task(self.handle_platform_message(message))

    return web.Response(status=200, text="OK")

  # --- Inbound conversion ---

  async def _convert_inbound(self, raw_message: Dict[str, Any]) -> Optional[InterfaceMessage]:
    """Convert a Telegram message dict to InterfaceMessage."""
    chat = raw_message.get("chat", {})
    from_user = raw_message.get("from", {})

    user_id = str(from_user.get("id", ""))
    chat_id = str(chat.get("id", ""))
    message_id = str(raw_message.get("message_id", ""))

    # Access control
    if self._tg_config.allowed_user_ids is not None:
      if int(user_id) not in self._tg_config.allowed_user_ids:
        log_debug(f"[telegram] Ignoring message from unauthorized user {user_id}")
        return None

    if self._tg_config.allowed_chat_ids is not None:
      if int(chat_id) not in self._tg_config.allowed_chat_ids:
        log_debug(f"[telegram] Ignoring message from unauthorized chat {chat_id}")
        return None

    # Extract text (message text or caption)
    text = raw_message.get("text") or raw_message.get("caption")

    # Extract username
    username = from_user.get("username") or from_user.get("first_name")

    # Extract media
    images: Optional[List[Image]] = None
    audio_list: Optional[List[Audio]] = None
    files: Optional[List[File]] = None

    # Photos — Telegram sends multiple sizes, pick the largest
    photo_list = raw_message.get("photo")
    if photo_list:
      largest_photo = max(photo_list, key=lambda p: p.get("file_size", 0))
      file_id = largest_photo.get("file_id", "")
      file_url = await self._get_file_url(file_id)
      if file_url:
        images = [Image(url=file_url)]

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

    # Reply context
    reply_to = raw_message.get("reply_to_message")
    reply_to_message_id: Optional[str] = None
    if reply_to:
      reply_to_message_id = str(reply_to.get("message_id", ""))

    return InterfaceMessage(
      text=text,
      platform="telegram",
      platform_user_id=user_id,
      platform_chat_id=chat_id,
      platform_message_id=message_id,
      username=username,
      images=images,
      audio=audio_list,
      files=files,
      reply_to_message_id=reply_to_message_id,
      metadata={"raw": raw_message},
    )

  # --- Response sending ---

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send response back to Telegram."""
    chat_id = original_msg.platform_chat_id

    # Send typing indicator
    if self.config.typing_indicator:
      with contextlib.suppress(Exception):
        await self._api_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})

    # Send text content (split if needed)
    if response.content:
      await self._send_text(chat_id, response.content, original_msg.platform_message_id)

    # Send images
    if response.images:
      for image in response.images:
        await self._send_photo(chat_id, image)

    # Send files
    if response.files:
      for file in response.files:
        await self._send_document(chat_id, file)

  async def _send_text(self, chat_id: str, text: str, reply_to_message_id: Optional[str] = None) -> None:
    """Send text message, splitting if it exceeds the max length."""
    max_len = self._tg_config.max_message_length
    chunks = self._split_text(text, max_len)

    for i, chunk in enumerate(chunks):
      data: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": chunk,
      }
      if self._tg_config.parse_mode is not None:
        data["parse_mode"] = self._tg_config.parse_mode

      # Only reply to the original message for the first chunk
      if i == 0 and reply_to_message_id:
        data["reply_to_message_id"] = reply_to_message_id

      try:
        await self._api_call("sendMessage", data)
      except InterfaceMessageError:
        # If parse_mode fails (e.g. invalid HTML), retry without it
        if self._tg_config.parse_mode is not None:
          data.pop("parse_mode", None)
          await self._api_call("sendMessage", data)
        else:
          raise

  async def _send_photo(self, chat_id: str, image: Image) -> None:
    """Send a photo to a Telegram chat."""
    if image.url:
      await self._api_call("sendPhoto", {"chat_id": chat_id, "photo": image.url})
    elif image.filepath:
      await self._upload_file("sendPhoto", chat_id, "photo", str(image.filepath))

  async def _send_document(self, chat_id: str, file: File) -> None:
    """Send a document to a Telegram chat."""
    if file.url:
      await self._api_call("sendDocument", {"chat_id": chat_id, "document": file.url})
    elif file.filepath:
      await self._upload_file("sendDocument", chat_id, "document", str(file.filepath))

  async def _upload_file(self, method: str, chat_id: str, field_name: str, filepath: str) -> None:
    """Upload a file to Telegram via multipart form."""
    assert self._client is not None
    with open(filepath, "rb") as f:
      response = await self._client.post(
        f"{self._base_url}/{method}",
        data={"chat_id": chat_id},
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
