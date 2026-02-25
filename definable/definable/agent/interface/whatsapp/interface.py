"""WhatsApp interface — webhook-based agent communication via Twilio WhatsApp API."""

from __future__ import annotations

import hashlib
import hmac
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.agent.interface.whatsapp.config import WhatsAppConfig
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver


class WhatsAppInterface(BaseInterface):
  """Interface connecting an agent to WhatsApp via Twilio's WhatsApp API.

  Uses webhook-based message delivery. Incoming messages arrive at
  ``webhook_path``, and responses are sent via the Twilio REST API.

  Requires the AgentServer (``enable_server=True``) for webhook delivery.
  Configure the Twilio WhatsApp webhook URL to point to your server's
  ``/whatsapp/webhook`` endpoint.

  Example::

    interface = WhatsAppInterface(
      agent=agent,
      account_sid="AC...",
      auth_token="...",
      from_number="whatsapp:+14155238886",
    )
    runtime = AgentRuntime(agent, interfaces=[interface], enable_server=True)
    await runtime.start()
  """

  def __init__(
    self,
    *,
    # WhatsApp-specific
    account_sid: str = "",
    auth_token: str = "",
    from_number: str = "",
    webhook_path: str = "/whatsapp/webhook",
    status_callback_path: str = "/whatsapp/status",
    validate_signatures: bool = True,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = False,
    max_message_length: int = 1600,
    rate_limit_messages_per_minute: int = 30,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    # Deprecated
    config: Optional[WhatsAppConfig] = None,
  ) -> None:
    if config is not None:
      warnings.warn(
        "Passing config= to WhatsAppInterface is deprecated. Pass params directly as keyword arguments.",
        DeprecationWarning,
        stacklevel=2,
      )
      resolved_config = config
    else:
      resolved_config = WhatsAppConfig(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        webhook_path=webhook_path,
        status_callback_path=status_callback_path,
        validate_signatures=validate_signatures,
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
    self._wa_config: WhatsAppConfig = self.config  # type: ignore[assignment]
    self._http_client: Optional[Any] = None

  # --- Router for AgentServer ---

  def create_router(self) -> Any:
    """Create a FastAPI APIRouter with webhook endpoints.

    Returns:
      FastAPI APIRouter instance.
    """
    from fastapi import APIRouter, Request
    from fastapi.responses import PlainTextResponse

    router = APIRouter()

    @router.post(self._wa_config.webhook_path)
    async def whatsapp_webhook(request: Request) -> PlainTextResponse:
      # Validate Twilio signature
      if self._wa_config.validate_signatures:
        if not await self._validate_signature(request):
          return PlainTextResponse("Unauthorized", status_code=403)

      form = await request.form()
      raw_message = dict(form)
      await self.handle_platform_message(raw_message)
      # Twilio expects an empty TwiML response for async replies
      return PlainTextResponse(
        '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
      )

    @router.post(self._wa_config.status_callback_path)
    async def status_callback(request: Request) -> PlainTextResponse:
      form = await request.form()
      log_debug(f"[whatsapp] Status callback: {dict(form)}")
      return PlainTextResponse("OK")

    return router

  # --- BaseInterface implementation ---

  async def _start_receiver(self) -> None:
    try:
      import httpx
    except ImportError:
      raise ImportError("httpx is required for WhatsAppInterface. Install it with: pip install httpx") from None
    self._http_client = httpx.AsyncClient()
    log_info(f"[whatsapp] Receiver started (webhook={self._wa_config.webhook_path})")

  async def _stop_receiver(self) -> None:
    if self._http_client is not None:
      await self._http_client.aclose()
      self._http_client = None
    log_info("[whatsapp] Receiver stopped")

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    body = raw_message.get("Body", "")
    from_number = raw_message.get("From", "")
    to_number = raw_message.get("To", "")

    if not body or not from_number:
      return None

    # Strip "whatsapp:" prefix for user_id
    user_id = from_number.replace("whatsapp:", "")

    return InterfaceMessage(
      text=body,
      platform="whatsapp",
      platform_user_id=user_id,
      platform_chat_id=user_id,  # 1:1 chats use phone as chat ID
      platform_message_id=raw_message.get("MessageSid", ""),
      metadata={
        "from_number": from_number,
        "to_number": to_number,
        "message_sid": raw_message.get("MessageSid", ""),
        "num_media": raw_message.get("NumMedia", "0"),
      },
    )

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    if not response.content:
      return

    from_number = raw_message.get("From", "")
    if not from_number:
      log_error("[whatsapp] Cannot send response: no From number in original message")
      return

    # Split long messages
    chunks = self._split_message(response.content, self._wa_config.max_message_length)
    for chunk in chunks:
      await self._send_message(to=from_number, body=chunk)

  async def _send_message(self, *, to: str, body: str) -> None:
    """Send a message via the Twilio REST API."""
    if self._http_client is None:
      log_error("[whatsapp] HTTP client not initialized")
      return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{self._wa_config.account_sid}/Messages.json"
    data = {
      "From": self._wa_config.from_number,
      "To": to,
      "Body": body,
    }

    try:
      resp = await self._http_client.post(
        url,
        data=data,
        auth=(self._wa_config.account_sid, self._wa_config.auth_token),
      )
      if resp.status_code >= 400:
        log_error(f"[whatsapp] Twilio API error: {resp.status_code} {resp.text}")
      else:
        log_debug(f"[whatsapp] Message sent to {to}")
    except Exception as e:
      log_error(f"[whatsapp] Failed to send message: {e}")

  # --- Signature validation ---

  async def _validate_signature(self, request: Any) -> bool:
    """Validate the Twilio X-Twilio-Signature header."""
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
      log_warning("[whatsapp] Missing X-Twilio-Signature header")
      return False

    # Reconstruct the full URL
    url = str(request.url)
    form = await request.form()
    params = dict(form)

    # Twilio signature validation
    data_str = url + "".join(f"{k}{params[k]}" for k in sorted(params.keys()))
    expected = hmac.new(
      self._wa_config.auth_token.encode(),
      data_str.encode(),
      hashlib.sha1,
    ).digest()

    import base64

    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(signature, expected_b64)

  @staticmethod
  def _split_message(text: str, max_length: int) -> List[str]:
    """Split a message into chunks respecting max length."""
    if len(text) <= max_length:
      return [text]
    chunks: List[str] = []
    while text:
      if len(text) <= max_length:
        chunks.append(text)
        break
      # Find last space within limit
      split_at = text.rfind(" ", 0, max_length)
      if split_at <= 0:
        split_at = max_length
      chunks.append(text[:split_at])
      text = text[split_at:].lstrip()
    return chunks

  def needs_server(self) -> bool:
    """WhatsApp interface requires the HTTP server for webhooks."""
    return True
