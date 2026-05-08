"""Twilio WhatsApp Business API provider."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Optional

from definable.agent.interface.whatsapp.normalize import normalize_e164, redact_phone
from definable.agent.interface.whatsapp.provider import (
  ConnectionStatus,
  InboundMessage,
  MessageCallback,
  OutboundMessage,
  PollMessage,
  ReactionMessage,
  SendResult,
  WhatsAppProvider,
)
from definable.utils.log import log_debug, log_error, log_info, log_warning


_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


class TwilioProvider(WhatsAppProvider):
  """Twilio WhatsApp Business API provider.

  Webhook-based inbound, REST API outbound.
  Supports: text, media (via MediaUrl).
  Does NOT support: polls, reactions, groups, QR login.

  Args:
    account_sid: Twilio account SID.
    auth_token: Twilio auth token.
    from_number: WhatsApp sender number (e.g. ``"whatsapp:+14155238886"``).
    validate_signatures: Validate ``X-Twilio-Signature`` on webhooks.
  """

  def __init__(
    self,
    *,
    account_sid: str,
    auth_token: str,
    from_number: str,
    validate_signatures: bool = True,
  ) -> None:
    self._account_sid = account_sid
    self._auth_token = auth_token
    self._from_number = from_number
    self._validate_signatures = validate_signatures
    self._on_message: Optional[MessageCallback] = None
    self._http_client: Optional[Any] = None
    self._connected = False
    self._send_count = 0
    self._error_count = 0
    self._last_error: Optional[str] = None

  # --- Provider protocol ---

  async def connect(self, on_message: MessageCallback) -> None:
    try:
      import httpx
    except ImportError:
      raise ImportError("httpx is required for TwilioProvider. Install: pip install httpx") from None
    self._on_message = on_message
    self._http_client = httpx.AsyncClient()
    self._connected = True
    log_info("[whatsapp:twilio] Provider connected")

  async def disconnect(self) -> None:
    if self._http_client is not None:
      await self._http_client.aclose()
      self._http_client = None
    self._connected = False
    self._on_message = None
    log_info("[whatsapp:twilio] Provider disconnected")

  async def send_text(self, to: str, body: str) -> SendResult:
    return await self._send(to=to, body=body)

  async def send_media(self, msg: OutboundMessage) -> SendResult:
    media_url: Optional[str] = None
    if msg.image and msg.image.url:
      media_url = msg.image.url
    elif msg.audio and msg.audio.url:
      media_url = msg.audio.url
    elif msg.video and msg.video.url:
      media_url = msg.video.url
    elif msg.file and msg.file.url:
      media_url = msg.file.url

    if media_url is None:
      return SendResult(success=False, error="Twilio requires a media URL (content/filepath not supported)")

    return await self._send(to=msg.to, body=msg.body, media_url=media_url)

  async def send_poll(self, poll: PollMessage) -> SendResult:
    return SendResult(success=False, error="Twilio does not support polls")

  async def send_reaction(self, reaction: ReactionMessage) -> SendResult:
    return SendResult(success=False, error="Twilio does not support reactions")

  async def send_composing(self, to: str) -> None:
    pass  # Twilio doesn't expose a composing API for WhatsApp

  async def status(self) -> ConnectionStatus:
    return ConnectionStatus(
      connected=self._connected,
      running=self._connected,
      linked=bool(self._account_sid and self._auth_token),
      last_error=self._last_error,
    )

  @property
  def supports_media(self) -> bool:
    return True

  @property
  def provider_name(self) -> str:
    return "twilio"

  # --- Webhook integration (called by WhatsAppInterface router) ---

  async def handle_webhook(self, form_data: dict[str, Any]) -> Optional[InboundMessage]:
    """Convert a Twilio webhook form payload to an InboundMessage.

    Returns ``None`` if the payload is missing required fields.
    """
    body = str(form_data.get("Body", ""))
    from_number = str(form_data.get("From", ""))

    if not body or not from_number:
      return None

    # Strip "whatsapp:" prefix
    from_phone = from_number.replace("whatsapp:", "")
    normalized = normalize_e164(from_phone) or from_phone

    msg = InboundMessage(
      id=str(form_data.get("MessageSid", "")),
      from_phone=normalized,
      from_jid=f"{normalized}@s.whatsapp.net",
      chat_jid=f"{normalized}@s.whatsapp.net",
      body=body,
      push_name="",
      is_group=False,
      is_from_me=False,
      raw=form_data,
    )

    # Fire callback
    if self._on_message is not None:
      import asyncio

      asyncio.create_task(self._on_message(msg))

    return msg

  async def validate_signature(self, request: Any) -> bool:
    """Validate a Twilio ``X-Twilio-Signature`` header.

    Returns ``True`` if valid or if signature validation is disabled.
    """
    if not self._validate_signatures:
      return True

    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
      log_warning("[whatsapp:twilio] Missing X-Twilio-Signature header")
      return False

    url = str(request.url)
    form = await request.form()
    params = dict(form)

    data_str = url + "".join(f"{k}{params[k]}" for k in sorted(params.keys()))
    expected = hmac.new(
      self._auth_token.encode(),
      data_str.encode(),
      hashlib.sha1,
    ).digest()

    import base64

    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(signature, expected_b64)

  # --- Internal ---

  async def _send(self, *, to: str, body: str, media_url: Optional[str] = None) -> SendResult:
    """Send a message via the Twilio REST API with retry on transient errors."""
    if self._http_client is None:
      return SendResult(success=False, error="HTTP client not initialized")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{self._account_sid}/Messages.json"
    data: dict[str, str] = {
      "From": self._from_number,
      "To": to if to.startswith("whatsapp:") else f"whatsapp:{to}",
      "Body": body,
    }
    if media_url:
      data["MediaUrl"] = media_url

    import asyncio

    last_error: Optional[str] = None
    for attempt in range(_MAX_RETRIES):
      try:
        resp = await self._http_client.post(
          url,
          data=data,
          auth=(self._account_sid, self._auth_token),
        )
        if resp.status_code < 400:
          self._send_count += 1
          log_debug(f"[whatsapp:twilio] Message sent to {redact_phone(to)}")
          # Extract SID from response
          try:
            result_data = resp.json()
            msg_sid = result_data.get("sid", "")
          except Exception:
            msg_sid = ""
          return SendResult(success=True, message_id=msg_sid)

        # 429 or 5xx → retry
        if resp.status_code == 429 or resp.status_code >= 500:
          last_error = f"HTTP {resp.status_code}"
          if attempt < _MAX_RETRIES - 1:
            await asyncio.sleep(_RETRY_BACKOFF[attempt])
            continue

        # 4xx (non-429) → permanent failure
        error_msg = f"Twilio API error: {resp.status_code} {resp.text[:200]}"
        log_error(f"[whatsapp:twilio] {error_msg}")
        self._error_count += 1
        self._last_error = error_msg
        return SendResult(success=False, error=error_msg)

      except Exception as e:
        last_error = str(e)
        if attempt < _MAX_RETRIES - 1:
          await asyncio.sleep(_RETRY_BACKOFF[attempt])
          continue
        log_error(f"[whatsapp:twilio] Failed to send message: {e}")
        self._error_count += 1
        self._last_error = last_error
        return SendResult(success=False, error=last_error)

    self._error_count += 1
    self._last_error = last_error
    return SendResult(success=False, error=last_error or "Max retries exceeded")
