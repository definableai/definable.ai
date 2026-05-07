"""WhatsApp interface — unified provider-based agent communication.

Supports both Twilio (managed, webhook-based) and Baileys (self-hosted,
direct WhatsApp Web protocol via Node.js sidecar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional

from definable.agent.interface.base import Interface as BaseInterface
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.agent.interface.whatsapp.config import WhatsAppConfig
from definable.agent.interface.whatsapp.formatting import markdown_to_whatsapp
from definable.agent.interface.whatsapp.normalize import normalize_e164
from definable.agent.interface.whatsapp.provider import (
  ConnectionStatus,
  InboundMessage,
  OutboundMessage,
  QRLoginResult,
  WhatsAppProvider,
)
from definable.utils.log import log_debug, log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver
  from definable.agent.interface.whatsapp.policy import WhatsAppPolicy


class WhatsAppInterface(BaseInterface):
  """Interface connecting an agent to WhatsApp.

  Supports two transport providers:

  - **twilio** (default when ``account_sid`` is provided): Managed
    webhook + REST API. Requires ``account_sid``, ``auth_token``,
    ``from_number``.
  - **baileys**: Self-hosted via Node.js sidecar wrapping the Baileys
    library. Requires ``auth_dir``. Full protocol access (polls,
    reactions, groups, QR login).

  Example (Twilio)::

    interface = WhatsAppInterface(
      provider="twilio",
      account_sid="AC...",
      auth_token="...",
      from_number="whatsapp:+14155238886",
    )

  Example (Baileys)::

    interface = WhatsAppInterface(
      provider="baileys",
      auth_dir="./whatsapp-auth",
      policy=WhatsAppPolicy(dm_policy="allowlist", allow_from=["+15551234567"]),
    )
  """

  def __init__(
    self,
    *,
    # Provider selection
    provider: Literal["twilio", "baileys"] = "twilio",
    # Twilio-specific
    account_sid: str = "",
    auth_token: str = "",
    from_number: str = "",
    validate_signatures: bool = True,
    # Baileys-specific
    auth_dir: str = "./whatsapp-auth",
    node_path: str = "node",
    bridge_port: int = 0,
    reconnect_max_attempts: int = 12,
    heartbeat_seconds: int = 60,
    # Shared WhatsApp
    policy: Optional["WhatsAppPolicy"] = None,
    text_chunk_limit: int = 4000,
    markdown_conversion: bool = True,
    webhook_path: str = "/whatsapp/webhook",
    status_callback_path: str = "/whatsapp/status",
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 3600,
    max_concurrent_requests: int = 10,
    error_message: str = "Sorry, something went wrong. Please try again.",
    typing_indicator: bool = True,
    max_message_length: int = 4000,
    rate_limit_messages_per_minute: int = 30,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    verbose: bool = False,
  ) -> None:
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
    self._policy = policy
    self._text_chunk_limit = text_chunk_limit
    self._markdown_conversion = markdown_conversion
    self._provider_type = provider
    self._verbose = verbose

    # Build provider
    self._provider: WhatsAppProvider = self._build_provider(
      provider=provider,
      account_sid=account_sid,
      auth_token=auth_token,
      from_number=from_number,
      validate_signatures=validate_signatures,
      auth_dir=auth_dir,
      node_path=node_path,
      bridge_port=bridge_port,
      reconnect_max_attempts=reconnect_max_attempts,
      heartbeat_seconds=heartbeat_seconds,
      verbose=verbose,
    )

  @staticmethod
  def _build_provider(
    *,
    provider: str,
    account_sid: str,
    auth_token: str,
    from_number: str,
    validate_signatures: bool,
    auth_dir: str,
    node_path: str,
    bridge_port: int,
    reconnect_max_attempts: int,
    heartbeat_seconds: int,
    verbose: bool,
  ) -> WhatsAppProvider:
    if provider == "twilio":
      from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

      return TwilioProvider(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        validate_signatures=validate_signatures,
      )
    elif provider == "baileys":
      from definable.agent.interface.whatsapp.providers.baileys import BaileysProvider

      return BaileysProvider(
        auth_dir=auth_dir,
        node_path=node_path,
        bridge_port=bridge_port,
        verbose=verbose,
        reconnect_max_attempts=reconnect_max_attempts,
        heartbeat_seconds=heartbeat_seconds,
      )
    else:
      raise ValueError(f"Unknown WhatsApp provider: {provider!r}. Use 'twilio' or 'baileys'.")

  # --- Router for AgentServer ---

  def create_router(self) -> Any:
    """Create a FastAPI APIRouter with webhook endpoints.

    Only relevant for the Twilio provider (webhook-based).
    """
    from fastapi import APIRouter, Request
    from fastapi.responses import PlainTextResponse

    from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

    router = APIRouter()

    @router.post(self._wa_config.webhook_path)
    async def whatsapp_webhook(request: Request) -> PlainTextResponse:
      # Signature validation (Twilio only)
      if isinstance(self._provider, TwilioProvider):
        if not await self._provider.validate_signature(request):
          return PlainTextResponse("Unauthorized", status_code=403)

      form = await request.form()
      raw_message = dict(form)

      # For Twilio: provider converts + fires callback → pipeline
      # For non-webhook providers this is a no-op fallback
      if isinstance(self._provider, TwilioProvider):
        inbound = await self._provider.handle_webhook(raw_message)
        if inbound is not None:
          raw_message["_provider_message"] = inbound  # type: ignore[assignment]

      await self.handle_platform_message(raw_message)
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
    await self._provider.connect(on_message=self._handle_provider_message)
    log_info(f"[whatsapp:{self._provider.provider_name}] Receiver started")

  async def _stop_receiver(self) -> None:
    await self._provider.disconnect()
    log_info(f"[whatsapp:{self._provider.provider_name}] Receiver stopped")

  async def _handle_provider_message(self, msg: InboundMessage) -> None:
    """Bridge from provider InboundMessage to BaseInterface pipeline."""
    # Check sender policy
    if self._policy is not None:
      if not self._policy.check_access(
        from_phone=msg.from_phone,
        chat_jid=msg.chat_jid,
        from_jid=msg.from_jid,
        is_group=msg.is_group,
        is_from_me=msg.is_from_me,
      ):
        return

    raw = dict(msg.raw) if msg.raw else {}
    raw["_provider_message"] = msg
    await self.handle_platform_message(raw)

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    # Check for pre-parsed provider message
    provider_msg: Optional[InboundMessage] = raw_message.get("_provider_message") if isinstance(raw_message, dict) else None

    if provider_msg is not None:
      return InterfaceMessage(
        text=provider_msg.body,
        platform="whatsapp",
        platform_user_id=provider_msg.from_phone,
        platform_chat_id=provider_msg.chat_jid,
        platform_message_id=provider_msg.id,
        username=provider_msg.push_name or None,
        images=provider_msg.images,
        audio=provider_msg.audio,
        videos=provider_msg.videos,
        files=provider_msg.files,
        reply_to_message_id=provider_msg.reply_to_id,
        metadata={
          "from_jid": provider_msg.from_jid,
          "chat_jid": provider_msg.chat_jid,
          "is_group": provider_msg.is_group,
          "is_from_me": provider_msg.is_from_me,
          "was_mentioned": provider_msg.was_mentioned,
          "group_subject": provider_msg.group_subject,
        },
      )

    # Legacy fallback: raw Twilio form data (backwards compat)
    body = raw_message.get("Body", "") if isinstance(raw_message, dict) else ""
    from_number = raw_message.get("From", "") if isinstance(raw_message, dict) else ""

    if not body or not from_number:
      return None

    user_id = from_number.replace("whatsapp:", "")
    normalized = normalize_e164(user_id) or user_id

    return InterfaceMessage(
      text=body,
      platform="whatsapp",
      platform_user_id=normalized,
      platform_chat_id=normalized,
      platform_message_id=raw_message.get("MessageSid", "") if isinstance(raw_message, dict) else "",
      metadata={
        "from_number": from_number,
        "to_number": raw_message.get("To", "") if isinstance(raw_message, dict) else "",
        "message_sid": raw_message.get("MessageSid", "") if isinstance(raw_message, dict) else "",
        "num_media": raw_message.get("NumMedia", "0") if isinstance(raw_message, dict) else "0",
      },
    )

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    # Determine target
    to = original_msg.metadata.get("chat_jid") or original_msg.metadata.get("from_number") or original_msg.platform_user_id

    if not to:
      log_error("[whatsapp] Cannot send response: no target")
      return

    # Composing indicator
    if self.config.typing_indicator:
      await self._provider.send_composing(to)

    # Send text (chunked) with delivery tracking
    if response.content:
      text = response.content
      if self._markdown_conversion:
        text = markdown_to_whatsapp(text)
      chunks = self._split_message(text, self._text_chunk_limit)
      for i, chunk in enumerate(chunks):
        result = await self._provider.send_text(to, chunk)
        if not result.success:
          log_error(f"[whatsapp] Failed to send text chunk {i + 1}/{len(chunks)} to {to}: {result.error}")
          break  # Don't send remaining chunks if one fails

    # Send media with delivery tracking
    if response.images:
      for img in response.images:
        result = await self._provider.send_media(OutboundMessage(to=to, image=img))
        if not result.success:
          log_error(f"[whatsapp] Failed to send image to {to}: {result.error}")
    if response.audio:
      for aud in response.audio:
        result = await self._provider.send_media(OutboundMessage(to=to, audio=aud))
        if not result.success:
          log_error(f"[whatsapp] Failed to send audio to {to}: {result.error}")
    if response.videos:
      for vid in response.videos:
        result = await self._provider.send_media(OutboundMessage(to=to, video=vid))
        if not result.success:
          log_error(f"[whatsapp] Failed to send video to {to}: {result.error}")
    if response.files:
      for f in response.files:
        result = await self._provider.send_media(OutboundMessage(to=to, file=f))
        if not result.success:
          log_error(f"[whatsapp] Failed to send file to {to}: {result.error}")

  # --- Public API ---

  def needs_server(self) -> bool:
    """Whether this interface requires an HTTP server."""
    from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

    return isinstance(self._provider, TwilioProvider)

  async def health(self) -> ConnectionStatus:
    """Get the current provider connection status."""
    return await self._provider.status()

  async def login_qr_start(self, force: bool = False) -> QRLoginResult:
    """Start QR-based login (Baileys only)."""
    return await self._provider.login_qr_start(force=force)

  async def login_qr_wait(self, timeout_ms: int = 60_000) -> QRLoginResult:
    """Wait for QR scan to complete (Baileys only)."""
    return await self._provider.login_qr_wait(timeout_ms=timeout_ms)

  @property
  def provider(self) -> WhatsAppProvider:
    """The underlying transport provider."""
    return self._provider

  # --- Utilities ---

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
      split_at = text.rfind(" ", 0, max_length)
      if split_at <= 0:
        split_at = max_length
      chunks.append(text[:split_at])
      text = text[split_at:].lstrip()
    return chunks
