"""WhatsApp interface — provider-based, supports Baileys (personal) + Twilio.

Two transport providers, both kept:

- **baileys** — self-hosted Node.js sidecar wrapping @whiskeysockets/baileys.
  Connects to your personal WhatsApp via QR-scan. Full protocol access:
  polls, reactions, groups, presence. Requires Node.js; the framework
  auto-`npm install`s the sidecar on first connect.
- **twilio** — managed webhook + REST API. Requires a Twilio account
  with WhatsApp Sandbox or approved Business number.

Usage (Baileys, personal WhatsApp)::

    from definable.agent.interface.whatsapp import WhatsAppInterface
    from definable.agent.interface.whatsapp.policy import WhatsAppPolicy

    iface = WhatsAppInterface(
      agent,
      provider="baileys",
      auth_dir="./whatsapp-auth",
      policy=WhatsAppPolicy(dm_policy="allowlist", allow_from=["+15551234567"]),
    )
    async with iface:
      await iface.serve()
    # First run: scan the QR shown in stdout with WhatsApp on your phone.

Usage (Twilio)::

    iface = WhatsAppInterface(
      agent,
      provider="twilio",
      account_sid="AC...",
      auth_token="...",
      from_number="whatsapp:+14155238886",
    )
    # Twilio additionally needs a webhook server — see create_router().
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from definable.agent.interface.base import Interface
from definable.agent.interface.whatsapp.formatting import markdown_to_whatsapp
from definable.agent.interface.whatsapp.provider import InboundMessage, WhatsAppProvider
from definable.utils.log import log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.whatsapp.policy import WhatsAppPolicy


class WhatsAppInterface(Interface):
  """WhatsApp adapter w/ pluggable transport (Baileys or Twilio)."""

  def __init__(
    self,
    agent: Agent,
    *,
    provider: Literal["twilio", "baileys"] = "baileys",
    # Twilio-specific
    account_sid: str = "",
    auth_token: str = "",
    from_number: str = "",
    validate_signatures: bool = True,
    webhook_path: str = "/whatsapp/webhook",
    status_callback_path: str = "/whatsapp/status",
    # Baileys-specific
    auth_dir: str = "./whatsapp-auth",
    node_path: str = "node",
    bridge_port: int = 0,
    reconnect_max_attempts: int = 12,
    heartbeat_seconds: int = 60,
    verbose: bool = False,
    # Shared
    policy: WhatsAppPolicy | None = None,
    markdown_conversion: bool = True,
    text_chunk_limit: int = 4000,
  ) -> None:
    super().__init__(agent)
    self._provider_kind = provider
    self._policy = policy
    self._markdown_conversion = markdown_conversion
    self._text_chunk_limit = text_chunk_limit
    self.webhook_path = webhook_path
    self.status_callback_path = status_callback_path

    if provider == "baileys":
      from definable.agent.interface.whatsapp.providers.baileys import BaileysProvider

      self._provider: WhatsAppProvider = BaileysProvider(
        auth_dir=auth_dir,
        node_path=node_path,
        bridge_port=bridge_port,
        verbose=verbose,
        reconnect_max_attempts=reconnect_max_attempts,
        heartbeat_seconds=heartbeat_seconds,
      )
    elif provider == "twilio":
      from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

      self._provider = TwilioProvider(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        validate_signatures=validate_signatures,
      )
    else:
      raise ValueError(f"Unknown WhatsApp provider: {provider!r}. Use 'baileys' or 'twilio'.")

    self._stopped = asyncio.Event()

  # ---- Interface contract -------------------------------------------------

  async def aopen(self) -> None:
    await self._provider.connect(on_message=self._on_inbound)
    log_info(f"[whatsapp] connected via {self._provider_kind}")

  async def aclose(self) -> None:
    self._stopped.set()
    await self._provider.disconnect()
    log_info("[whatsapp] disconnected")

  async def serve(self) -> None:
    """Block until aclose is called. Provider drives messages via callback."""
    self._stopped.clear()
    await self._stopped.wait()

  async def _convert(self, raw_message: Any) -> str:
    msg: InboundMessage = raw_message
    return msg.body or ""

  async def _send(self, raw_message: Any, reply: str) -> None:
    msg: InboundMessage = raw_message
    body = markdown_to_whatsapp(reply) if self._markdown_conversion else reply
    if self._text_chunk_limit and len(body) > self._text_chunk_limit:
      body = body[: self._text_chunk_limit - 3] + "..."
    try:
      await self._provider.send_text(to=msg.chat_jid, body=body)
    except Exception as e:
      log_error(f"[whatsapp] send failed: {e}")

  # ---- Provider callback --------------------------------------------------

  async def _on_inbound(self, msg: InboundMessage) -> None:
    """Provider hands us each InboundMessage. Apply policy then dispatch."""
    if msg.is_from_me:
      return
    if self._policy is not None:
      allowed = self._policy.check_access(
        from_phone=msg.from_phone,
        chat_jid=msg.chat_jid,
        from_jid=msg.from_jid,
        is_group=msg.is_group,
        is_from_me=msg.is_from_me,
      )
      if not allowed:
        return
    await self.handle(msg)

  # ---- Twilio webhook helper ---------------------------------------------

  def create_router(self) -> Any:
    """FastAPI router exposing the Twilio webhook endpoints.

    Only meaningful for `provider="twilio"`. Mount on your own FastAPI app::

        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(iface.create_router())
    """
    if self._provider_kind != "twilio":
      raise RuntimeError("create_router is only meaningful for the twilio provider")

    from fastapi import APIRouter, Request
    from fastapi.responses import PlainTextResponse

    from definable.agent.interface.whatsapp.providers.twilio import TwilioProvider

    router = APIRouter()
    twilio: TwilioProvider = self._provider  # type: ignore[assignment]

    @router.post(self.webhook_path)
    async def webhook(request: Request) -> PlainTextResponse:
      if not await twilio.validate_signature(request):
        return PlainTextResponse("Unauthorized", status_code=403)
      form = await request.form()
      raw = dict(form)
      inbound = await twilio.handle_webhook(raw)
      if inbound is not None:
        await self._on_inbound(inbound)
      return PlainTextResponse(
        '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
      )

    @router.post(self.status_callback_path)
    async def status_callback(request: Request) -> PlainTextResponse:
      del request
      return PlainTextResponse("")

    return router
