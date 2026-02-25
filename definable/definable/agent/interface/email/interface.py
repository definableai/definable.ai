"""Email interface — IMAP polling + SMTP sending for agent communication."""

from __future__ import annotations

import asyncio
import email
import email.mime.multipart
import email.mime.text
import email.utils
import imaplib
import smtplib
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from definable.agent.interface.base import BaseInterface
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.agent.interface.email.config import EmailConfig
from definable.utils.log import log_debug, log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.identity import IdentityResolver


class EmailInterface(BaseInterface):
  """Interface connecting an agent to email via IMAP/SMTP.

  Polls an IMAP mailbox for new messages, processes them through
  the agent, and sends replies via SMTP. Uses threading for I/O
  (IMAP/SMTP are blocking protocols).

  Thread tracking: Uses In-Reply-To and References headers to
  maintain conversation threads across email exchanges.

  Example::

    interface = EmailInterface(
      agent=agent,
      imap_host="imap.gmail.com",
      smtp_host="smtp.gmail.com",
      email_address="agent@example.com",
      email_password="app-specific-password",
    )
    async with interface:
      await interface.serve_forever()
  """

  def __init__(
    self,
    *,
    # Email-specific
    imap_host: str = "",
    imap_port: int = 993,
    smtp_host: str = "",
    smtp_port: int = 587,
    email_address: str = "",
    email_password: str = "",
    imap_folder: str = "INBOX",
    poll_interval: float = 30.0,
    mark_as_read: bool = True,
    subject_prefix: str = "Re: ",
    reply_quote_original: bool = True,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 86400,
    max_concurrent_requests: int = 5,
    error_message: str = "Sorry, something went wrong processing your email. Please try again.",
    typing_indicator: bool = False,
    max_message_length: int = 50000,
    rate_limit_messages_per_minute: int = 10,
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
    # Deprecated
    config: Optional[EmailConfig] = None,
  ) -> None:
    if config is not None:
      warnings.warn(
        "Passing config= to EmailInterface is deprecated. Pass params directly as keyword arguments.",
        DeprecationWarning,
        stacklevel=2,
      )
      resolved_config = config
    else:
      resolved_config = EmailConfig(
        imap_host=imap_host,
        imap_port=imap_port,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        email_address=email_address,
        email_password=email_password,
        imap_folder=imap_folder,
        poll_interval=poll_interval,
        mark_as_read=mark_as_read,
        subject_prefix=subject_prefix,
        reply_quote_original=reply_quote_original,
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
    self._email_config: EmailConfig = self.config  # type: ignore[assignment]
    self._poll_task: Optional[asyncio.Task] = None
    self._imap: Optional[imaplib.IMAP4_SSL] = None

  # --- BaseInterface implementation ---

  async def _start_receiver(self) -> None:
    if not self._email_config.imap_host:
      raise ValueError("imap_host is required for EmailInterface")
    if not self._email_config.email_address:
      raise ValueError("email_address is required for EmailInterface")

    self._poll_task = asyncio.create_task(self._poll_loop())
    log_info(f"[email] Receiver started (polling {self._email_config.imap_folder} every {self._email_config.poll_interval}s)")

  async def _stop_receiver(self) -> None:
    if self._poll_task is not None:
      self._poll_task.cancel()
      import contextlib

      with contextlib.suppress(asyncio.CancelledError):
        await self._poll_task
      self._poll_task = None
    self._close_imap()
    log_info("[email] Receiver stopped")

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    msg: email.message.Message = raw_message["email_message"]
    uid: str = raw_message["uid"]

    # Extract sender
    from_header = msg.get("From", "")
    sender_email = email.utils.parseaddr(from_header)[1]
    if not sender_email:
      return None

    # Extract text body
    body = self._extract_body(msg)
    if not body:
      return None

    subject = msg.get("Subject", "")
    message_id = msg.get("Message-ID", "")
    in_reply_to = msg.get("In-Reply-To", "")
    references = msg.get("References", "")

    return InterfaceMessage(
      text=body,
      platform="email",
      platform_user_id=sender_email,
      platform_chat_id=sender_email,  # Thread by sender
      platform_message_id=message_id,
      metadata={
        "uid": uid,
        "subject": subject,
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "references": references,
        "from": from_header,
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

    to_email = original_msg.platform_user_id
    subject = original_msg.metadata.get("subject", "")
    if not subject.startswith(self._email_config.subject_prefix.strip()):
      subject = f"{self._email_config.subject_prefix}{subject}"

    original_message_id = original_msg.metadata.get("message_id", "")
    references = original_msg.metadata.get("references", "")
    if original_message_id:
      if references:
        references = f"{references} {original_message_id}"
      else:
        references = original_message_id

    body = response.content
    if self._email_config.reply_quote_original and original_msg.text:
      quoted = "\n".join(f"> {line}" for line in original_msg.text.splitlines())
      body = f"{response.content}\n\n---\n{quoted}"

    await self._send_email(
      to=to_email,
      subject=subject,
      body=body,
      in_reply_to=original_message_id,
      references=references,
    )

  # --- IMAP polling ---

  async def _poll_loop(self) -> None:
    """Poll IMAP for new messages."""
    try:
      while True:
        try:
          new_messages = await asyncio.get_event_loop().run_in_executor(None, self._fetch_new_emails)
          for raw_msg in new_messages:
            await self.handle_platform_message(raw_msg)
        except Exception as e:
          log_error(f"[email] Poll error: {e}")
          self._close_imap()

        await asyncio.sleep(self._email_config.poll_interval)
    except asyncio.CancelledError:
      pass

  def _fetch_new_emails(self) -> List[Dict[str, Any]]:
    """Fetch unread emails from IMAP (runs in thread)."""
    try:
      if self._imap is None:
        self._imap = imaplib.IMAP4_SSL(
          self._email_config.imap_host,
          self._email_config.imap_port,
        )
        self._imap.login(self._email_config.email_address, self._email_config.email_password)

      self._imap.select(self._email_config.imap_folder)
      _, message_ids = self._imap.search(None, "UNSEEN")

      results: List[Dict[str, Any]] = []
      for uid in message_ids[0].split():
        if not uid:
          continue
        _, data = self._imap.fetch(uid, "(RFC822)")
        if data[0] is None:
          continue
        raw_email = data[0][1]  # type: ignore[index]
        msg = email.message_from_bytes(raw_email)  # type: ignore[arg-type]

        results.append({
          "uid": uid.decode(),
          "email_message": msg,
        })

        if self._email_config.mark_as_read:
          self._imap.store(uid, "+FLAGS", "\\Seen")

      return results

    except Exception as e:
      log_error(f"[email] IMAP fetch error: {e}")
      self._close_imap()
      return []

  def _close_imap(self) -> None:
    """Close the IMAP connection safely."""
    if self._imap is not None:
      try:
        self._imap.close()
        self._imap.logout()
      except Exception:
        pass
      self._imap = None

  # --- SMTP sending ---

  async def _send_email(
    self,
    *,
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
    references: str = "",
  ) -> None:
    """Send an email via SMTP (runs in thread)."""
    await asyncio.get_event_loop().run_in_executor(
      None,
      lambda: self._send_email_sync(
        to=to,
        subject=subject,
        body=body,
        in_reply_to=in_reply_to,
        references=references,
      ),
    )

  def _send_email_sync(
    self,
    *,
    to: str,
    subject: str,
    body: str,
    in_reply_to: str = "",
    references: str = "",
  ) -> None:
    """Send an email via SMTP (blocking, runs in executor)."""
    msg = email.mime.multipart.MIMEMultipart("alternative")
    msg["From"] = self._email_config.email_address
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = email.utils.make_msgid()

    if in_reply_to:
      msg["In-Reply-To"] = in_reply_to
    if references:
      msg["References"] = references

    msg.attach(email.mime.text.MIMEText(body, "plain"))

    try:
      with smtplib.SMTP(self._email_config.smtp_host, self._email_config.smtp_port) as server:
        server.starttls()
        server.login(self._email_config.email_address, self._email_config.email_password)
        server.send_message(msg)
      log_debug(f"[email] Sent reply to {to}")
    except Exception as e:
      log_error(f"[email] SMTP send error: {e}")

  # --- Helpers ---

  @staticmethod
  def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message."""
    if msg.is_multipart():
      for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
          payload = part.get_payload(decode=True)
          if isinstance(payload, bytes):
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
      return ""
    else:
      payload = msg.get_payload(decode=True)
      if isinstance(payload, bytes):
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
      return ""
