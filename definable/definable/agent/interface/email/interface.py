"""Email interface — IMAP polling + SMTP sending.

Polls an IMAP mailbox for unread mail, runs each through `agent.arun()`,
and replies via SMTP with In-Reply-To / References headers so threads
hold across exchanges.

Usage::

    iface = EmailInterface(
      agent,
      imap_host="imap.gmail.com", smtp_host="smtp.gmail.com",
      email_address="agent@example.com",
      email_password="<app-specific-password>",
    )
    async with iface:
      await iface.serve()
"""

from __future__ import annotations

import asyncio
import contextlib
import email
import email.mime.multipart
import email.mime.text
import email.utils
import imaplib
import smtplib
from typing import TYPE_CHECKING, Any

from definable.agent.interface.base import Interface
from definable.utils.log import log_debug, log_error, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class EmailInterface(Interface):
  """IMAP poll + SMTP send. Each unread email is one agent invocation."""

  def __init__(
    self,
    agent: Agent,
    *,
    imap_host: str,
    imap_port: int = 993,
    smtp_host: str,
    smtp_port: int = 587,
    email_address: str,
    email_password: str,
    imap_folder: str = "INBOX",
    poll_interval: float = 30.0,
    mark_as_read: bool = True,
    subject_prefix: str = "Re: ",
    reply_quote_original: bool = True,
  ) -> None:
    super().__init__(agent)
    self.imap_host = imap_host
    self.imap_port = imap_port
    self.smtp_host = smtp_host
    self.smtp_port = smtp_port
    self.email_address = email_address
    self.email_password = email_password
    self.imap_folder = imap_folder
    self.poll_interval = poll_interval
    self.mark_as_read = mark_as_read
    self.subject_prefix = subject_prefix
    self.reply_quote_original = reply_quote_original

    self._poll_task: asyncio.Task[Any] | None = None
    self._imap: imaplib.IMAP4_SSL | None = None

  # ---- Interface contract -------------------------------------------------

  async def aopen(self) -> None:
    if not self.imap_host or not self.email_address:
      raise ValueError("imap_host and email_address are required")
    self._poll_task = asyncio.create_task(self._poll_loop())
    log_info(f"[email] polling {self.imap_folder} every {self.poll_interval}s")

  async def aclose(self) -> None:
    if self._poll_task is not None:
      self._poll_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._poll_task
      self._poll_task = None
    self._close_imap()
    log_info("[email] stopped")

  async def _convert(self, raw_message: Any) -> str:
    msg: email.message.Message = raw_message["email_message"]
    return self._extract_body(msg)

  async def _send(self, raw_message: Any, reply: str) -> None:
    msg: email.message.Message = raw_message["email_message"]
    sender = email.utils.parseaddr(msg.get("From", ""))[1]
    if not sender:
      return
    subject = msg.get("Subject", "")
    if not subject.startswith(self.subject_prefix.strip()):
      subject = f"{self.subject_prefix}{subject}"
    in_reply_to = msg.get("Message-ID", "")
    refs = msg.get("References", "")
    if in_reply_to:
      refs = f"{refs} {in_reply_to}".strip() if refs else in_reply_to

    body = reply
    if self.reply_quote_original:
      original = self._extract_body(msg)
      if original:
        quoted = "\n".join(f"> {line}" for line in original.splitlines())
        body = f"{reply}\n\n---\n{quoted}"

    await asyncio.get_event_loop().run_in_executor(
      None,
      lambda: self._send_email_sync(
        to=sender,
        subject=subject,
        body=body,
        in_reply_to=in_reply_to,
        references=refs,
      ),
    )

  # ---- IMAP polling -------------------------------------------------------

  async def _poll_loop(self) -> None:
    try:
      while True:
        try:
          new_messages = await asyncio.get_event_loop().run_in_executor(None, self._fetch_unread)
          for raw in new_messages:
            await self.handle(raw)
        except Exception as e:
          log_error(f"[email] poll error: {e}")
          self._close_imap()
        await asyncio.sleep(self.poll_interval)
    except asyncio.CancelledError:
      pass

  def _fetch_unread(self) -> list[dict[str, Any]]:
    try:
      if self._imap is None:
        self._imap = imaplib.IMAP4_SSL(self.imap_host, self.imap_port)
        self._imap.login(self.email_address, self.email_password)
      self._imap.select(self.imap_folder)
      _, message_ids = self._imap.search(None, "UNSEEN")
      results: list[dict[str, Any]] = []
      for uid in message_ids[0].split():
        if not uid:
          continue
        _, data = self._imap.fetch(uid, "(RFC822)")
        if data[0] is None:
          continue
        raw_email = data[0][1]  # type: ignore[index]
        if not isinstance(raw_email, bytes):
          continue
        msg = email.message_from_bytes(raw_email)
        results.append({"uid": uid.decode(), "email_message": msg})
        if self.mark_as_read:
          self._imap.store(uid, "+FLAGS", "\\Seen")
      return results
    except Exception as e:
      log_error(f"[email] IMAP fetch error: {e}")
      self._close_imap()
      return []

  def _close_imap(self) -> None:
    if self._imap is not None:
      with contextlib.suppress(Exception):
        self._imap.close()
        self._imap.logout()
      self._imap = None

  # ---- SMTP sending -------------------------------------------------------

  def _send_email_sync(self, *, to: str, subject: str, body: str, in_reply_to: str = "", references: str = "") -> None:
    msg = email.mime.multipart.MIMEMultipart("alternative")
    msg["From"] = self.email_address
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = email.utils.make_msgid()
    if in_reply_to:
      msg["In-Reply-To"] = in_reply_to
    if references:
      msg["References"] = references
    msg.attach(email.mime.text.MIMEText(body, "plain"))
    try:
      with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
        server.starttls()
        server.login(self.email_address, self.email_password)
        server.send_message(msg)
      log_debug(f"[email] sent reply to {to}")
    except Exception as e:
      log_error(f"[email] SMTP send error: {e}")

  @staticmethod
  def _extract_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
      for part in msg.walk():
        if part.get_content_type() == "text/plain":
          payload = part.get_payload(decode=True)
          if isinstance(payload, bytes):
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
      return ""
    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
      charset = msg.get_content_charset() or "utf-8"
      return payload.decode(charset, errors="replace")
    return ""
