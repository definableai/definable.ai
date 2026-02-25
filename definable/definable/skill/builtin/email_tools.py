"""Email skill — send emails via SMTP."""

from __future__ import annotations

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from definable.skill.base import Skill
from definable.tool.decorator import tool


class EmailTools(Skill):
  """Send emails via SMTP. Supports Gmail, Outlook, and custom SMTP servers.

  For Gmail, use an App Password (not your regular password).

  Args:
      smtp_host: SMTP server hostname. Default "smtp.gmail.com".
      smtp_port: SMTP port. Default 587 (TLS).
      sender_email: Sender email address. Falls back to SMTP_EMAIL env var.
      sender_password: Sender password/app password. Falls back to SMTP_PASSWORD env var.
      sender_name: Display name for sender. Default None.
      use_tls: Enable TLS. Default True.

  Example::

      from definable.skill.builtin import EmailTools
      agent = Agent(model=model, skills=[EmailTools(
          sender_email="me@gmail.com",
          sender_password="app-password-here",
      )])
  """

  name = "email_tools"
  instructions = (
    "You have access to email tools. Use send_email to send messages. "
    "Always confirm with the user before sending emails. "
    "Provide a clear subject and well-formatted body."
  )

  def __init__(
    self,
    *,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    sender_email: Optional[str] = None,
    sender_password: Optional[str] = None,
    sender_name: Optional[str] = None,
    use_tls: bool = True,
  ):
    super().__init__()
    self._smtp_host = smtp_host
    self._smtp_port = smtp_port
    self._sender_email = sender_email or os.getenv("SMTP_EMAIL")
    self._sender_password = sender_password or os.getenv("SMTP_PASSWORD")
    self._sender_name = sender_name
    self._use_tls = use_tls

  def _send(self, to: str, subject: str, body: str, cc: str = "", html: bool = False) -> dict:
    if not self._sender_email or not self._sender_password:
      return {"error": "SMTP credentials not configured. Set sender_email/sender_password or SMTP_EMAIL/SMTP_PASSWORD env vars."}

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{self._sender_name} <{self._sender_email}>" if self._sender_name else self._sender_email
    msg["To"] = to
    if cc:
      msg["Cc"] = cc

    content_type = "html" if html else "plain"
    msg.attach(MIMEText(body, content_type, "utf-8"))

    recipients = [to] + ([addr.strip() for addr in cc.split(",") if addr.strip()] if cc else [])

    with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
      if self._use_tls:
        server.starttls()
      server.login(self._sender_email, self._sender_password)
      server.sendmail(self._sender_email, recipients, msg.as_string())

    return {"ok": True, "to": to, "subject": subject}

  @property
  def tools(self) -> list:
    skill = self

    @tool
    def send_email(to: str, subject: str, body: str, cc: str = "") -> str:
      """Send a plain-text email. CC is optional (comma-separated addresses)."""
      try:
        result = skill._send(to=to, subject=subject, body=body, cc=cc)
        return json.dumps(result)
      except Exception as e:
        return json.dumps({"error": str(e)})

    @tool
    def send_html_email(to: str, subject: str, html_body: str, cc: str = "") -> str:
      """Send an HTML-formatted email. CC is optional."""
      try:
        result = skill._send(to=to, subject=subject, body=html_body, cc=cc, html=True)
        return json.dumps(result)
      except Exception as e:
        return json.dumps({"error": str(e)})

    return [send_email, send_html_email]
