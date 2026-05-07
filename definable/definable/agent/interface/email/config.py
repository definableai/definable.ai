"""Configuration for Email interface."""

from dataclasses import dataclass

from definable.agent.interface.config import InterfaceConfig


@dataclass(frozen=True)
class EmailConfig(InterfaceConfig):
  """Configuration for the Email interface.

  Attributes:
    imap_host: IMAP server hostname.
    imap_port: IMAP server port (993 for SSL).
    smtp_host: SMTP server hostname.
    smtp_port: SMTP server port (587 for STARTTLS).
    email_address: Email address for login and sending.
    email_password: Email password or app-specific password.
    imap_folder: IMAP folder to monitor (default: "INBOX").
    poll_interval: Seconds between IMAP polls (default: 30).
    mark_as_read: Mark processed emails as read (default: True).
    subject_prefix: Prefix for reply subject lines (default: "Re: ").
    reply_quote_original: Include original message in replies (default: True).
  """

  platform: str = "email"
  imap_host: str = ""
  imap_port: int = 993
  smtp_host: str = ""
  smtp_port: int = 587
  email_address: str = ""
  email_password: str = ""
  imap_folder: str = "INBOX"
  poll_interval: float = 30.0
  mark_as_read: bool = True
  subject_prefix: str = "Re: "
  reply_quote_original: bool = True
  max_message_length: int = 50000
