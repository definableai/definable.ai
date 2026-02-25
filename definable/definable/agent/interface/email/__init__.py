"""Email interface — connect agents to email via IMAP/SMTP."""

from definable.agent.interface.email.config import EmailConfig
from definable.agent.interface.email.interface import EmailInterface

__all__ = [
  "EmailConfig",
  "EmailInterface",
]
