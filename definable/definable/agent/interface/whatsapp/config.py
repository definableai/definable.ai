"""Configuration for WhatsApp interface."""

from dataclasses import dataclass

from definable.agent.interface.config import InterfaceConfig


@dataclass(frozen=True)
class WhatsAppConfig(InterfaceConfig):
  """Configuration for the WhatsApp interface (Twilio provider).

  Attributes:
    account_sid: Twilio account SID.
    auth_token: Twilio auth token.
    from_number: WhatsApp sender number (format: "whatsapp:+14155238886").
    webhook_path: Path for incoming message webhooks.
    status_callback_path: Path for message status callbacks.
    validate_signatures: Validate Twilio request signatures.
  """

  platform: str = "whatsapp"
  account_sid: str = ""
  auth_token: str = ""
  from_number: str = ""
  webhook_path: str = "/whatsapp/webhook"
  status_callback_path: str = "/whatsapp/status"
  validate_signatures: bool = True
  max_message_length: int = 1600
