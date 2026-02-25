"""Call interface configuration."""

from dataclasses import dataclass
from typing import Literal, Optional

from definable.agent.interface.config import InterfaceConfig
from definable.agent.interface.errors import InterfaceError


@dataclass(frozen=True)
class CallConfig(InterfaceConfig):
  """Configuration for the call/voice interface.

  Extends InterfaceConfig with telephony and voice pipeline settings.

  Attributes:
    telephony_provider: Telephony provider name ("twilio" or "plivo").
    pipeline_mode: Voice pipeline strategy ("managed", "cascading", or "realtime").
    phone_number: The phone number to receive/make calls on.
    webhook_path: URL path for incoming call webhooks.
    stream_path: URL path for WebSocket audio/text streams.
    welcome_message: Greeting spoken when a call connects.
    max_call_duration_seconds: Maximum call duration before automatic hangup.
    language: BCP-47 language code for STT/TTS.
    voice: Voice name/ID for TTS.
    interruptible: When the caller can interrupt the agent.
    interrupt_sensitivity: How sensitive barge-in detection is.
    stt_provider: STT provider name for managed mode.
    tts_provider: TTS provider name for managed mode.
  """

  platform: str = "call"

  # Telephony
  telephony_provider: str = "twilio"

  # Pipeline
  pipeline_mode: Literal["managed", "cascading", "realtime"] = "managed"

  # Call settings
  phone_number: str = ""
  webhook_path: str = "/call/incoming"
  stream_path: str = "/call/stream"
  welcome_message: Optional[str] = None
  max_call_duration_seconds: int = 3600

  # Voice settings
  language: str = "en-US"
  voice: str = "en-US-Standard-A"
  interruptible: Literal["none", "dtmf", "speech", "any"] = "any"
  interrupt_sensitivity: Literal["low", "medium", "high"] = "medium"

  # Managed mode — provider names for ConversationRelay
  stt_provider: str = "deepgram"
  tts_provider: str = "google"

  # Override defaults for call-specific behavior
  session_ttl_seconds: int = 7200  # calls can be long
  max_concurrent_requests: int = 50  # calls are concurrent by nature
  error_message: str = "Sorry, I encountered an error. Please try again."

  def __post_init__(self) -> None:
    if not self.phone_number:
      raise InterfaceError("phone_number is required for CallConfig", platform="call")
    if self.telephony_provider not in ("twilio", "plivo"):
      raise InterfaceError(
        f"Unsupported telephony provider: {self.telephony_provider!r}. Use 'twilio' or 'plivo'.",
        platform="call",
      )
    if self.pipeline_mode not in ("managed", "cascading", "realtime"):
      raise InterfaceError(
        f"Unsupported pipeline mode: {self.pipeline_mode!r}. Use 'managed', 'cascading', or 'realtime'.",
        platform="call",
      )
