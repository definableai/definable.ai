"""Call lifecycle types — state, session, and events."""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional

from definable.agent.interface.session import InterfaceSession


class CallState(Enum):
  """Lifecycle state of a voice call."""

  RINGING = "ringing"
  ACTIVE = "active"
  ON_HOLD = "on_hold"
  ENDED = "ended"


class CallEventType(Enum):
  """Types of call lifecycle events."""

  CALL_STARTED = "call_started"
  CALL_ENDED = "call_ended"
  UTTERANCE = "utterance"
  INTERRUPTION = "interruption"
  SILENCE = "silence"
  DTMF = "dtmf"
  ERROR = "error"


@dataclass
class CallEvent:
  """A single event during a call's lifecycle.

  Attributes:
    type: The event type.
    call_id: Provider call ID.
    timestamp: Unix timestamp of the event.
    data: Event-specific payload.
  """

  type: CallEventType
  call_id: str
  timestamp: float = field(default_factory=time)
  data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallSession:
  """Represents an active voice call and its state.

  Bridges the telephony layer (call_id, stream_id, phone number)
  with the interface layer (InterfaceSession for conversation history).

  Attributes:
    call_id: Provider-assigned call identifier (e.g. Twilio CallSid).
    stream_id: WebSocket stream identifier (e.g. Twilio StreamSid).
    from_number: Caller's phone number.
    to_number: Called phone number (our number).
    state: Current call lifecycle state.
    started_at: Unix timestamp when the call connected.
    interface_session: The underlying InterfaceSession for conversation history.
    conversation: Text conversation history for the LLM.
    events: Ordered list of call events.
    metadata: Provider-specific metadata.
  """

  call_id: str
  stream_id: str = ""
  from_number: str = ""
  to_number: str = ""
  state: CallState = CallState.RINGING
  started_at: float = field(default_factory=time)
  interface_session: Optional[InterfaceSession] = None
  conversation: List[Dict[str, str]] = field(default_factory=list)
  events: List[CallEvent] = field(default_factory=list)
  metadata: Dict[str, Any] = field(default_factory=dict)

  def add_event(self, event_type: CallEventType, **data: Any) -> CallEvent:
    """Record a call event.

    Args:
      event_type: The event type.
      **data: Event payload fields.

    Returns:
      The created CallEvent.
    """
    event = CallEvent(type=event_type, call_id=self.call_id, data=data)
    self.events.append(event)
    return event

  def add_user_message(self, text: str) -> None:
    """Add a user utterance to conversation history.

    Args:
      text: The transcribed user speech.
    """
    self.conversation.append({"role": "user", "content": text})

  def add_assistant_message(self, text: str) -> None:
    """Add an assistant response to conversation history.

    Args:
      text: The agent's response text.
    """
    self.conversation.append({"role": "assistant", "content": text})

  def truncate_last_assistant(self, spoken_text: str) -> None:
    """Truncate the last assistant message to what was actually spoken.

    Used when the caller interrupts — we only keep the text that
    was actually heard.

    Args:
      spoken_text: The portion of the response that was spoken before interruption.
    """
    if self.conversation and self.conversation[-1]["role"] == "assistant":
      self.conversation[-1]["content"] = spoken_text

  @property
  def duration_seconds(self) -> float:
    """Call duration in seconds from start to now."""
    return time() - self.started_at
