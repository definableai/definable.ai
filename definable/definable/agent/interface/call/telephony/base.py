"""Abstract base for telephony providers (Twilio, Plivo, etc.)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TelephonyEvent:
  """Parsed event from a telephony provider's WebSocket stream.

  Attributes:
    event: Event name (e.g. "start", "media", "stop", "prompt", "interrupt").
    call_id: Provider call identifier.
    stream_id: WebSocket stream identifier.
    payload: Event-specific data (audio bytes, text, DTMF digits, etc.).
    metadata: Additional provider metadata.
  """

  event: str
  call_id: str = ""
  stream_id: str = ""
  payload: Any = None
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallInfo:
  """Information about an initiated or received call.

  Attributes:
    call_id: Provider call identifier.
    from_number: Caller's phone number.
    to_number: Called phone number.
    status: Call status string from the provider.
    metadata: Additional provider-specific data.
  """

  call_id: str
  from_number: str = ""
  to_number: str = ""
  status: str = ""
  metadata: Dict[str, Any] = field(default_factory=dict)


class TelephonyProvider(ABC):
  """Abstract base for telephony providers.

  Handles call lifecycle, XML generation for incoming call webhooks,
  and WebSocket audio/text stream protocol translation.

  Subclasses implement provider-specific XML generation, WebSocket
  message parsing, and call management.
  """

  @abstractmethod
  def generate_answer_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate XML response for an incoming call webhook.

    The XML instructs the telephony provider to connect the call
    to our WebSocket endpoint.

    Args:
      websocket_url: Full WebSocket URL (wss://...) for the audio/text stream.
      welcome_message: Optional greeting to speak before connecting.
      **kwargs: Provider-specific XML attributes.

    Returns:
      XML string (TwiML for Twilio, Plivo XML for Plivo).
    """
    ...

  @abstractmethod
  def parse_websocket_event(self, data: Dict[str, Any]) -> TelephonyEvent:
    """Parse a WebSocket message from the provider into a TelephonyEvent.

    Args:
      data: Parsed JSON from the WebSocket message.

    Returns:
      Normalized TelephonyEvent.
    """
    ...

  @abstractmethod
  def encode_audio_response(self, audio_bytes: bytes, stream_id: str) -> Dict[str, Any]:
    """Encode audio for sending back to the provider via WebSocket.

    Args:
      audio_bytes: Raw audio bytes (mu-law 8kHz for telephony).
      stream_id: The stream identifier from the provider.

    Returns:
      JSON-serializable dict to send over the WebSocket.
    """
    ...

  @abstractmethod
  def encode_clear_audio(self, stream_id: str) -> Dict[str, Any]:
    """Encode a 'clear audio buffer' command for barge-in handling.

    Flushes any queued audio on the provider side so the caller
    stops hearing the previous response.

    Args:
      stream_id: The stream identifier from the provider.

    Returns:
      JSON-serializable dict to send over the WebSocket.
    """
    ...

  @abstractmethod
  def encode_text_response(self, text: str, *, last: bool = False) -> Dict[str, Any]:
    """Encode a text token for managed (text-based) pipelines.

    Used by ConversationRelay-style managed pipelines where
    the provider handles TTS.

    Args:
      text: Text token to send.
      last: Whether this is the final token in the response.

    Returns:
      JSON-serializable dict to send over the WebSocket.
    """
    ...

  @abstractmethod
  def validate_webhook_signature(self, body: bytes, signature: str, url: str, **kwargs: Any) -> bool:
    """Validate that a webhook request is authentically from the provider.

    Args:
      body: Raw request body bytes.
      signature: Signature header value from the request.
      url: The full request URL.
      **kwargs: Provider-specific params (e.g. ``nonce`` for Plivo V3).

    Returns:
      True if the signature is valid.
    """
    ...

  async def make_call(self, to: str, from_: str, webhook_url: str, **kwargs: Any) -> CallInfo:
    """Initiate an outbound call.

    Args:
      to: Destination phone number.
      from_: Source phone number (must be owned by you).
      webhook_url: URL to receive call events.
      **kwargs: Provider-specific call options.

    Returns:
      CallInfo with the new call's details.

    Raises:
      NotImplementedError: If outbound calls are not yet supported.
    """
    raise NotImplementedError("Outbound calls are not yet implemented for this provider.")

  async def end_call(self, call_id: str) -> None:
    """Hang up an active call.

    Args:
      call_id: Provider call identifier.

    Raises:
      NotImplementedError: If call management is not yet supported.
    """
    raise NotImplementedError("Call management is not yet implemented for this provider.")
