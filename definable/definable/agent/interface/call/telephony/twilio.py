"""Twilio telephony provider — Media Streams and ConversationRelay."""

import base64
import hashlib
import hmac
import os
from typing import Any, Dict, Optional

from definable.agent.interface.call.telephony.base import TelephonyEvent, TelephonyProvider
from definable.utils.log import log_debug


class TwilioProvider(TelephonyProvider):
  """Twilio telephony provider.

  Supports two WebSocket modes:
    - **ConversationRelay** (managed pipeline): Twilio handles STT/TTS,
      sends transcribed text, receives text tokens.
    - **Media Streams** (cascading/realtime pipeline): Raw bidirectional
      audio over WebSocket (mu-law 8kHz).

  Args:
    account_sid: Twilio account SID. Falls back to ``TWILIO_ACCOUNT_SID`` env var.
    auth_token: Twilio auth token. Falls back to ``TWILIO_AUTH_TOKEN`` env var.
  """

  def __init__(
    self,
    *,
    account_sid: str = "",
    auth_token: str = "",
  ) -> None:
    self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
    self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")

  # --- XML generation ---

  def generate_answer_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate TwiML for incoming call.

    For managed mode, generates ``<Connect><ConversationRelay>``.
    For cascading/realtime mode, generates ``<Connect><Stream>``.

    The ``mode`` kwarg determines which TwiML is generated:
      - ``mode="managed"``: ConversationRelay with STT/TTS config.
      - ``mode="stream"``: Bidirectional Media Stream.

    Args:
      websocket_url: WebSocket URL to connect to.
      welcome_message: Greeting to speak/send on connect.
      **kwargs: Additional TwiML attributes.

    Returns:
      TwiML XML string.
    """
    mode = kwargs.get("mode", "managed")

    if mode == "managed":
      return self._generate_conversation_relay_xml(websocket_url, welcome_message=welcome_message, **kwargs)
    else:
      return self._generate_media_stream_xml(websocket_url, welcome_message=welcome_message, **kwargs)

  def _generate_conversation_relay_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate TwiML with ConversationRelay.

    Twilio manages STT (Deepgram/Google) and TTS (ElevenLabs/Google/Amazon).
    We receive/send text over WebSocket.
    """
    tts_provider = kwargs.get("tts_provider", "google")
    stt_provider = kwargs.get("stt_provider", "deepgram")
    voice = kwargs.get("voice", "en-US-Standard-A")
    language = kwargs.get("language", "en-US")
    interruptible = kwargs.get("interruptible", "any")
    interrupt_sensitivity = kwargs.get("interrupt_sensitivity", "medium")
    dtmf_detection = kwargs.get("dtmf_detection", "true")

    # Build ConversationRelay attributes
    attrs = [
      f'url="{websocket_url}"',
      f'ttsProvider="{tts_provider}"',
      f'transcriptionProvider="{stt_provider}"',
      f'voice="{voice}"',
      f'language="{language}"',
      f'interruptible="{interruptible}"',
      f'interruptSensitivity="{interrupt_sensitivity}"',
      f'dtmfDetection="{dtmf_detection}"',
    ]
    if welcome_message:
      # Escape XML special chars in the greeting
      safe_greeting = _escape_xml(welcome_message)
      attrs.append(f'welcomeGreeting="{safe_greeting}"')

    attrs_str = " ".join(attrs)

    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><ConversationRelay {attrs_str} /></Connect></Response>'
    log_debug(f"Generated ConversationRelay TwiML for {websocket_url}")
    return twiml

  def _generate_media_stream_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate TwiML with bidirectional Media Stream.

    Raw mu-law 8kHz audio is streamed over WebSocket.
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]

    if welcome_message:
      safe_msg = _escape_xml(welcome_message)
      parts.append(f"<Say>{safe_msg}</Say>")

    parts.append("<Connect>")
    parts.append(f'<Stream url="{websocket_url}" />')
    parts.append("</Connect>")
    parts.append("</Response>")

    twiml = "".join(parts)
    log_debug(f"Generated Media Stream TwiML for {websocket_url}")
    return twiml

  # --- WebSocket event parsing ---

  def parse_websocket_event(self, data: Dict[str, Any]) -> TelephonyEvent:
    """Parse a Twilio WebSocket message.

    Handles both ConversationRelay events (type-based) and
    Media Stream events (event-based).
    """
    # ConversationRelay uses "type" field
    if "type" in data:
      return self._parse_conversation_relay_event(data)

    # Media Streams uses "event" field
    if "event" in data:
      return self._parse_media_stream_event(data)

    return TelephonyEvent(event="unknown", metadata=data)

  def _parse_conversation_relay_event(self, data: Dict[str, Any]) -> TelephonyEvent:
    """Parse ConversationRelay WebSocket events.

    Events:
      setup — initial connection with callSid and custom params.
      prompt — transcribed caller speech.
      interrupt — caller interrupted, includes text spoken so far.
      dtmf — caller pressed a digit.
    """
    event_type = data.get("type", "unknown")
    call_id = data.get("callSid", "")

    if event_type == "setup":
      return TelephonyEvent(
        event="setup",
        call_id=call_id,
        metadata={
          "custom_parameters": data.get("customParameters", {}),
        },
      )

    if event_type == "prompt":
      return TelephonyEvent(
        event="prompt",
        call_id=call_id,
        payload=data.get("voicePrompt", ""),
      )

    if event_type == "interrupt":
      return TelephonyEvent(
        event="interrupt",
        call_id=call_id,
        payload=data.get("utteranceUntilInterrupt", ""),
      )

    if event_type == "dtmf":
      return TelephonyEvent(
        event="dtmf",
        call_id=call_id,
        payload=data.get("digit", ""),
      )

    return TelephonyEvent(event=event_type, call_id=call_id, metadata=data)

  def _parse_media_stream_event(self, data: Dict[str, Any]) -> TelephonyEvent:
    """Parse Media Stream WebSocket events.

    Events:
      connected — WebSocket connected.
      start — stream started, includes streamSid, callSid, mediaFormat.
      media — audio chunk (base64 mu-law).
      dtmf — digit pressed.
      mark — playback marker reached.
      stop — stream ended.
    """
    event_name = data.get("event", "unknown")

    if event_name == "start":
      start_data = data.get("start", {})
      return TelephonyEvent(
        event="start",
        call_id=start_data.get("callSid", ""),
        stream_id=start_data.get("streamSid", ""),
        metadata={
          "media_format": start_data.get("mediaFormat", {}),
          "custom_parameters": start_data.get("customParameters", {}),
        },
      )

    if event_name == "media":
      media_data = data.get("media", {})
      payload_b64 = media_data.get("payload", "")
      return TelephonyEvent(
        event="media",
        stream_id=data.get("streamSid", ""),
        payload=base64.b64decode(payload_b64) if payload_b64 else b"",
        metadata={
          "chunk": media_data.get("chunk", ""),
          "timestamp": media_data.get("timestamp", ""),
        },
      )

    if event_name == "dtmf":
      dtmf_data = data.get("dtmf", {})
      return TelephonyEvent(
        event="dtmf",
        stream_id=data.get("streamSid", ""),
        payload=dtmf_data.get("digit", ""),
      )

    if event_name == "mark":
      mark_data = data.get("mark", {})
      return TelephonyEvent(
        event="mark",
        stream_id=data.get("streamSid", ""),
        payload=mark_data.get("name", ""),
      )

    if event_name == "stop":
      return TelephonyEvent(
        event="stop",
        stream_id=data.get("streamSid", ""),
      )

    return TelephonyEvent(
      event=event_name,
      stream_id=data.get("streamSid", ""),
      metadata=data,
    )

  # --- Response encoding ---

  def encode_audio_response(self, audio_bytes: bytes, stream_id: str) -> Dict[str, Any]:
    """Encode audio for Media Stream WebSocket (base64 mu-law)."""
    return {
      "event": "media",
      "streamSid": stream_id,
      "media": {
        "payload": base64.b64encode(audio_bytes).decode("ascii"),
      },
    }

  def encode_clear_audio(self, stream_id: str) -> Dict[str, Any]:
    """Encode a clear-buffer command for barge-in."""
    return {
      "event": "clear",
      "streamSid": stream_id,
    }

  def encode_text_response(self, text: str, *, last: bool = False) -> Dict[str, Any]:
    """Encode a text token for ConversationRelay."""
    return {
      "type": "text",
      "token": text,
      "last": last,
    }

  # --- Webhook signature validation ---

  def validate_webhook_signature(self, body: bytes, signature: str, url: str, **kwargs: Any) -> bool:
    """Validate Twilio webhook signature (X-Twilio-Signature).

    Uses HMAC-SHA1 as specified by Twilio's security docs.
    """
    if not self.auth_token:
      return False

    # Twilio signature validation: HMAC-SHA1 of URL + sorted POST params
    mac = hmac.new(self.auth_token.encode("utf-8"), url.encode("utf-8"), hashlib.sha1)
    expected = base64.b64encode(mac.digest()).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _escape_xml(text: str) -> str:
  """Escape XML special characters in text."""
  return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
