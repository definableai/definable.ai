"""Plivo telephony provider — bidirectional Audio Streaming.

Uses Plivo's Audio Streaming for real-time bidirectional audio over WebSocket.

Protocol:
  - Incoming call → Plivo sends webhook to ``answer_url``
  - Return Plivo XML with ``<Stream bidirectional="true">`` pointing to our WebSocket
  - Plivo connects WebSocket, sends ``start`` event with ``callId`` / ``streamId``
  - Audio arrives as ``media`` events with base64-encoded payload
  - Send audio back via ``playAudio`` events
  - Clear audio buffer for barge-in via ``clearAudio`` events
  - DTMF digits arrive as top-level ``dtmf`` events
  - Stream ends with ``stop`` event

Key differences from Twilio:
  - No ConversationRelay — managed mode is NOT supported
  - ``<Stream>`` URL is text content, not a ``url=`` attribute
  - ``keepCallAlive="true"`` is MANDATORY or the call drops immediately
  - Send audio via ``playAudio`` (not ``media``)
  - Clear audio via ``clearAudio`` (not ``clear``)
  - DTMF ``digit`` is top-level (not nested in ``dtmf.digit``)
  - Supports 16kHz PCM natively (Twilio only supports 8kHz mu-law)
  - Webhook signature uses HMAC-SHA256 V3 with nonce (not HMAC-SHA1)
"""

import base64
import hashlib
import hmac
import os
from typing import Any, Dict, Optional

from definable.agent.interface.call.telephony.base import TelephonyEvent, TelephonyProvider
from definable.utils.log import log_debug


class PlivoProvider(TelephonyProvider):
  """Plivo telephony provider.

  Supports bidirectional Audio Streaming for cascading and realtime
  pipelines. Raw audio (mu-law 8kHz or PCM L16) flows over WebSocket.

  **Managed mode is NOT supported** — Plivo does not have a
  ConversationRelay equivalent. Use ``pipeline="cascading"`` or
  ``pipeline="realtime"`` instead.

  Args:
    auth_id: Plivo Auth ID. Falls back to ``PLIVO_AUTH_ID`` env var.
    auth_token: Plivo Auth Token. Falls back to ``PLIVO_AUTH_TOKEN`` env var.

  Example::

      from definable.agent.interface.call import CallInterface, PlivoProvider

      call = CallInterface(
        provider="plivo",
        auth_id="MA...",
        auth_token="...",
        phone_number="+15551234567",
        pipeline="cascading",
        stt=DeepgramSTT(...),
        tts=CartesiaTTS(...),
      )
  """

  def __init__(
    self,
    *,
    auth_id: str = "",
    auth_token: str = "",
  ) -> None:
    self.auth_id = auth_id or os.environ.get("PLIVO_AUTH_ID", "")
    self.auth_token = auth_token or os.environ.get("PLIVO_AUTH_TOKEN", "")

  # --- XML generation ---

  def generate_answer_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate Plivo XML for incoming call.

    For cascading/realtime mode, generates ``<Stream bidirectional="true">``.

    Managed mode is NOT supported — Plivo has no ConversationRelay
    equivalent. Raises ``ValueError`` if ``mode="managed"``.

    Args:
      websocket_url: WebSocket URL to connect to.
      welcome_message: Greeting to speak on connect.
      **kwargs: Additional XML attributes (``mode`` is required).

    Returns:
      Plivo XML string.

    Raises:
      ValueError: If ``mode="managed"`` (not supported by Plivo).
    """
    mode = kwargs.get("mode", "stream")

    if mode == "managed":
      raise ValueError(
        "Plivo does not support managed pipeline mode (no ConversationRelay equivalent). Use pipeline='cascading' or pipeline='realtime' instead."
      )

    return self._generate_audio_stream_xml(websocket_url, welcome_message=welcome_message, **kwargs)

  def _generate_audio_stream_xml(
    self,
    websocket_url: str,
    *,
    welcome_message: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Generate Plivo XML with bidirectional Audio Stream.

    Audio format defaults to mu-law 8kHz for telephony compatibility.
    ``keepCallAlive="true"`` is always set — without it, the call
    drops immediately when the stream element is processed.
    """
    content_type = kwargs.get("content_type", "audio/x-mulaw;rate=8000")

    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<Response>"]

    if welcome_message:
      safe_msg = _escape_xml(welcome_message)
      parts.append(f"<Speak>{safe_msg}</Speak>")

    # keepCallAlive MUST be true or the call drops immediately
    parts.append(f'<Stream bidirectional="true" keepCallAlive="true" contentType="{content_type}">')
    parts.append(websocket_url)
    parts.append("</Stream>")
    parts.append("</Response>")

    xml = "".join(parts)
    log_debug(f"Generated Plivo Audio Stream XML for {websocket_url}")
    return xml

  # --- WebSocket event parsing ---

  def parse_websocket_event(self, data: Dict[str, Any]) -> TelephonyEvent:
    """Parse a Plivo Audio Stream WebSocket message.

    Events:
      start — stream started, includes callId, streamId, mediaFormat.
      media — audio chunk (base64, encoding per stream config).
      dtmf — digit pressed (top-level ``digit`` field).
      stop — stream ended.
    """
    event_name = data.get("event", "unknown")

    if event_name == "start":
      start_data = data.get("start", {})
      return TelephonyEvent(
        event="start",
        call_id=start_data.get("callId", ""),
        stream_id=start_data.get("streamId", ""),
        metadata={
          "account_id": start_data.get("accountId", ""),
          "tracks": start_data.get("tracks", []),
          "media_format": start_data.get("mediaFormat", {}),
          "extra_headers": data.get("extra_headers", ""),
        },
      )

    if event_name == "media":
      media_data = data.get("media", {})
      payload_b64 = media_data.get("payload", "")
      return TelephonyEvent(
        event="media",
        stream_id=data.get("streamId", ""),
        payload=base64.b64decode(payload_b64) if payload_b64 else b"",
        metadata={
          "track": media_data.get("track", ""),
          "chunk": media_data.get("chunk", ""),
          "timestamp": media_data.get("timestamp", ""),
        },
      )

    if event_name == "dtmf":
      # Plivo sends DTMF as top-level fields (not nested like Twilio)
      return TelephonyEvent(
        event="dtmf",
        stream_id=data.get("streamId", ""),
        payload=data.get("digit", ""),
        metadata={
          "track": data.get("track", ""),
        },
      )

    if event_name == "stop":
      return TelephonyEvent(
        event="stop",
        stream_id=data.get("streamId", ""),
      )

    return TelephonyEvent(
      event=event_name,
      stream_id=data.get("streamId", ""),
      metadata=data,
    )

  # --- Response encoding ---

  def encode_audio_response(self, audio_bytes: bytes, stream_id: str) -> Dict[str, Any]:
    """Encode audio for Plivo bidirectional stream (``playAudio`` event).

    Audio is base64-encoded and sent with content type metadata.
    Defaults to mu-law 8kHz, matching telephony standard.
    """
    return {
      "event": "playAudio",
      "media": {
        "contentType": "audio/x-mulaw",
        "sampleRate": "8000",
        "payload": base64.b64encode(audio_bytes).decode("ascii"),
      },
    }

  def encode_clear_audio(self, stream_id: str) -> Dict[str, Any]:
    """Encode a clear-buffer command for barge-in (``clearAudio`` event)."""
    return {
      "event": "clearAudio",
      "streamId": stream_id,
    }

  def encode_text_response(self, text: str, *, last: bool = False) -> Dict[str, Any]:
    """Not supported — Plivo has no ConversationRelay.

    Raises:
      NotImplementedError: Always. Plivo does not support managed mode.
    """
    raise NotImplementedError(
      "Plivo does not support text-based responses (no ConversationRelay). Use pipeline='cascading' or pipeline='realtime' instead."
    )

  # --- Webhook signature validation ---

  def validate_webhook_signature(self, body: bytes, signature: str, url: str, **kwargs: Any) -> bool:
    """Validate Plivo webhook signature (V3 HMAC-SHA256).

    Uses Plivo's V3 signature scheme: HMAC-SHA256 of
    ``url + nonce`` (GET) or ``url + nonce + sorted_params`` (POST).

    Args:
      body: Raw request body (unused — params passed via kwargs).
      signature: Value of ``X-Plivo-Signature-V3`` header.
      url: The full request URL.
      **kwargs: Must include ``nonce`` (from ``X-Plivo-Signature-V3-Nonce``).
        Optionally include ``method`` ("GET" or "POST", default "POST")
        and ``params`` (dict of POST form params for POST requests).

    Returns:
      True if the signature is valid.
    """
    if not self.auth_token:
      return False

    nonce = kwargs.get("nonce", "")
    if not nonce:
      return False

    method = kwargs.get("method", "POST")
    params = kwargs.get("params")  # Optional[Dict[str, str]]

    # Build the string to sign
    string_to_sign = url + nonce

    if method == "POST" and params:
      sorted_params = sorted(params.items())
      query_string = "&".join(f"{k}={v}" for k, v in sorted_params)
      string_to_sign += query_string

    # Compute HMAC-SHA256
    mac = hmac.new(
      self.auth_token.encode("utf-8"),
      string_to_sign.encode("utf-8"),
      hashlib.sha256,
    )
    expected = base64.b64encode(mac.digest()).decode("ascii")
    return hmac.compare_digest(expected, signature)


def _escape_xml(text: str) -> str:
  """Escape XML special characters in text."""
  return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
