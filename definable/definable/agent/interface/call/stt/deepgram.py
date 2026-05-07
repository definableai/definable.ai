"""Deepgram real-time streaming STT provider.

Uses raw WebSocket connection (no SDK dependency) for minimal footprint.

Protocol:
  - Connect to ``wss://api.deepgram.com/v1/listen?model=nova-3&...``
  - Send binary audio frames (mu-law 8kHz for telephony)
  - Receive JSON text frames with transcript results
  - Send JSON KeepAlive every 5s to prevent 10s timeout
  - Send ``{"type": "CloseStream"}`` to close gracefully
"""

import asyncio
import contextlib
import json
import os
from typing import TYPE_CHECKING, AsyncIterator, Optional

from definable.agent.interface.call.stt.base import Transcript
from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.interface.call._ws import WebSocketClient


class DeepgramSTT:
  """Deepgram Nova real-time streaming STT.

  Connects to Deepgram's WebSocket API and streams audio for
  real-time transcription. Supports interim and final results,
  voice activity detection, and configurable endpointing.

  Args:
    api_key: Deepgram API key. Falls back to ``DEEPGRAM_API_KEY`` env var.
    model: Deepgram model (e.g. "nova-3", "nova-2").
    language: BCP-47 language code.
    interim_results: Whether to receive partial transcripts.
    endpointing: Silence duration (ms) to trigger speech_final.
      Set to 0 to disable.
    vad_events: Whether to receive SpeechStarted events.
    utterance_end_ms: Milliseconds after last word to fire UtteranceEnd.
    smart_format: Enable smart formatting (numbers, dates, etc.).
    punctuate: Enable automatic punctuation.
    keepalive_interval: Seconds between KeepAlive messages.

  Example::

      stt = DeepgramSTT(api_key="...")
      await stt.connect(sample_rate=8000, encoding="mulaw")
      await stt.send_audio(audio_bytes)
      async for transcript in stt.receive_transcripts():
          if transcript.is_final:
              print(f"User said: {transcript.text}")
      await stt.close()
  """

  def __init__(
    self,
    *,
    api_key: str = "",
    model: str = "nova-3",
    language: str = "en-US",
    interim_results: bool = True,
    endpointing: int = 300,
    vad_events: bool = True,
    utterance_end_ms: int = 1000,
    smart_format: bool = True,
    punctuate: bool = True,
    keepalive_interval: float = 5.0,
  ) -> None:
    self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
    self._model = model
    self._language = language
    self._interim_results = interim_results
    self._endpointing = endpointing
    self._vad_events = vad_events
    self._utterance_end_ms = utterance_end_ms
    self._smart_format = smart_format
    self._punctuate = punctuate
    self._keepalive_interval = keepalive_interval

    self._ws: Optional["WebSocketClient"] = None
    self._keepalive_task: Optional[asyncio.Task] = None
    self._connected = False

  async def connect(
    self,
    *,
    sample_rate: int = 8000,
    encoding: str = "mulaw",
    channels: int = 1,
  ) -> None:
    """Open a WebSocket connection to Deepgram.

    Args:
      sample_rate: Audio sample rate in Hz.
      encoding: Audio encoding ("mulaw", "linear16", "opus", etc.).
      channels: Number of audio channels.

    Raises:
      ImportError: If websockets is not installed.
      ConnectionError: If the connection fails.
    """
    try:
      import websockets
    except ImportError as e:
      raise ImportError("websockets is required for DeepgramSTT. Install it with: pip install 'definable[call]'") from e

    if not self._api_key:
      raise ValueError("Deepgram API key is required. Set api_key= or DEEPGRAM_API_KEY env var.")

    url = self._build_url(sample_rate=sample_rate, encoding=encoding, channels=channels)
    headers = {"Authorization": f"Token {self._api_key}"}

    try:
      self._ws = await websockets.connect(url, extra_headers=headers)  # type: ignore[attr-defined]
      self._connected = True
      self._keepalive_task = asyncio.create_task(self._keepalive_loop())
      log_info(f"[deepgram] Connected: model={self._model}, encoding={encoding}, rate={sample_rate}")
    except Exception as e:
      raise ConnectionError(f"Failed to connect to Deepgram: {e}") from e

  async def send_audio(self, audio_bytes: bytes) -> None:
    """Stream an audio chunk to Deepgram.

    Args:
      audio_bytes: Raw audio bytes in the configured encoding.
    """
    if self._ws is None or not self._connected:
      return
    try:
      await self._ws.send(audio_bytes)
    except Exception as e:
      log_warning(f"[deepgram] Failed to send audio: {e}")

  async def receive_transcripts(self) -> AsyncIterator[Transcript]:
    """Async iterate over transcript results from Deepgram.

    Yields both interim (partial) and final transcripts.
    A transcript with ``is_final=True`` indicates a completed utterance.

    Yields:
      Transcript results as they arrive.
    """
    if self._ws is None:
      return

    try:
      async for raw_msg in self._ws:
        try:
          data = json.loads(raw_msg)
        except json.JSONDecodeError:
          continue

        msg_type = data.get("type", "")

        if msg_type == "Results":
          channel = data.get("channel", {})
          alternatives = channel.get("alternatives", [{}])
          if not alternatives:
            continue

          alt = alternatives[0]
          text = alt.get("transcript", "")
          if not text:
            continue

          is_final = data.get("is_final", False)
          speech_final = data.get("speech_final", False)

          yield Transcript(
            text=text,
            is_final=is_final and speech_final,
            confidence=alt.get("confidence", 0.0),
            duration_ms=int(data.get("duration", 0) * 1000),
            language=self._language,
          )

        elif msg_type == "UtteranceEnd":
          # Signal that speaker has stopped — yield a marker
          yield Transcript(text="", is_final=True, confidence=1.0)

        elif msg_type == "SpeechStarted":
          log_debug("[deepgram] Speech started")

        elif msg_type == "Metadata":
          log_debug(f"[deepgram] Metadata: request_id={data.get('request_id', 'unknown')}")

    except Exception as e:
      if self._connected:
        log_warning(f"[deepgram] WebSocket receive error: {e}")

  async def close(self) -> None:
    """Close the WebSocket connection gracefully."""
    self._connected = False

    if self._keepalive_task is not None:
      self._keepalive_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._keepalive_task
      self._keepalive_task = None

    if self._ws is not None:
      try:
        # Send CloseStream control message
        await self._ws.send(json.dumps({"type": "CloseStream"}))
        await self._ws.close()
      except Exception:
        pass
      self._ws = None

    log_debug("[deepgram] Connection closed")

  # --- Private ---

  def _build_url(self, *, sample_rate: int, encoding: str, channels: int) -> str:
    """Build the Deepgram WebSocket URL with query parameters."""
    params = {
      "model": self._model,
      "language": self._language,
      "encoding": encoding,
      "sample_rate": str(sample_rate),
      "channels": str(channels),
      "interim_results": str(self._interim_results).lower(),
      "endpointing": str(self._endpointing),
      "vad_events": str(self._vad_events).lower(),
      "utterance_end_ms": str(self._utterance_end_ms),
      "smart_format": str(self._smart_format).lower(),
      "punctuate": str(self._punctuate).lower(),
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"wss://api.deepgram.com/v1/listen?{qs}"

  async def _keepalive_loop(self) -> None:
    """Send periodic KeepAlive messages to prevent Deepgram's 10s timeout."""
    try:
      while self._connected and self._ws is not None:
        await asyncio.sleep(self._keepalive_interval)
        if self._connected and self._ws is not None:
          try:
            await self._ws.send(json.dumps({"type": "KeepAlive"}))
          except Exception:
            break
    except asyncio.CancelledError:
      pass
