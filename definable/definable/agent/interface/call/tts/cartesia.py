"""Cartesia Sonic real-time streaming TTS provider.

Uses raw WebSocket connection (no SDK dependency) for minimal footprint.

Protocol:
  - Connect to ``wss://api.cartesia.ai/tts/websocket?api_key=...``
  - Send JSON request with model_id, transcript, voice, output_format
  - Receive JSON messages with base64-encoded audio chunks
  - Audio chunks arrive as ``{"type": "chunk", "data": "base64..."}``
  - Completion signaled by ``{"type": "done"}``
  - Supports context continuations for LLM token streaming
"""

import base64
import contextlib
import json
import os
from typing import TYPE_CHECKING, AsyncIterator, Optional

from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.interface.call._ws import WebSocketClient

# Cartesia API version
_CARTESIA_VERSION = "2024-11-13"


class CartesiaTTS:
  """Cartesia Sonic real-time streaming TTS.

  Connects to Cartesia's WebSocket API for ultra-low latency
  text-to-speech synthesis (40-90ms TTFB).

  Supports ``pcm_mulaw`` at 8kHz for direct telephony output,
  eliminating the need for audio transcoding.

  Args:
    api_key: Cartesia API key. Falls back to ``CARTESIA_API_KEY`` env var.
    model: Cartesia model ID (e.g. "sonic-2", "sonic-3").
    voice_id: Default voice ID for synthesis.
    language: Language code (e.g. "en", "fr", "es").
    speed: Speech speed ("slow", "normal", "fast").
    cartesia_version: API version string.

  Example::

      tts = CartesiaTTS(api_key="...", voice_id="a0e99841-...")
      async for chunk in tts.synthesize_stream("Hello!", encoding="mulaw", sample_rate=8000):
          send_to_caller(chunk)  # raw mu-law audio bytes
      await tts.close()
  """

  def __init__(
    self,
    *,
    api_key: str = "",
    model: str = "sonic-2",
    voice_id: str = "",
    language: str = "en",
    speed: str = "normal",
    cartesia_version: str = _CARTESIA_VERSION,
  ) -> None:
    self._api_key = api_key or os.environ.get("CARTESIA_API_KEY", "")
    self._model = model
    self._voice_id = voice_id
    self._language = language
    self._speed = speed
    self._cartesia_version = cartesia_version

    self._ws: Optional["WebSocketClient"] = None
    self._connected = False
    self._context_counter = 0

  async def _ensure_connected(self) -> None:
    """Ensure we have an active WebSocket connection. Reconnect if needed."""
    if self._ws is not None and self._connected:
      return

    try:
      import websockets
    except ImportError as e:
      raise ImportError("websockets is required for CartesiaTTS. Install it with: pip install 'definable[call]'") from e

    if not self._api_key:
      raise ValueError("Cartesia API key is required. Set api_key= or CARTESIA_API_KEY env var.")

    url = f"wss://api.cartesia.ai/tts/websocket?api_key={self._api_key}&cartesia_version={self._cartesia_version}"

    try:
      self._ws = await websockets.connect(url)  # type: ignore[attr-defined]
      self._connected = True
      log_info(f"[cartesia] Connected: model={self._model}")
    except Exception as e:
      raise ConnectionError(f"Failed to connect to Cartesia: {e}") from e

  async def synthesize_stream(
    self,
    text: str,
    *,
    voice: str = "default",
    encoding: str = "mulaw",
    sample_rate: int = 8000,
  ) -> AsyncIterator[bytes]:
    """Synthesize text into streaming audio chunks.

    Connects (or reuses connection) to Cartesia's WebSocket API,
    sends the text, and yields audio chunks as they're generated.

    Args:
      text: Text to synthesize.
      voice: Voice ID to use. Falls back to the default voice_id.
      encoding: Output encoding ("mulaw", "pcm_s16le", "pcm_f32le", "pcm_alaw").
      sample_rate: Output sample rate in Hz.

    Yields:
      Raw audio bytes in the requested encoding.
    """
    if not text or not text.strip():
      return

    await self._ensure_connected()

    # Resolve voice ID
    voice_id = voice if voice != "default" else self._voice_id
    if not voice_id:
      raise ValueError("Voice ID is required. Set voice_id= in constructor or pass voice= to synthesize_stream().")

    # Map encoding names to Cartesia format
    cartesia_encoding = _map_encoding(encoding)

    # Generate unique context ID for this utterance
    self._context_counter += 1
    context_id = f"ctx-{self._context_counter}"

    request = {
      "model_id": self._model,
      "transcript": text,
      "voice": {"mode": "id", "id": voice_id},
      "context_id": context_id,
      "output_format": {
        "container": "raw",
        "encoding": cartesia_encoding,
        "sample_rate": sample_rate,
      },
      "language": self._language,
    }

    if self._speed != "normal":
      request["speed"] = self._speed

    assert self._ws is not None  # guaranteed by _ensure_connected()

    try:
      await self._ws.send(json.dumps(request))

      async for raw_msg in self._ws:
        try:
          msg = json.loads(raw_msg)
        except json.JSONDecodeError:
          continue

        msg_context = msg.get("context_id", "")
        if msg_context != context_id:
          # Response for a different context (multiplexing)
          continue

        msg_type = msg.get("type", "")

        if msg_type == "chunk":
          audio_b64 = msg.get("data", "")
          if audio_b64:
            yield base64.b64decode(audio_b64)

        elif msg_type == "error":
          error_msg = msg.get("error", "Unknown Cartesia error")
          log_error(f"[cartesia] TTS error: {error_msg}")
          raise RuntimeError(f"Cartesia TTS error: {error_msg}")

        elif msg_type == "done":
          log_debug(f"[cartesia] Synthesis complete for context {context_id}")
          break

    except RuntimeError:
      raise
    except Exception as e:
      log_warning(f"[cartesia] Stream error: {e}")
      self._connected = False
      self._ws = None

  async def close(self) -> None:
    """Close the WebSocket connection."""
    self._connected = False

    if self._ws is not None:
      with contextlib.suppress(Exception):
        await self._ws.close()
      self._ws = None

    log_debug("[cartesia] Connection closed")


def _map_encoding(encoding: str) -> str:
  """Map generic encoding names to Cartesia's encoding enum."""
  mapping = {
    "mulaw": "pcm_mulaw",
    "pcm_mulaw": "pcm_mulaw",
    "alaw": "pcm_alaw",
    "pcm_alaw": "pcm_alaw",
    "linear16": "pcm_s16le",
    "pcm_s16le": "pcm_s16le",
    "pcm_f32le": "pcm_f32le",
  }
  result = mapping.get(encoding)
  if result is None:
    raise ValueError(f"Unsupported Cartesia encoding: {encoding!r}. Supported: {', '.join(sorted(mapping.keys()))}")
  return result
