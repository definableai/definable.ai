"""OpenAI Realtime API provider — speech-to-speech over WebSocket.

Uses the OpenAI Realtime API (Beta) for ultra-low latency voice AI.
Supports ``g711_ulaw`` natively for zero-transcoding with Twilio Media Streams.

Protocol:
  - Connect to ``wss://api.openai.com/v1/realtime?model=MODEL_ID``
  - Header: ``OpenAI-Beta: realtime=v1``
  - Send ``session.update`` to configure voice, tools, VAD, audio format
  - Forward audio via ``input_audio_buffer.append``
  - Receive audio via ``response.audio.delta``
  - Tool calls arrive as ``response.function_call_arguments.done``
  - Barge-in detected via ``input_audio_buffer.speech_started``
  - Send tool results via ``conversation.item.create`` + ``response.create``
"""

import base64
import contextlib
import json
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from definable.agent.interface.call.realtime.base import RealtimeEvent, RealtimeProvider
from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.interface.call._ws import WebSocketClient

# Map generic encoding names to OpenAI Realtime format strings
_ENCODING_MAP = {
  "mulaw": "g711_ulaw",
  "g711_ulaw": "g711_ulaw",
  "alaw": "g711_alaw",
  "g711_alaw": "g711_alaw",
  "pcm16": "pcm16",
  "linear16": "pcm16",
}


class OpenAIRealtimeProvider(RealtimeProvider):
  """OpenAI Realtime API speech-to-speech provider.

  Connects to OpenAI's WebSocket-based Realtime API for
  ultra-low latency voice AI with native function calling.

  Uses the Beta API (``OpenAI-Beta: realtime=v1``) which is
  proven for Twilio Media Streams integration.

  Supports ``g711_ulaw`` (mu-law 8kHz) natively — audio from
  Twilio flows directly to/from OpenAI with zero transcoding.

  Args:
    api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
    model: Realtime model ID.
    voice: Voice for synthesis ("alloy", "echo", "fable", "onyx", "nova", "shimmer").
    temperature: Generation temperature (0.6-1.2).
    max_response_output_tokens: Max tokens per response. "inf" for unlimited.
    turn_detection: Turn detection config. Defaults to server_vad.
    transcription_model: Model for input audio transcription.

  Example::

      provider = OpenAIRealtimeProvider(
        api_key="sk-...",
        model="gpt-4o-realtime-preview",
        voice="alloy",
      )
      await provider.connect(
        instructions="You are a phone agent.",
        tools=[{"type": "function", "name": "search", ...}],
      )
  """

  def __init__(
    self,
    *,
    api_key: str = "",
    model: str = "gpt-4o-realtime-preview",
    voice: str = "alloy",
    temperature: float = 0.8,
    max_response_output_tokens: str = "inf",
    turn_detection: Optional[Dict[str, Any]] = None,
    transcription_model: str = "whisper-1",
  ) -> None:
    self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    self._model = model
    self._voice = voice
    self._temperature = temperature
    self._max_response_output_tokens = max_response_output_tokens
    self._transcription_model = transcription_model
    self._turn_detection = turn_detection or {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500,
    }

    self._ws: Optional["WebSocketClient"] = None
    self._connected = False
    self._session_id: str = ""

  async def connect(
    self,
    *,
    instructions: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    voice: str = "alloy",
    input_encoding: str = "mulaw",
    input_sample_rate: int = 8000,
    output_encoding: str = "mulaw",
    output_sample_rate: int = 8000,
  ) -> None:
    """Connect to OpenAI Realtime API and configure the session.

    Establishes the WebSocket, waits for ``session.created``,
    then sends ``session.update`` with tools, voice, and audio format.
    """
    try:
      import websockets
    except ImportError as e:
      raise ImportError("websockets is required for OpenAIRealtimeProvider. Install it with: pip install 'definable[call]'") from e

    if not self._api_key:
      raise ValueError("OpenAI API key is required. Set api_key= or OPENAI_API_KEY env var.")

    url = f"wss://api.openai.com/v1/realtime?model={self._model}"
    headers = {
      "Authorization": f"Bearer {self._api_key}",
      "OpenAI-Beta": "realtime=v1",
    }

    try:
      self._ws = await websockets.connect(url, additional_headers=headers)  # type: ignore[attr-defined]
      assert self._ws is not None
      self._connected = True

      # Wait for session.created
      raw = await self._ws.recv()
      data = json.loads(raw)
      if data.get("type") == "session.created":
        self._session_id = data.get("session", {}).get("id", "")
        log_info(f"[openai-realtime] Session created: {self._session_id}")

      # Configure the session
      input_format = _ENCODING_MAP.get(input_encoding, "g711_ulaw")
      output_format = _ENCODING_MAP.get(output_encoding, "g711_ulaw")

      session_config: Dict[str, Any] = {
        "type": "session.update",
        "session": {
          "modalities": ["text", "audio"],
          "instructions": instructions,
          "voice": voice or self._voice,
          "input_audio_format": input_format,
          "output_audio_format": output_format,
          "input_audio_transcription": {
            "model": self._transcription_model,
          },
          "turn_detection": self._turn_detection,
          "temperature": self._temperature,
          "max_response_output_tokens": self._max_response_output_tokens,
        },
      }

      if tools:
        session_config["session"]["tools"] = tools
        session_config["session"]["tool_choice"] = "auto"

      await self._ws.send(json.dumps(session_config))
      log_info(f"[openai-realtime] Connected: model={self._model}, voice={voice or self._voice}, format={input_format}")

    except Exception as e:
      self._connected = False
      self._ws = None
      raise ConnectionError(f"Failed to connect to OpenAI Realtime API: {e}") from e

  async def send_audio(self, audio_bytes: bytes) -> None:
    """Send audio via ``input_audio_buffer.append``.

    Audio must be in the format specified during connect()
    (typically g711_ulaw for Twilio integration).
    The bytes are base64-encoded before sending.
    """
    if self._ws is None or not self._connected:
      return

    try:
      audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
      await self._ws.send(
        json.dumps({
          "type": "input_audio_buffer.append",
          "audio": audio_b64,
        })
      )
    except Exception as e:
      log_warning(f"[openai-realtime] Failed to send audio: {e}")

  async def receive_events(self) -> AsyncIterator[RealtimeEvent]:
    """Async iterate over events from the OpenAI Realtime API.

    Maps OpenAI server events to normalized :class:`RealtimeEvent` types:

      - ``response.audio.delta`` → ``audio_delta``
      - ``conversation.item.input_audio_transcription.completed`` → ``transcript``
      - ``response.audio_transcript.done`` → ``assistant_transcript``
      - ``response.function_call_arguments.done`` → ``tool_call``
      - ``input_audio_buffer.speech_started`` → ``speech_started``
      - ``input_audio_buffer.speech_stopped`` → ``speech_stopped``
      - ``response.done`` → ``turn_complete`` / ``interrupted``
      - ``error`` → ``error``
    """
    if self._ws is None:
      return

    try:
      async for raw in self._ws:
        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          continue

        event_type = data.get("type", "")

        # Audio chunk from the model
        if event_type == "response.audio.delta":
          audio_b64 = data.get("delta", "")
          if audio_b64:
            yield RealtimeEvent(
              type="audio_delta",
              audio=base64.b64decode(audio_b64),
              metadata={
                "response_id": data.get("response_id", ""),
                "item_id": data.get("item_id", ""),
              },
            )

        # User speech transcript (async, from Whisper)
        elif event_type == "conversation.item.input_audio_transcription.completed":
          yield RealtimeEvent(
            type="transcript",
            text=data.get("transcript", ""),
            metadata={"item_id": data.get("item_id", "")},
          )

        # Assistant speech transcript
        elif event_type == "response.audio_transcript.done":
          yield RealtimeEvent(
            type="assistant_transcript",
            text=data.get("transcript", ""),
            metadata={
              "response_id": data.get("response_id", ""),
              "item_id": data.get("item_id", ""),
            },
          )

        # Function call arguments complete
        elif event_type == "response.function_call_arguments.done":
          yield RealtimeEvent(
            type="tool_call",
            tool_call={
              "id": data.get("call_id", ""),
              "name": data.get("name", ""),
              "arguments": data.get("arguments", "{}"),
              "item_id": data.get("item_id", ""),
            },
            metadata={"response_id": data.get("response_id", "")},
          )

        # Speech started (barge-in signal)
        elif event_type == "input_audio_buffer.speech_started":
          yield RealtimeEvent(
            type="speech_started",
            metadata={
              "audio_start_ms": data.get("audio_start_ms", 0),
              "item_id": data.get("item_id", ""),
            },
          )

        # Speech stopped
        elif event_type == "input_audio_buffer.speech_stopped":
          yield RealtimeEvent(
            type="speech_stopped",
            metadata={
              "audio_end_ms": data.get("audio_end_ms", 0),
              "item_id": data.get("item_id", ""),
            },
          )

        # Response completed / cancelled / failed
        elif event_type == "response.done":
          response = data.get("response", {})
          status = response.get("status", "")
          usage = response.get("usage", {})

          if status == "cancelled":
            event_name = "interrupted"
          elif status == "failed":
            event_name = "error"
          else:
            event_name = "turn_complete"

          yield RealtimeEvent(
            type=event_name,
            metadata={
              "response_id": response.get("id", ""),
              "status": status,
              "usage": usage,
            },
          )

        # Error
        elif event_type == "error":
          error = data.get("error", {})
          yield RealtimeEvent(
            type="error",
            text=error.get("message", "Unknown error"),
            metadata=error,
          )

        # Session updated (confirmation — log only)
        elif event_type == "session.updated":
          log_debug("[openai-realtime] Session updated")

    except Exception as e:
      if self._connected:
        log_warning(f"[openai-realtime] WebSocket receive error: {e}")

  async def send_tool_result(self, call_id: str, result: str) -> None:
    """Return a tool call result and trigger a new response.

    Sends ``conversation.item.create`` with the function output,
    then ``response.create`` to have the model continue.
    """
    if self._ws is None or not self._connected:
      return

    try:
      # Send the function call output
      await self._ws.send(
        json.dumps({
          "type": "conversation.item.create",
          "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": result,
          },
        })
      )

      # Trigger the model to generate a response
      await self._ws.send(
        json.dumps({
          "type": "response.create",
        })
      )

      log_debug(f"[openai-realtime] Tool result sent for call_id={call_id}")
    except Exception as e:
      log_warning(f"[openai-realtime] Failed to send tool result: {e}")

  async def interrupt(self) -> None:
    """Cancel the current in-progress response."""
    if self._ws is None or not self._connected:
      return

    try:
      await self._ws.send(
        json.dumps({
          "type": "response.cancel",
        })
      )
      log_debug("[openai-realtime] Response cancelled")
    except Exception as e:
      log_warning(f"[openai-realtime] Failed to cancel response: {e}")

  async def send_truncate(self, item_id: str, audio_end_ms: int) -> None:
    """Truncate a previous assistant audio message.

    Tells the server how much audio was actually played before
    interruption, so the model's internal conversation state
    accurately reflects what the user heard.

    Args:
      item_id: The item ID of the interrupted response.
      audio_end_ms: Milliseconds of audio that were actually played.
    """
    if self._ws is None or not self._connected:
      return

    try:
      await self._ws.send(
        json.dumps({
          "type": "conversation.item.truncate",
          "item_id": item_id,
          "content_index": 0,
          "audio_end_ms": audio_end_ms,
        })
      )
    except Exception as e:
      log_warning(f"[openai-realtime] Failed to truncate: {e}")

  async def close(self) -> None:
    """Close the WebSocket connection."""
    self._connected = False

    if self._ws is not None:
      with contextlib.suppress(Exception):
        await self._ws.close()
      self._ws = None

    log_debug("[openai-realtime] Connection closed")
