"""Abstract base for real-time speech-to-speech providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class RealtimeEvent:
  """An event from a speech-to-speech provider.

  Attributes:
    type: Event type — "audio_delta", "transcript", "tool_call",
      "interrupted", "turn_complete", "error".
    audio: Audio bytes (for audio_delta events).
    text: Text content (for transcript events).
    tool_call: Tool call dict (for tool_call events).
    metadata: Additional event data.
  """

  type: str
  audio: Optional[bytes] = None
  text: Optional[str] = None
  tool_call: Optional[Dict[str, Any]] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


class RealtimeProvider(ABC):
  """Abstract base for speech-to-speech providers.

  A RealtimeProvider connects to a service that natively handles
  audio input and output (e.g. OpenAI Realtime API, ElevenLabs
  Conversational AI). The provider handles STT and TTS internally,
  bypassing separate STT/TTS pipelines.

  Lifecycle:
    1. ``connect()`` — establish session with instructions and tools.
    2. ``send_audio()`` — stream audio input.
    3. ``receive_events()`` — async iterate over response events.
    4. ``send_tool_result()`` — return tool call results.
    5. ``interrupt()`` — signal barge-in.
    6. ``close()`` — tear down the session.
  """

  @abstractmethod
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
    """Open a session with the speech-to-speech service.

    Args:
      instructions: System instructions for the model.
      tools: Tool definitions in OpenAI function-calling format.
      voice: Voice name/ID for output audio.
      input_encoding: Input audio encoding.
      input_sample_rate: Input audio sample rate.
      output_encoding: Output audio encoding.
      output_sample_rate: Output audio sample rate.
    """
    ...

  @abstractmethod
  async def send_audio(self, audio_bytes: bytes) -> None:
    """Stream audio input to the provider.

    Args:
      audio_bytes: Raw audio bytes in the configured encoding.
    """
    ...

  @abstractmethod
  def receive_events(self) -> AsyncIterator[RealtimeEvent]:
    """Async iterate over response events.

    Yields events as they arrive: audio deltas (streaming audio
    chunks), transcripts, tool calls, interruption signals, and
    turn completion markers.

    Yields:
      RealtimeEvent instances.
    """
    ...

  @abstractmethod
  async def send_tool_result(self, call_id: str, result: str) -> None:
    """Return a tool call result to the provider.

    Args:
      call_id: The tool call ID from the RealtimeEvent.
      result: The tool's return value as a string.
    """
    ...

  @abstractmethod
  async def interrupt(self) -> None:
    """Signal the provider to stop current generation (barge-in)."""
    ...

  @abstractmethod
  async def close(self) -> None:
    """Close the session and release resources."""
    ...
