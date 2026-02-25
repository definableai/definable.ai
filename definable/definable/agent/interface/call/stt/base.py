"""Abstract base for real-time streaming speech-to-text providers."""

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class Transcript:
  """A speech-to-text transcript result.

  Attributes:
    text: The transcribed text.
    is_final: True if this is a finalized transcript (utterance complete).
      False for interim/partial results.
    confidence: Confidence score (0.0 to 1.0).
    duration_ms: Duration of the audio segment in milliseconds.
    language: Detected language code (if available).
  """

  text: str
  is_final: bool = False
  confidence: float = 1.0
  duration_ms: int = 0
  language: str = ""


@runtime_checkable
class STTProvider(Protocol):
  """Real-time streaming speech-to-text provider.

  Connects to an external STT service via WebSocket and streams
  audio chunks for real-time transcription.

  Lifecycle:
    1. ``connect()`` — establish connection with audio format config.
    2. ``send_audio()`` — stream audio chunks continuously.
    3. ``receive_transcripts()`` — async iterate over results.
    4. ``close()`` — tear down the connection.
  """

  async def connect(
    self,
    *,
    sample_rate: int = 8000,
    encoding: str = "mulaw",
    channels: int = 1,
  ) -> None:
    """Open a connection to the STT service.

    Args:
      sample_rate: Audio sample rate in Hz.
      encoding: Audio encoding (e.g. "mulaw", "linear16", "opus").
      channels: Number of audio channels.
    """
    ...

  async def send_audio(self, audio_bytes: bytes) -> None:
    """Stream an audio chunk to the STT service.

    Args:
      audio_bytes: Raw audio bytes in the configured encoding.
    """
    ...

  def receive_transcripts(self) -> AsyncIterator[Transcript]:
    """Async iterate over transcript results.

    Yields both interim (partial) and final transcripts.
    Final transcripts indicate a complete utterance.

    Yields:
      Transcript results as they arrive.
    """
    ...

  async def close(self) -> None:
    """Close the connection to the STT service."""
    ...
