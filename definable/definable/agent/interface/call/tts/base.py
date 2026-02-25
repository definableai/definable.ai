"""Abstract base for streaming text-to-speech providers."""

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class TTSProvider(Protocol):
  """Streaming text-to-speech provider.

  Synthesizes text into audio chunks streamed in real-time,
  suitable for telephony playback.

  Lifecycle:
    1. Call ``synthesize_stream()`` with text to speak.
    2. Async iterate over audio chunks as they're generated.
    3. Call ``close()`` when done.
  """

  def synthesize_stream(
    self,
    text: str,
    *,
    voice: str = "default",
    encoding: str = "mulaw",
    sample_rate: int = 8000,
  ) -> AsyncIterator[bytes]:
    """Synthesize text into streaming audio chunks.

    Implementations should be async generators (``async def`` with ``yield``).
    Starts generating audio immediately and yields chunks
    as they become available (low TTFB).

    Args:
      text: Text to synthesize.
      voice: Voice name/ID to use.
      encoding: Output audio encoding (e.g. "mulaw", "linear16", "mp3").
      sample_rate: Output sample rate in Hz.

    Yields:
      Audio chunks in the requested encoding.
    """
    ...

  async def close(self) -> None:
    """Close the connection to the TTS service."""
    ...
