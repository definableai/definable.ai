"""Audio parser — passthrough for audio files.

Packages raw audio bytes as a ContentBlock for downstream consumers
(transcription APIs, audio processing pipelines). No parsing is performed.
"""

from typing import List, Set

from definable.reader.models import ContentBlock, ReaderConfig
from definable.reader.parsers.base_parser import BaseParser


class AudioParser(BaseParser):
  """Passthrough parser for audio files.

  Wraps raw audio bytes in an audio ContentBlock. Downstream consumers
  (transcription APIs, etc.) handle the actual audio processing.
  """

  def supported_mime_types(self) -> List[str]:
    return [
      "audio/mpeg",
      "audio/mp3",
      "audio/wav",
      "audio/x-wav",
      "audio/ogg",
      "audio/flac",
      "audio/x-flac",
      "audio/mp4",
      "audio/x-m4a",
      "audio/webm",
    ]

  def supported_extensions(self) -> Set[str]:
    return {
      ".mp3",
      ".wav",
      ".ogg",
      ".flac",
      ".m4a",
      ".webm",
    }

  def parse(self, data: bytes, *, mime_type: str | None = None, config: ReaderConfig | None = None) -> List[ContentBlock]:
    return [
      ContentBlock(
        content_type="audio",
        content=data,
        mime_type=mime_type or "audio/mpeg",
      )
    ]
