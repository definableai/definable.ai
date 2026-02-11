"""Audio file reader — transcribes audio files to text.

Uses an AudioTranscriber protocol so any transcription backend can be
plugged in. Ships with OpenAITranscriber (uses the Whisper API) as the
default.
"""

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from definable.media import File
from definable.readers.base import FileReader, FileReaderConfig, ReaderResult


@runtime_checkable
class AudioTranscriber(Protocol):
  """Protocol for audio transcription backends."""

  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs: Any) -> str: ...

  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs: Any) -> str: ...


_MIME_TO_EXT = {
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
  "audio/x-flac": "flac",
  "audio/mp4": "m4a",
  "audio/x-m4a": "m4a",
  "audio/webm": "webm",
}


@dataclass
class OpenAITranscriber:
  """Default transcriber using the OpenAI Whisper API."""

  model: str = "whisper-1"
  language: Optional[str] = None
  api_key: Optional[str] = None

  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs: Any) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
    ext = _MIME_TO_EXT.get(mime_type, "mp3")
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = f"audio.{ext}"

    create_kwargs: Dict[str, Any] = {"model": self.model, "file": audio_file}
    if self.language is not None:
      create_kwargs["language"] = self.language

    transcript = client.audio.transcriptions.create(**create_kwargs)
    return transcript.text

  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs: Any) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=self.api_key) if self.api_key else AsyncOpenAI()
    ext = _MIME_TO_EXT.get(mime_type, "mp3")
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = f"audio.{ext}"

    create_kwargs: Dict[str, Any] = {"model": self.model, "file": audio_file}
    if self.language is not None:
      create_kwargs["language"] = self.language

    transcript = await client.audio.transcriptions.create(**create_kwargs)
    return transcript.text


@dataclass
class AudioFileReader(FileReader):
  """Reads audio files by transcribing them to text.

  Uses ``OpenAITranscriber`` by default. Pass a custom ``AudioTranscriber``
  implementation to use a different backend.
  """

  config: Optional[FileReaderConfig] = None
  transcriber: Any = None  # AudioTranscriber — set in __post_init__

  def __post_init__(self) -> None:
    if self.transcriber is None:
      self.transcriber = OpenAITranscriber()

  def supported_mime_types(self) -> List[str]:
    return list(_MIME_TO_EXT.keys())

  def supported_extensions(self) -> Set[str]:
    return {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}

  def read_file(self, file: File) -> ReaderResult:
    try:
      raw = self._get_file_bytes(file)
      self._check_file_size(raw)
      mime = file.mime_type or "audio/mpeg"
      text: str = self.transcriber.transcribe(raw, mime)
      text, truncated = self._truncate(text)
      return ReaderResult(
        filename=self._get_filename(file),
        content=text,
        mime_type=mime,
        word_count=len(text.split()),
        truncated=truncated,
      )
    except Exception as e:
      return self._make_error_result(file, str(e))

  async def aread_file(self, file: File) -> ReaderResult:
    try:
      raw = await self._aget_file_bytes(file)
      self._check_file_size(raw)
      mime = file.mime_type or "audio/mpeg"
      text: str = await self.transcriber.atranscribe(raw, mime)
      text, truncated = self._truncate(text)
      return ReaderResult(
        filename=self._get_filename(file),
        content=text,
        mime_type=mime,
        word_count=len(text.split()),
        truncated=truncated,
      )
    except Exception as e:
      return self._make_error_result(file, str(e))
