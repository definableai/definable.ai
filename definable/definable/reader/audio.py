"""Audio file reader — transcribes audio files to text.

Uses an AudioTranscriber protocol so any transcription backend can be
plugged in. Ships with OpenAITranscriber (uses the Whisper API) as the
default.

Also provides format normalization utilities so any consumer
(model input_audio APIs, transcription services, etc.) can convert
audio from platform-native formats (Telegram OGA, Discord OGG) to
formats the target consumer accepts.
"""

import io
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from definable.media import File
from definable.reader.base import BaseReader
from definable.reader.models import ReaderConfig, ReaderOutput
from definable.utils.log import log_debug


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
  "audio/x-ogg": "ogg",
  "audio/opus": "ogg",
  "audio/flac": "flac",
  "audio/x-flac": "flac",
  "audio/mp4": "m4a",
  "audio/x-m4a": "m4a",
  "audio/webm": "webm",
}

# Extension aliases — different names for the same underlying codec/container
_EXT_ALIASES: Dict[str, str] = {
  "oga": "ogg",  # Telegram OGG Opus
  "opus": "ogg",  # Raw Opus
  "mpga": "mp3",  # MPEG audio
  "mpeg": "mp3",  # MPEG audio
}


# ---------------------------------------------------------------------------
# Format normalization
# ---------------------------------------------------------------------------

# Formats that OpenAI's input_audio API accepts
OPENAI_INPUT_AUDIO_FORMATS: Set[str] = {"wav", "mp3"}


def is_ffmpeg_available() -> bool:
  """Check whether ffmpeg is on PATH."""
  return shutil.which("ffmpeg") is not None


def normalize_audio_format(
  audio_bytes: bytes,
  source_format: str,
  *,
  target_formats: Optional[Set[str]] = None,
) -> Tuple[bytes, str]:
  """Normalize audio bytes to a format the target consumer accepts.

  If ``source_format`` (after alias resolution) is already in
  ``target_formats``, the original bytes are returned unchanged.
  Otherwise, ``ffmpeg`` is used to transcode to the first supported
  target format (``wav`` by default).

  Args:
    audio_bytes: Raw audio data.
    source_format: File extension of the source (e.g. ``"oga"``, ``"ogg"``).
    target_formats: Accepted output formats. Defaults to
      :data:`OPENAI_INPUT_AUDIO_FORMATS` (``{"wav", "mp3"}``).

  Returns:
    ``(output_bytes, output_format)`` — either the original data
    or transcoded bytes with the new format string.

  Raises:
    RuntimeError: If transcoding is required but ``ffmpeg`` is not
      installed.
  """
  if target_formats is None:
    target_formats = OPENAI_INPUT_AUDIO_FORMATS

  # Resolve aliases (oga → ogg, mpga → mp3, etc.)
  resolved = _EXT_ALIASES.get(source_format.lower(), source_format.lower())

  # Already compatible — return as-is
  if resolved in target_formats:
    return audio_bytes, resolved

  # Need to transcode — pick the first target format (prefer wav for lossless)
  out_fmt = "wav" if "wav" in target_formats else next(iter(target_formats))

  if not is_ffmpeg_available():
    raise RuntimeError(
      f"Audio format '{source_format}' is not accepted by the target API "
      f"(accepts: {', '.join(sorted(target_formats))}). "
      f"Install ffmpeg to enable automatic transcoding, or set "
      f"audio_transcriber=True on the Agent to transcribe voice to text instead."
    )

  log_debug(f"Transcoding audio from '{source_format}' ({resolved}) to '{out_fmt}' via ffmpeg")
  return _transcode_ffmpeg(audio_bytes, resolved, out_fmt), out_fmt


# ffmpeg demuxer names — maps our canonical format names to what ffmpeg
# expects for ``-f <input_format>``.  Most match 1:1, but a few differ
# (e.g. ``m4a`` must be read as ``mp4``).
_FFMPEG_INPUT_FORMATS: Dict[str, str] = {
  "ogg": "ogg",
  "mp3": "mp3",
  "wav": "wav",
  "flac": "flac",
  "m4a": "mp4",
  "webm": "webm",
}


def _transcode_ffmpeg(audio_bytes: bytes, src_fmt: str, dst_fmt: str) -> bytes:
  """Transcode audio bytes using ffmpeg subprocess (stdin → stdout).

  Uses ``-f <demuxer>`` for the input format, resolved via
  :data:`_FFMPEG_INPUT_FORMATS`.  If the format isn't in the map we
  omit ``-f`` entirely and let ffmpeg probe the stream.
  """
  ffmpeg_src = _FFMPEG_INPUT_FORMATS.get(src_fmt)

  cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
  if ffmpeg_src:
    cmd += ["-f", ffmpeg_src]
  cmd += ["-i", "pipe:0", "-f", dst_fmt, "pipe:1"]

  result = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=30)
  if result.returncode != 0:
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    raise RuntimeError(f"ffmpeg transcoding failed ({src_fmt} → {dst_fmt}): {stderr}")
  return result.stdout


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
class AudioFileReader(BaseReader):
  """Reads audio files by transcribing them to text.

  Uses ``OpenAITranscriber`` by default. Pass a custom ``AudioTranscriber``
  implementation to use a different backend.
  """

  config: Optional[ReaderConfig] = None
  transcriber: Any = None  # set in __post_init__

  def __post_init__(self) -> None:
    if self.transcriber is None:
      self.transcriber = OpenAITranscriber()

  def supported_mime_types(self) -> List[str]:
    return list(_MIME_TO_EXT.keys())

  def supported_extensions(self) -> Set[str]:
    return {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}

  def read_file(self, file: File) -> ReaderOutput:
    try:
      raw = self._get_file_bytes(file)  # type: ignore[attr-defined]
      self._check_file_size(raw)
      mime = file.mime_type or "audio/mpeg"
      text: str = self.transcriber.transcribe(raw, mime)
      text, truncated = self._truncate(text)  # type: ignore[attr-defined]
      return ReaderOutput(
        filename=self._get_filename(file),  # type: ignore[attr-defined]
        content=text,  # type: ignore[call-arg]
        mime_type=mime,
        word_count=len(text.split()),
        truncated=truncated,
      )
    except Exception as e:
      return self._make_error_result(file, str(e))  # type: ignore[attr-defined]

  async def aread_file(self, file: File) -> ReaderOutput:
    try:
      raw = await self._aget_file_bytes(file)  # type: ignore[attr-defined]
      self._check_file_size(raw)
      mime = file.mime_type or "audio/mpeg"
      text: str = await self.transcriber.atranscribe(raw, mime)
      text, truncated = self._truncate(text)  # type: ignore[attr-defined]
      return ReaderOutput(
        filename=self._get_filename(file),  # type: ignore[attr-defined]
        content=text,  # type: ignore[call-arg]
        mime_type=mime,
        word_count=len(text.split()),
        truncated=truncated,
      )
    except Exception as e:
      return self._make_error_result(file, str(e))  # type: ignore[attr-defined]
