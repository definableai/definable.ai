"""
Unit tests for audio format normalization and transcoding.

Tests the normalize_audio_format() function in reader/audio.py
that converts platform-native audio formats (Telegram OGA, Discord OGG)
to formats accepted by target APIs (OpenAI input_audio: wav/mp3).
"""

import subprocess
from unittest.mock import patch

import pytest

from definable.reader.audio import (
  OPENAI_INPUT_AUDIO_FORMATS,
  _EXT_ALIASES,
  _MIME_TO_EXT,
  is_ffmpeg_available,
  normalize_audio_format,
)


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioConstants:
  def test_openai_formats_are_wav_and_mp3(self):
    """OpenAI input_audio API accepts only wav and mp3."""
    assert OPENAI_INPUT_AUDIO_FORMATS == {"wav", "mp3"}

  def test_ext_aliases_cover_telegram_oga(self):
    """Telegram's .oga extension maps to ogg."""
    assert _EXT_ALIASES["oga"] == "ogg"

  def test_ext_aliases_cover_opus(self):
    """Raw opus extension maps to ogg."""
    assert _EXT_ALIASES["opus"] == "ogg"

  def test_ext_aliases_cover_mpeg_variants(self):
    """MPEG audio aliases map to mp3."""
    assert _EXT_ALIASES["mpga"] == "mp3"
    assert _EXT_ALIASES["mpeg"] == "mp3"

  def test_mime_to_ext_includes_ogg_variants(self):
    """MIME mapping covers standard OGG types."""
    assert _MIME_TO_EXT["audio/ogg"] == "ogg"
    assert _MIME_TO_EXT["audio/x-ogg"] == "ogg"
    assert _MIME_TO_EXT["audio/opus"] == "ogg"


# ---------------------------------------------------------------------------
# normalize_audio_format — passthrough cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizePassthrough:
  def test_wav_passes_through(self):
    """WAV format is already accepted — returns unchanged."""
    data = b"RIFF\x00\x00\x00\x00WAVEfmt "
    out_bytes, out_fmt = normalize_audio_format(data, "wav")
    assert out_bytes is data
    assert out_fmt == "wav"

  def test_mp3_passes_through(self):
    """MP3 format is already accepted — returns unchanged."""
    data = b"\xff\xfb\x90\x00"
    out_bytes, out_fmt = normalize_audio_format(data, "mp3")
    assert out_bytes is data
    assert out_fmt == "mp3"

  def test_mpga_alias_resolves_to_mp3(self):
    """mpga alias resolves to mp3 — passes through."""
    data = b"\xff\xfb\x90\x00"
    out_bytes, out_fmt = normalize_audio_format(data, "mpga")
    assert out_bytes is data
    assert out_fmt == "mp3"

  def test_mpeg_alias_resolves_to_mp3(self):
    """mpeg alias resolves to mp3 — passes through."""
    data = b"\xff\xfb\x90\x00"
    out_bytes, out_fmt = normalize_audio_format(data, "mpeg")
    assert out_bytes is data
    assert out_fmt == "mp3"

  def test_case_insensitive(self):
    """Format comparison is case-insensitive."""
    data = b"test"
    out_bytes, out_fmt = normalize_audio_format(data, "WAV")
    assert out_bytes is data
    assert out_fmt == "wav"

  def test_custom_target_formats(self):
    """Custom target_formats set is respected."""
    data = b"test"
    out_bytes, out_fmt = normalize_audio_format(data, "ogg", target_formats={"ogg", "wav"})
    assert out_bytes is data
    assert out_fmt == "ogg"


# ---------------------------------------------------------------------------
# normalize_audio_format — transcoding cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeTranscode:
  def test_oga_requires_transcoding(self):
    """OGA (Telegram voice) needs transcoding — not in OpenAI accepted set."""
    # oga → alias to ogg → ogg not in {wav, mp3} → needs ffmpeg
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      with pytest.raises(RuntimeError, match="Audio format 'oga' is not accepted"):
        normalize_audio_format(b"fake-ogg-data", "oga")

  def test_ogg_requires_transcoding(self):
    """OGG needs transcoding — not in OpenAI accepted set."""
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      with pytest.raises(RuntimeError, match="Audio format 'ogg' is not accepted"):
        normalize_audio_format(b"fake-ogg-data", "ogg")

  def test_flac_requires_transcoding(self):
    """FLAC needs transcoding — not in OpenAI accepted set."""
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      with pytest.raises(RuntimeError, match="Audio format 'flac' is not accepted"):
        normalize_audio_format(b"fake-flac-data", "flac")

  def test_error_message_suggests_ffmpeg(self):
    """Error message tells user to install ffmpeg."""
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      with pytest.raises(RuntimeError, match="Install ffmpeg"):
        normalize_audio_format(b"data", "ogg")

  def test_error_message_suggests_transcriber(self):
    """Error message tells user about audio_transcriber alternative."""
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      with pytest.raises(RuntimeError, match="audio_transcriber=True"):
        normalize_audio_format(b"data", "ogg")

  def test_ffmpeg_called_when_available(self):
    """When ffmpeg is available, transcoding is attempted."""
    fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    with (
      patch("definable.reader.audio.is_ffmpeg_available", return_value=True),
      patch("definable.reader.audio._transcode_ffmpeg", return_value=fake_wav) as mock_transcode,
    ):
      out_bytes, out_fmt = normalize_audio_format(b"ogg-data", "oga")
      assert out_bytes == fake_wav
      assert out_fmt == "wav"
      mock_transcode.assert_called_once_with(b"ogg-data", "ogg", "wav")  # resolved alias

  def test_transcoding_prefers_wav(self):
    """When both wav and mp3 are targets, wav is preferred."""
    with (
      patch("definable.reader.audio.is_ffmpeg_available", return_value=True),
      patch("definable.reader.audio._transcode_ffmpeg", return_value=b"wav-data") as mock_transcode,
    ):
      _, out_fmt = normalize_audio_format(b"data", "ogg")
      assert out_fmt == "wav"
      assert mock_transcode.call_args[0][2] == "wav"  # dst_fmt


# ---------------------------------------------------------------------------
# _transcode_ffmpeg
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscodeFfmpeg:
  def test_ffmpeg_uses_demuxer_name_for_known_format(self):
    """For known formats (ogg), ffmpeg receives the correct -f demuxer name."""
    with patch("definable.reader.audio.subprocess.run") as mock_run:
      mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"wav-output", stderr=b"")
      from definable.reader.audio import _transcode_ffmpeg

      result = _transcode_ffmpeg(b"input-data", "ogg", "wav")
      assert result == b"wav-output"

      cmd = mock_run.call_args[0][0]
      assert cmd[0] == "ffmpeg"
      # Should use -f ogg (the ffmpeg demuxer name)
      f_idx = cmd.index("-f")
      assert cmd[f_idx + 1] == "ogg"
      assert "pipe:0" in cmd
      assert "pipe:1" in cmd
      assert mock_run.call_args[1]["input"] == b"input-data"

  def test_ffmpeg_omits_input_format_for_unknown(self):
    """For unknown formats, -f is omitted so ffmpeg probes the stream."""
    with patch("definable.reader.audio.subprocess.run") as mock_run:
      mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"wav-output", stderr=b"")
      from definable.reader.audio import _transcode_ffmpeg

      _transcode_ffmpeg(b"input-data", "unknown_fmt", "wav")

      cmd = mock_run.call_args[0][0]
      # -f should only appear once (for the output), not for input
      f_positions = [i for i, c in enumerate(cmd) if c == "-f"]
      # The only -f should be right before the output format
      assert len(f_positions) == 1
      assert cmd[f_positions[0] + 1] == "wav"  # output format

  def test_ffmpeg_failure_raises_runtime_error(self):
    """Non-zero ffmpeg exit raises RuntimeError with stderr."""
    with patch("definable.reader.audio.subprocess.run") as mock_run:
      mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"Unknown format")
      from definable.reader.audio import _transcode_ffmpeg

      with pytest.raises(RuntimeError, match="ffmpeg transcoding failed"):
        _transcode_ffmpeg(b"bad-data", "xyz", "wav")


# ---------------------------------------------------------------------------
# is_ffmpeg_available
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFfmpegDetection:
  def test_returns_true_when_on_path(self):
    """Returns True when ffmpeg is found on PATH."""
    with patch("definable.reader.audio.shutil.which", return_value="/usr/bin/ffmpeg"):
      assert is_ffmpeg_available() is True

  def test_returns_false_when_not_on_path(self):
    """Returns False when ffmpeg is not found."""
    with patch("definable.reader.audio.shutil.which", return_value=None):
      assert is_ffmpeg_available() is False


# ---------------------------------------------------------------------------
# Integration: audio_to_message with format normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioToMessageNormalization:
  def test_wav_audio_passes_through(self):
    """WAV audio in audio_to_message goes through without transcoding."""
    from definable.media import Audio
    from definable.utils.openai import audio_to_message

    audio = Audio(content=b"RIFF\x00\x00\x00\x00WAVEfmt ", format="wav")
    result = audio_to_message([audio])
    assert len(result) == 1
    assert result[0]["input_audio"]["format"] == "wav"

  def test_mp3_audio_passes_through(self):
    """MP3 audio in audio_to_message goes through without transcoding."""
    from definable.media import Audio
    from definable.utils.openai import audio_to_message

    audio = Audio(content=b"\xff\xfb\x90\x00", format="mp3")
    result = audio_to_message([audio])
    assert len(result) == 1
    assert result[0]["input_audio"]["format"] == "mp3"

  def test_oga_audio_triggers_normalization(self):
    """OGA audio triggers format normalization in audio_to_message."""
    from definable.media import Audio
    from definable.utils.openai import audio_to_message

    audio = Audio(content=b"ogg-opus-data", format="oga")
    with (
      patch("definable.reader.audio.is_ffmpeg_available", return_value=True),
      patch("definable.reader.audio._transcode_ffmpeg", return_value=b"wav-data"),
    ):
      result = audio_to_message([audio])
      assert len(result) == 1
      assert result[0]["input_audio"]["format"] == "wav"

  def test_oga_without_ffmpeg_skips_audio(self):
    """OGA without ffmpeg logs error and skips the audio snippet."""
    from definable.media import Audio
    from definable.utils.openai import audio_to_message

    audio = Audio(content=b"ogg-opus-data", format="oga")
    with patch("definable.reader.audio.is_ffmpeg_available", return_value=False):
      result = audio_to_message([audio])
      assert len(result) == 0  # skipped, not crashed


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioFormatImports:
  def test_normalize_from_reader_audio(self):
    from definable.reader.audio import normalize_audio_format

    assert callable(normalize_audio_format)

  def test_normalize_from_reader_package(self):
    from definable.reader import normalize_audio_format

    assert callable(normalize_audio_format)

  def test_constants_from_reader_package(self):
    from definable.reader import OPENAI_INPUT_AUDIO_FORMATS

    assert isinstance(OPENAI_INPUT_AUDIO_FORMATS, set)
