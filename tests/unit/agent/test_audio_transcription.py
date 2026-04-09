"""
Unit tests for Agent-level audio transcription.

Tests that audio in incoming messages is transcribed to text before
reaching the model, when an audio_transcriber is configured.
"""

import pytest

from definable.agent.agent import Agent
from definable.agent.testing import MockModel
from definable.media import Audio
from definable.model.message import Message
from definable.reader.audio import AudioTranscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeTranscriber:
  """Deterministic transcriber for tests — returns canned text."""

  def __init__(self, text: str = "hello from voice"):
    self._text = text
    self.calls: list[tuple[bytes, str]] = []

  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    self.calls.append((audio_bytes, mime_type))
    return self._text

  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    self.calls.append((audio_bytes, mime_type))
    return self._text


class FailingTranscriber:
  """Transcriber that always raises."""

  def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    raise RuntimeError("Whisper unavailable")

  async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
    raise RuntimeError("Whisper unavailable")


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioTranscriberInit:
  def test_default_is_none(self):
    """No audio_transcriber → _audio_transcriber is None."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    assert agent._audio_transcriber is None

  def test_true_resolves_to_openai_transcriber(self):
    """audio_transcriber=True → OpenAITranscriber instance."""
    from definable.reader.audio import OpenAITranscriber

    agent = Agent(model=MockModel(), audio_transcriber=True)  # type: ignore[arg-type]
    assert isinstance(agent._audio_transcriber, OpenAITranscriber)

  def test_custom_transcriber_stored(self):
    """audio_transcriber=FakeTranscriber() → stores the instance."""
    transcriber = FakeTranscriber()
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]
    assert agent._audio_transcriber is transcriber

  def test_protocol_conformance(self):
    """FakeTranscriber satisfies the AudioTranscriber protocol."""
    assert isinstance(FakeTranscriber(), AudioTranscriber)


# ---------------------------------------------------------------------------
# _transcribe_audio tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscribeAudioMethod:
  @pytest.mark.asyncio
  async def test_no_transcriber_is_noop(self):
    """When no transcriber is set, messages are untouched."""
    agent = Agent(model=MockModel())  # type: ignore[arg-type]
    audio = Audio(content=b"fake-ogg-data", mime_type="audio/ogg")
    msg = Message(role="user", content="", audio=[audio])
    await agent._transcribe_audio([msg])
    assert audio.transcript is None
    assert msg.content == ""
    assert msg.audio is not None  # audio preserved for audio-capable models

  @pytest.mark.asyncio
  async def test_transcribes_audio_content(self):
    """Transcriber receives audio bytes and sets transcript."""
    transcriber = FakeTranscriber("I need help with my code")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio = Audio(content=b"fake-ogg-data", mime_type="audio/ogg")
    msg = Message(role="user", content="", audio=[audio])
    await agent._transcribe_audio([msg])

    assert audio.transcript == "I need help with my code"
    assert msg.content == "I need help with my code"
    assert msg.audio is None  # cleared so model doesn't send input_audio blocks
    assert len(transcriber.calls) == 1
    assert transcriber.calls[0] == (b"fake-ogg-data", "audio/ogg")

  @pytest.mark.asyncio
  async def test_appends_to_existing_text(self):
    """When message has text + audio, transcript is appended."""
    transcriber = FakeTranscriber("the voice part")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio = Audio(content=b"data", mime_type="audio/ogg")
    msg = Message(role="user", content="caption text", audio=[audio])
    await agent._transcribe_audio([msg])

    assert msg.content == "caption text\n\nthe voice part"
    assert msg.audio is None

  @pytest.mark.asyncio
  async def test_skips_already_transcribed(self):
    """Audio with existing transcript is not re-transcribed."""
    transcriber = FakeTranscriber("should not appear")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio = Audio(content=b"data", mime_type="audio/ogg", transcript="already done")
    msg = Message(role="user", content="", audio=[audio])
    await agent._transcribe_audio([msg])

    assert audio.transcript == "already done"
    assert msg.content == "already done"
    assert len(transcriber.calls) == 0  # not called

  @pytest.mark.asyncio
  async def test_multiple_audio_items(self):
    """Multiple audio items are all transcribed and joined."""
    call_count = 0

    class CountingTranscriber:
      def transcribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
        return f"part-{len(audio_bytes)}"

      async def atranscribe(self, audio_bytes: bytes, mime_type: str, **kwargs) -> str:
        nonlocal call_count
        call_count += 1
        return f"part-{call_count}"

    agent = Agent(model=MockModel(), audio_transcriber=CountingTranscriber())  # type: ignore[arg-type]

    audio1 = Audio(content=b"aaa", mime_type="audio/ogg")
    audio2 = Audio(content=b"bbb", mime_type="audio/mp3")
    msg = Message(role="user", content="", audio=[audio1, audio2])
    await agent._transcribe_audio([msg])

    assert audio1.transcript == "part-1"
    assert audio2.transcript == "part-2"
    assert msg.content == "part-1\npart-2"

  @pytest.mark.asyncio
  async def test_defaults_mime_type_to_ogg(self):
    """When audio has no mime_type, defaults to audio/ogg."""
    transcriber = FakeTranscriber("text")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio = Audio(content=b"data")
    msg = Message(role="user", content="", audio=[audio])
    await agent._transcribe_audio([msg])

    assert transcriber.calls[0][1] == "audio/ogg"

  @pytest.mark.asyncio
  async def test_no_audio_in_message_is_noop(self):
    """Messages without audio are untouched."""
    transcriber = FakeTranscriber("text")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    msg = Message(role="user", content="just text")
    await agent._transcribe_audio([msg])

    assert msg.content == "just text"
    assert len(transcriber.calls) == 0

  @pytest.mark.asyncio
  async def test_transcription_failure_logs_and_continues(self):
    """If transcription fails, the audio is skipped without crashing."""
    agent = Agent(model=MockModel(), audio_transcriber=FailingTranscriber())  # type: ignore[arg-type]

    audio = Audio(content=b"data", mime_type="audio/ogg")
    msg = Message(role="user", content="hello", audio=[audio])
    await agent._transcribe_audio([msg])

    assert audio.transcript is None
    assert msg.content == "hello"  # unchanged
    assert msg.audio is not None  # audio preserved — transcription failed, don't drop data

  @pytest.mark.asyncio
  async def test_audio_without_bytes_skipped(self):
    """Audio with no downloadable content is skipped gracefully."""
    transcriber = FakeTranscriber("text")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    # Audio with a URL that can't be downloaded — get_content_bytes returns None
    audio = Audio.model_construct(url=None, filepath=None, content=None, id="test")
    msg = Message(role="user", content="", audio=[audio])
    await agent._transcribe_audio([msg])

    assert len(transcriber.calls) == 0

  @pytest.mark.asyncio
  async def test_mixed_transcribed_and_new_audio(self):
    """Mix of already-transcribed and new audio items works correctly."""
    transcriber = FakeTranscriber("new transcript")
    agent = Agent(model=MockModel(), audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio1 = Audio(content=b"old", mime_type="audio/ogg", transcript="existing")
    audio2 = Audio(content=b"new", mime_type="audio/ogg")
    msg = Message(role="user", content="", audio=[audio1, audio2])
    await agent._transcribe_audio([msg])

    assert audio1.transcript == "existing"
    assert audio2.transcript == "new transcript"
    assert msg.content == "existing\nnew transcript"
    assert len(transcriber.calls) == 1  # only called for audio2


# ---------------------------------------------------------------------------
# End-to-end: arun with audio transcription
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArunWithAudioTranscription:
  @pytest.mark.asyncio
  async def test_arun_transcribes_voice_note(self):
    """agent.arun() with audio + transcriber produces text the model can see."""
    transcriber = FakeTranscriber("please help me debug this")
    model = MockModel(responses=["Sure, I'll help you debug."])
    agent = Agent(model=model, audio_transcriber=transcriber)  # type: ignore[arg-type]

    audio = Audio(content=b"ogg-voice-data", mime_type="audio/ogg")
    result = await agent.arun("", audio=[audio])

    # Model received the transcribed text
    assert result.content == "Sure, I'll help you debug."
    assert len(transcriber.calls) == 1

    # The user message in history contains the transcript, audio cleared
    assert result.messages is not None
    user_messages = [m for m in result.messages if m.role == "user"]
    assert len(user_messages) >= 1
    assert "please help me debug this" in (user_messages[-1].content or "")
    assert user_messages[-1].audio is None  # cleared after transcription

  @pytest.mark.asyncio
  async def test_arun_without_transcriber_passes_raw_audio(self):
    """Without transcriber, audio passes through to model unchanged."""
    model = MockModel(responses=["I can see audio."])
    agent = Agent(model=model)  # type: ignore[arg-type]

    audio = Audio(content=b"ogg-voice-data", mime_type="audio/ogg")
    result = await agent.arun("analyze this", audio=[audio])

    assert result.content == "I can see audio."
    # Audio is on the message but transcript is not set
    assert result.messages is not None
    user_messages = [m for m in result.messages if m.role == "user"]
    assert user_messages[-1].audio is not None
    assert user_messages[-1].audio[0].transcript is None


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscriberImports:
  def test_import_from_reader(self):
    """AudioTranscriber and OpenAITranscriber importable from reader.audio."""
    from definable.reader.audio import AudioTranscriber, OpenAITranscriber

    assert AudioTranscriber is not None
    assert OpenAITranscriber is not None

  def test_import_from_top_level(self):
    """OpenAITranscriber importable from definable top-level."""
    from definable import OpenAITranscriber

    assert OpenAITranscriber is not None

  def test_import_from_reader_package(self):
    """OpenAITranscriber importable from definable.reader."""
    from definable.reader import OpenAITranscriber

    assert OpenAITranscriber is not None
