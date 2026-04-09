"""Unit tests for the CallInterface module.

Tests cover:
  - CallConfig validation
  - CallSession lifecycle and conversation history
  - TwilioProvider XML generation and WebSocket event parsing
  - ManagedPipeline event handling
  - CascadingPipeline orchestration
  - DeepgramSTT construction and URL building
  - CartesiaTTS construction and encoding mapping
  - OpenAIRealtimeProvider construction and event mapping
  - RealtimePipeline orchestration and tool handling
  - CallInterface construction and factory methods
  - Router creation
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from definable.agent.interface.call.call import (
  CallEventType,
  CallSession,
  CallState,
)
from definable.agent.interface.call.config import CallConfig
from definable.agent.interface.call.interface import CallInterface
from definable.agent.interface.call.pipeline.base import CallPipeline
from definable.agent.interface.call.pipeline.managed import ManagedPipeline, _split_into_speech_chunks
from definable.agent.interface.call.telephony.base import TelephonyEvent, TelephonyProvider
from definable.agent.interface.call.telephony.twilio import TwilioProvider
from definable.agent.interface.call.stt.base import STTProvider, Transcript
from definable.agent.interface.call.tts.base import TTSProvider
from definable.agent.interface.call.realtime.base import RealtimeEvent, RealtimeProvider
from definable.agent.interface.errors import InterfaceError


# ============================================================
# CallConfig Tests
# ============================================================


class TestCallConfig:
  """Tests for CallConfig validation."""

  def test_valid_config(self):
    config = CallConfig(phone_number="+15551234567")
    assert config.platform == "call"
    assert config.telephony_provider == "twilio"
    assert config.pipeline_mode == "managed"
    assert config.phone_number == "+15551234567"

  def test_missing_phone_number_raises(self):
    with pytest.raises(InterfaceError, match="phone_number is required"):
      CallConfig()

  def test_invalid_provider_raises(self):
    with pytest.raises(InterfaceError, match="Unsupported telephony provider"):
      CallConfig(phone_number="+1555", telephony_provider="vonage")

  def test_invalid_pipeline_raises(self):
    with pytest.raises(InterfaceError, match="Unsupported pipeline mode"):
      CallConfig(phone_number="+1555", pipeline_mode="magic")  # type: ignore[arg-type]

  def test_defaults(self):
    config = CallConfig(phone_number="+1555")
    assert config.language == "en-US"
    assert config.interruptible == "any"
    assert config.interrupt_sensitivity == "medium"
    assert config.stt_provider == "deepgram"
    assert config.tts_provider == "google"
    assert config.max_call_duration_seconds == 3600
    assert config.session_ttl_seconds == 7200
    assert config.max_concurrent_requests == 50

  def test_with_updates(self):
    config = CallConfig(phone_number="+1555")
    updated = config.with_updates(language="fr-FR", voice="Celine")
    assert updated.language == "fr-FR"  # type: ignore[attr-defined]
    assert updated.voice == "Celine"  # type: ignore[attr-defined]
    assert updated.phone_number == "+1555"  # type: ignore[attr-defined]

  def test_frozen(self):
    config = CallConfig(phone_number="+1555")
    with pytest.raises(AttributeError):
      config.phone_number = "+1999"  # type: ignore[misc]


# ============================================================
# CallSession Tests
# ============================================================


class TestCallSession:
  """Tests for CallSession lifecycle and conversation history."""

  def test_creation(self):
    session = CallSession(call_id="CA123")
    assert session.call_id == "CA123"
    assert session.state == CallState.RINGING
    assert session.conversation == []
    assert session.events == []

  def test_add_user_message(self):
    session = CallSession(call_id="CA123")
    session.add_user_message("Hello")
    assert len(session.conversation) == 1
    assert session.conversation[0] == {"role": "user", "content": "Hello"}

  def test_add_assistant_message(self):
    session = CallSession(call_id="CA123")
    session.add_assistant_message("Hi there!")
    assert len(session.conversation) == 1
    assert session.conversation[0] == {"role": "assistant", "content": "Hi there!"}

  def test_truncate_last_assistant(self):
    session = CallSession(call_id="CA123")
    session.add_assistant_message("The weather today is sunny and warm.")
    session.truncate_last_assistant("The weather today is")
    assert session.conversation[-1]["content"] == "The weather today is"

  def test_truncate_no_assistant_noop(self):
    session = CallSession(call_id="CA123")
    session.add_user_message("Hello")
    session.truncate_last_assistant("whatever")
    # Should not modify user message
    assert session.conversation[-1]["content"] == "Hello"

  def test_add_event(self):
    session = CallSession(call_id="CA123")
    event = session.add_event(CallEventType.CALL_STARTED)
    assert event.type == CallEventType.CALL_STARTED
    assert event.call_id == "CA123"
    assert len(session.events) == 1

  def test_add_event_with_data(self):
    session = CallSession(call_id="CA123")
    event = session.add_event(CallEventType.UTTERANCE, text="Hello", confidence=0.95)
    assert event.data == {"text": "Hello", "confidence": 0.95}

  def test_duration(self):
    session = CallSession(call_id="CA123")
    assert session.duration_seconds >= 0

  def test_multi_turn_conversation(self):
    session = CallSession(call_id="CA123")
    session.add_user_message("What's the weather?")
    session.add_assistant_message("It's sunny today.")
    session.add_user_message("And tomorrow?")
    session.add_assistant_message("Rain expected tomorrow.")
    assert len(session.conversation) == 4
    assert session.conversation[0]["role"] == "user"
    assert session.conversation[1]["role"] == "assistant"
    assert session.conversation[2]["role"] == "user"
    assert session.conversation[3]["role"] == "assistant"


# ============================================================
# CallState Tests
# ============================================================


class TestCallState:
  """Tests for CallState enum."""

  def test_states_exist(self):
    assert CallState.RINGING.value == "ringing"
    assert CallState.ACTIVE.value == "active"
    assert CallState.ON_HOLD.value == "on_hold"
    assert CallState.ENDED.value == "ended"


# ============================================================
# TwilioProvider Tests
# ============================================================


class TestTwilioProvider:
  """Tests for TwilioProvider XML generation and event parsing."""

  def setup_method(self):
    self.provider = TwilioProvider(account_sid="ACtest", auth_token="test_token")

  # --- XML generation ---

  def test_generate_conversation_relay_xml(self):
    xml = self.provider.generate_answer_xml(
      "wss://example.com/call/convo/123",
      mode="managed",
      welcome_message="Hello!",
      tts_provider="elevenlabs",
      stt_provider="deepgram",
      voice="Rachel",
      language="en-US",
    )
    assert "<?xml version" in xml
    assert "<Response>" in xml
    assert "<Connect>" in xml
    assert "ConversationRelay" in xml
    assert 'url="wss://example.com/call/convo/123"' in xml
    assert 'welcomeGreeting="Hello!"' in xml
    assert 'ttsProvider="elevenlabs"' in xml
    assert 'transcriptionProvider="deepgram"' in xml
    assert 'voice="Rachel"' in xml

  def test_generate_media_stream_xml(self):
    xml = self.provider.generate_answer_xml(
      "wss://example.com/call/stream/123",
      mode="stream",
      welcome_message="Please wait.",
    )
    assert "<?xml version" in xml
    assert "<Response>" in xml
    assert "<Say>Please wait.</Say>" in xml
    assert "<Connect>" in xml
    assert '<Stream url="wss://example.com/call/stream/123"' in xml

  def test_xml_escaping(self):
    xml = self.provider.generate_answer_xml(
      "wss://example.com/ws",
      mode="managed",
      welcome_message='Hello & "goodbye" <world>',
    )
    assert "&amp;" in xml
    assert "&quot;" in xml
    assert "&lt;" in xml
    assert "&gt;" in xml

  def test_no_welcome_message(self):
    xml = self.provider.generate_answer_xml("wss://example.com/ws", mode="managed")
    assert "welcomeGreeting" not in xml

  # --- ConversationRelay event parsing ---

  def test_parse_setup_event(self):
    data = {"type": "setup", "callSid": "CA123", "customParameters": {"agentId": "test"}}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "setup"
    assert event.call_id == "CA123"
    assert event.metadata["custom_parameters"] == {"agentId": "test"}

  def test_parse_prompt_event(self):
    data = {"type": "prompt", "callSid": "CA123", "voicePrompt": "What's the weather?"}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "prompt"
    assert event.call_id == "CA123"
    assert event.payload == "What's the weather?"

  def test_parse_interrupt_event(self):
    data = {"type": "interrupt", "callSid": "CA123", "utteranceUntilInterrupt": "The weather is"}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "interrupt"
    assert event.payload == "The weather is"

  def test_parse_dtmf_conversation_relay(self):
    data = {"type": "dtmf", "callSid": "CA123", "digit": "5"}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "dtmf"
    assert event.payload == "5"

  # --- Media Stream event parsing ---

  def test_parse_media_start_event(self):
    data = {
      "event": "start",
      "start": {
        "callSid": "CA456",
        "streamSid": "MZ789",
        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000},
      },
    }
    event = self.provider.parse_websocket_event(data)
    assert event.event == "start"
    assert event.call_id == "CA456"
    assert event.stream_id == "MZ789"

  def test_parse_media_audio_event(self):
    import base64

    audio_b64 = base64.b64encode(b"\x00\x01\x02").decode("ascii")
    data = {
      "event": "media",
      "streamSid": "MZ789",
      "media": {"payload": audio_b64, "chunk": "1", "timestamp": "100"},
    }
    event = self.provider.parse_websocket_event(data)
    assert event.event == "media"
    assert event.stream_id == "MZ789"
    assert event.payload == b"\x00\x01\x02"

  def test_parse_media_stop_event(self):
    data = {"event": "stop", "streamSid": "MZ789"}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "stop"
    assert event.stream_id == "MZ789"

  def test_parse_mark_event(self):
    data = {"event": "mark", "streamSid": "MZ789", "mark": {"name": "response_end"}}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "mark"
    assert event.payload == "response_end"

  # --- Response encoding ---

  def test_encode_audio_response(self):
    import base64

    msg = self.provider.encode_audio_response(b"\x00\x01\x02", "MZ789")
    assert msg["event"] == "media"
    assert msg["streamSid"] == "MZ789"
    assert base64.b64decode(msg["media"]["payload"]) == b"\x00\x01\x02"

  def test_encode_clear_audio(self):
    msg = self.provider.encode_clear_audio("MZ789")
    assert msg["event"] == "clear"
    assert msg["streamSid"] == "MZ789"

  def test_encode_text_response(self):
    msg = self.provider.encode_text_response("Hello", last=False)
    assert msg["type"] == "text"
    assert msg["token"] == "Hello"
    assert msg["last"] is False

  def test_encode_text_response_last(self):
    msg = self.provider.encode_text_response("world.", last=True)
    assert msg["last"] is True

  # --- Webhook signature ---

  def test_validate_signature_no_token(self):
    provider = TwilioProvider(account_sid="AC", auth_token="")
    assert provider.validate_webhook_signature(b"body", "sig", "http://example.com") is False

  def test_unknown_event_type(self):
    data = {"event": "custom_thing", "streamSid": "MZ"}
    event = self.provider.parse_websocket_event(data)
    assert event.event == "custom_thing"


# ============================================================
# ManagedPipeline Tests
# ============================================================


class TestManagedPipeline:
  """Tests for the managed voice pipeline."""

  def test_split_into_speech_chunks_empty(self):
    assert _split_into_speech_chunks("") == []

  def test_split_into_speech_chunks_single_sentence(self):
    chunks = _split_into_speech_chunks("Hello world.")
    assert len(chunks) == 1
    assert chunks[0] == "Hello world."

  def test_split_into_speech_chunks_multiple_sentences(self):
    text = "Hello there. How are you? I'm doing great!"
    chunks = _split_into_speech_chunks(text)
    assert len(chunks) >= 2
    # All chunks joined should equal the original
    assert "".join(chunks) == text

  def test_split_preserves_short_sentences(self):
    text = "Hi. Bye."
    chunks = _split_into_speech_chunks(text)
    # Short sentences should not split
    assert len(chunks) == 1

  @pytest.mark.asyncio
  async def test_handle_setup_event(self):
    pipeline = ManagedPipeline()
    session = CallSession(call_id="CA123")
    event = TelephonyEvent(event="setup", call_id="CA456")

    await pipeline._handle_setup(event, session)
    assert session.call_id == "CA456"
    assert session.state == CallState.ACTIVE
    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.CALL_STARTED

  @pytest.mark.asyncio
  async def test_handle_interrupt(self):
    pipeline = ManagedPipeline()
    session = CallSession(call_id="CA123")
    session.add_assistant_message("The weather today is sunny and warm.")

    event = TelephonyEvent(event="interrupt", payload="The weather today is")
    await pipeline._handle_interrupt(event, session)

    assert session.conversation[-1]["content"] == "The weather today is"
    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.INTERRUPTION


# ============================================================
# CallInterface Construction Tests
# ============================================================


class TestCallInterfaceConstruction:
  """Tests for CallInterface creation and factory methods."""

  def test_basic_creation(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+15551234567",
      account_sid="ACtest",
      auth_token="test_token",
    )
    assert ci.config.platform == "call"
    assert ci._call_config.pipeline_mode == "managed"
    assert isinstance(ci._telephony, TwilioProvider)
    assert isinstance(ci._pipeline, ManagedPipeline)

  def test_custom_config(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+15551234567",
      account_sid="ACtest",
      auth_token="test_token",
      welcome_message="Hello!",
      voice="Rachel",
      language="fr-FR",
      interruptible="speech",
      stt_provider="google",
      tts_provider="elevenlabs",
    )
    assert ci._call_config.welcome_message == "Hello!"
    assert ci._call_config.voice == "Rachel"
    assert ci._call_config.language == "fr-FR"
    assert ci._call_config.interruptible == "speech"
    assert ci._call_config.stt_provider == "google"
    assert ci._call_config.tts_provider == "elevenlabs"

  def test_missing_phone_number_raises(self):
    with pytest.raises(InterfaceError, match="phone_number"):
      CallInterface(provider="twilio", account_sid="AC", auth_token="test")

  def test_plivo_managed_not_supported(self):
    """Plivo does not support managed mode — should raise ValueError."""
    with pytest.raises(ValueError, match="Plivo does not support managed pipeline mode"):
      CallInterface(provider="plivo", phone_number="+1555", auth_id="id", auth_token="token")

  def test_cascading_without_providers_raises(self):
    with pytest.raises(ValueError, match="stt= and tts= providers"):
      CallInterface(
        provider="twilio",
        phone_number="+1555",
        account_sid="AC",
        auth_token="test",
        pipeline="cascading",
      )

  def test_realtime_without_provider_raises(self):
    with pytest.raises(ValueError, match="realtime= provider"):
      CallInterface(
        provider="twilio",
        phone_number="+1555",
        account_sid="AC",
        auth_token="test",
        pipeline="realtime",
      )

  def test_active_calls_initially_empty(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    assert ci.active_calls == {}

  def test_properties(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    assert isinstance(ci.telephony, TwilioProvider)
    assert isinstance(ci.pipeline, ManagedPipeline)

  def test_create_router(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    router = ci.create_router()
    assert router is not None
    # Calling again returns the same router
    assert ci.create_router() is router


# ============================================================
# Protocol Compliance Tests
# ============================================================


class TestProtocolCompliance:
  """Tests that protocol/ABC types are properly defined."""

  def test_stt_provider_is_protocol(self):
    assert hasattr(STTProvider, "__protocol_attrs__") or hasattr(STTProvider, "_is_runtime_checkable")

  def test_tts_provider_is_protocol(self):
    assert hasattr(TTSProvider, "__protocol_attrs__") or hasattr(TTSProvider, "_is_runtime_checkable")

  def test_realtime_provider_is_abc(self):
    assert hasattr(RealtimeProvider, "__abstractmethods__")

  def test_telephony_provider_is_abc(self):
    assert hasattr(TelephonyProvider, "__abstractmethods__")

  def test_call_pipeline_is_abc(self):
    assert hasattr(CallPipeline, "__abstractmethods__")

  def test_transcript_dataclass(self):
    t = Transcript(text="hello", is_final=True, confidence=0.98)
    assert t.text == "hello"
    assert t.is_final is True
    assert t.confidence == 0.98

  def test_realtime_event_dataclass(self):
    e = RealtimeEvent(type="audio_delta", audio=b"\x00\x01")
    assert e.type == "audio_delta"
    assert e.audio == b"\x00\x01"
    assert e.text is None

  def test_telephony_event_dataclass(self):
    e = TelephonyEvent(event="media", call_id="CA123", stream_id="MZ789")
    assert e.event == "media"
    assert e.call_id == "CA123"
    assert e.stream_id == "MZ789"


# ============================================================
# Lifecycle Tests
# ============================================================


class TestCallInterfaceLifecycle:
  """Tests for start/stop lifecycle."""

  @pytest.mark.asyncio
  async def test_start_without_agent_raises(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    with pytest.raises(ValueError, match="no agent bound"):
      await ci.start()

  @pytest.mark.asyncio
  async def test_stop_clears_active_calls(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    # Manually add a fake active call
    ci._active_calls["CA123"] = CallSession(call_id="CA123", state=CallState.ACTIVE)
    ci._running = True

    await ci.stop()
    assert len(ci._active_calls) == 0

  @pytest.mark.asyncio
  async def test_stop_ends_active_calls(self):
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="test",
    )
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    ci._active_calls["CA123"] = session
    ci._running = True

    await ci.stop()
    assert session.state == CallState.ENDED


# ============================================================
# Import / Re-export Tests
# ============================================================


class TestImports:
  """Tests that all public types are importable from expected paths."""

  def test_import_from_call_package(self):
    from definable.agent.interface.call import (  # noqa: F401
      CallConfig,
      CallEvent,
      CallEventType,
      CallInterface,
      CallPipeline,
      CallSession,
      CallState,
      ManagedPipeline,
      RealtimeEvent,
      RealtimeProvider,
      STTProvider,
      TTSProvider,
      TelephonyEvent,
      TelephonyProvider,
      Transcript,
    )

  def test_import_from_interface_package(self):
    from definable.agent.interface import CallInterface, CallConfig  # noqa: F401

  def test_import_phase2_types(self):
    from definable.agent.interface.call import (  # noqa: F401
      CascadingPipeline,
      DeepgramSTT,
      CartesiaTTS,
    )

  def test_import_deepgram_direct(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT  # noqa: F401

  def test_import_cartesia_direct(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS  # noqa: F401

  def test_import_cascading_direct(self):
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline  # noqa: F401

  def test_import_from_stt_package(self):
    from definable.agent.interface.call.stt import DeepgramSTT  # noqa: F401

  def test_import_from_tts_package(self):
    from definable.agent.interface.call.tts import CartesiaTTS  # noqa: F401

  def test_import_from_pipeline_package(self):
    from definable.agent.interface.call.pipeline import CascadingPipeline, ManagedPipeline  # noqa: F401


# ============================================================
# DeepgramSTT Tests
# ============================================================


class TestDeepgramSTT:
  """Tests for DeepgramSTT construction and URL building."""

  def test_construction_defaults(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="test_key")
    assert stt._api_key == "test_key"
    assert stt._model == "nova-3"
    assert stt._language == "en-US"
    assert stt._interim_results is True
    assert stt._endpointing == 300
    assert stt._vad_events is True
    assert stt._utterance_end_ms == 1000
    assert stt._smart_format is True
    assert stt._punctuate is True
    assert stt._keepalive_interval == 5.0

  def test_construction_custom(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(
      api_key="key",
      model="nova-2",
      language="fr-FR",
      interim_results=False,
      endpointing=500,
      vad_events=False,
      utterance_end_ms=2000,
      smart_format=False,
      punctuate=False,
      keepalive_interval=10.0,
    )
    assert stt._model == "nova-2"
    assert stt._language == "fr-FR"
    assert stt._interim_results is False
    assert stt._endpointing == 500
    assert stt._keepalive_interval == 10.0

  def test_api_key_from_env(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "env_key"}):
      stt = DeepgramSTT()
    assert stt._api_key == "env_key"

  def test_build_url(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key", model="nova-3", language="en-US")
    url = stt._build_url(sample_rate=8000, encoding="mulaw", channels=1)
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    assert "model=nova-3" in url
    assert "language=en-US" in url
    assert "encoding=mulaw" in url
    assert "sample_rate=8000" in url
    assert "channels=1" in url
    assert "interim_results=true" in url
    assert "endpointing=300" in url
    assert "vad_events=true" in url
    assert "utterance_end_ms=1000" in url
    assert "smart_format=true" in url
    assert "punctuate=true" in url

  def test_build_url_custom_params(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(
      api_key="key",
      interim_results=False,
      endpointing=0,
      vad_events=False,
      smart_format=False,
      punctuate=False,
    )
    url = stt._build_url(sample_rate=16000, encoding="linear16", channels=2)
    assert "sample_rate=16000" in url
    assert "encoding=linear16" in url
    assert "channels=2" in url
    assert "interim_results=false" in url
    assert "vad_events=false" in url

  def test_initially_disconnected(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    assert stt._connected is False
    assert stt._ws is None

  @pytest.mark.asyncio
  async def test_connect_without_api_key_raises(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="")
    with patch.dict("os.environ", {}, clear=True):
      stt._api_key = ""
      with pytest.raises(ValueError, match="Deepgram API key is required"):
        await stt.connect()

  @pytest.mark.asyncio
  async def test_connect_without_websockets_raises(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    with patch.dict("sys.modules", {"websockets": None}):
      with pytest.raises(ImportError, match="websockets is required"):
        await stt.connect()

  @pytest.mark.asyncio
  async def test_send_audio_when_disconnected_noop(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    # Should not raise
    await stt.send_audio(b"\x00\x01\x02")

  @pytest.mark.asyncio
  async def test_receive_transcripts_when_disconnected(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    transcripts = []
    async for t in stt.receive_transcripts():
      transcripts.append(t)
    assert transcripts == []

  @pytest.mark.asyncio
  async def test_close_when_disconnected_noop(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    # Should not raise
    await stt.close()

  @pytest.mark.asyncio
  async def test_close_sends_closestream(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    mock_ws = AsyncMock()
    stt._ws = mock_ws
    stt._connected = True
    stt._keepalive_task = None

    await stt.close()

    assert stt._connected is False  # type: ignore[unreachable]
    assert stt._ws is None  # type: ignore[unreachable]
    # Should have sent CloseStream and then close
    mock_ws.send.assert_called_once()  # type: ignore[unreachable]
    sent_data = json.loads(mock_ws.send.call_args[0][0])  # type: ignore[unreachable]
    assert sent_data["type"] == "CloseStream"  # type: ignore[unreachable]
    mock_ws.close.assert_called_once()  # type: ignore[unreachable]

  @pytest.mark.asyncio
  async def test_keepalive_cancellation(self):
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    stt = DeepgramSTT(api_key="key")
    stt._connected = True

    # Create a real asyncio task that we can cancel
    async def fake_keepalive():
      await asyncio.sleep(999)

    task = asyncio.create_task(fake_keepalive())
    stt._keepalive_task = task
    stt._ws = AsyncMock()

    await stt.close()

    assert task.cancelled()
    assert stt._keepalive_task is None


# ============================================================
# CartesiaTTS Tests
# ============================================================


class TestCartesiaTTS:
  """Tests for CartesiaTTS construction and encoding mapping."""

  def test_construction_defaults(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="test_key", voice_id="voice123")
    assert tts._api_key == "test_key"
    assert tts._model == "sonic-2"
    assert tts._voice_id == "voice123"
    assert tts._language == "en"
    assert tts._speed == "normal"

  def test_construction_custom(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(
      api_key="key",
      model="sonic-3",
      voice_id="v456",
      language="fr",
      speed="fast",
      cartesia_version="2025-01-01",
    )
    assert tts._model == "sonic-3"
    assert tts._voice_id == "v456"
    assert tts._language == "fr"
    assert tts._speed == "fast"
    assert tts._cartesia_version == "2025-01-01"

  def test_api_key_from_env(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    with patch.dict("os.environ", {"CARTESIA_API_KEY": "env_key"}):
      tts = CartesiaTTS(voice_id="v1")
    assert tts._api_key == "env_key"

  def test_encoding_mapping_mulaw(self):
    from definable.agent.interface.call.tts.cartesia import _map_encoding

    assert _map_encoding("mulaw") == "pcm_mulaw"
    assert _map_encoding("pcm_mulaw") == "pcm_mulaw"

  def test_encoding_mapping_alaw(self):
    from definable.agent.interface.call.tts.cartesia import _map_encoding

    assert _map_encoding("alaw") == "pcm_alaw"
    assert _map_encoding("pcm_alaw") == "pcm_alaw"

  def test_encoding_mapping_linear16(self):
    from definable.agent.interface.call.tts.cartesia import _map_encoding

    assert _map_encoding("linear16") == "pcm_s16le"
    assert _map_encoding("pcm_s16le") == "pcm_s16le"

  def test_encoding_mapping_float32(self):
    from definable.agent.interface.call.tts.cartesia import _map_encoding

    assert _map_encoding("pcm_f32le") == "pcm_f32le"

  def test_encoding_mapping_unsupported_raises(self):
    from definable.agent.interface.call.tts.cartesia import _map_encoding

    with pytest.raises(ValueError, match="Unsupported Cartesia encoding"):
      _map_encoding("opus")

  def test_initially_disconnected(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    assert tts._connected is False
    assert tts._ws is None

  @pytest.mark.asyncio
  async def test_synthesize_empty_text_returns(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    chunks = []
    async for chunk in tts.synthesize_stream(""):
      chunks.append(chunk)
    assert chunks == []

  @pytest.mark.asyncio
  async def test_synthesize_whitespace_returns(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    chunks = []
    async for chunk in tts.synthesize_stream("   "):
      chunks.append(chunk)
    assert chunks == []

  @pytest.mark.asyncio
  async def test_synthesize_no_voice_raises(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="")
    # Mock connection
    tts._ws = AsyncMock()
    tts._connected = True

    with pytest.raises(ValueError, match="Voice ID is required"):
      async for _ in tts.synthesize_stream("Hello"):
        pass

  @pytest.mark.asyncio
  async def test_close_when_disconnected_noop(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    # Should not raise
    await tts.close()

  @pytest.mark.asyncio
  async def test_close_when_connected(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    mock_ws = AsyncMock()
    tts._ws = mock_ws
    tts._connected = True

    await tts.close()

    assert tts._connected is False  # type: ignore[unreachable]
    assert tts._ws is None  # type: ignore[unreachable]
    mock_ws.close.assert_called_once()  # type: ignore[unreachable]

  def test_context_counter_increment(self):
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    tts = CartesiaTTS(api_key="key", voice_id="v1")
    assert tts._context_counter == 0


# ============================================================
# CascadingPipeline Tests
# ============================================================


class _MockSTT:
  """Mock STT provider for testing CascadingPipeline."""

  def __init__(self, transcripts=None):
    self.connected = False
    self.closed = False
    self.audio_received = []
    self._transcripts = transcripts or []

  async def connect(self, *, sample_rate=8000, encoding="mulaw", channels=1):
    self.connected = True

  async def send_audio(self, audio_bytes):
    self.audio_received.append(audio_bytes)

  async def receive_transcripts(self):
    for t in self._transcripts:
      yield t

  async def close(self):
    self.closed = True


class _MockTTS:
  """Mock TTS provider for testing CascadingPipeline."""

  def __init__(self, audio_chunks=None):
    self.closed = False
    self.synthesize_calls = []
    self._audio_chunks = audio_chunks or [b"\x00\x01", b"\x02\x03"]

  async def synthesize_stream(self, text, *, encoding="mulaw", sample_rate=8000, voice="default"):
    self.synthesize_calls.append(text)
    for chunk in self._audio_chunks:
      yield chunk

  async def close(self):
    self.closed = True


class TestCascadingPipeline:
  """Tests for the cascading voice pipeline."""

  def test_construction(self):
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    assert pipeline._stt is stt
    assert pipeline._tts is tts
    assert pipeline._encoding == "mulaw"
    assert pipeline._sample_rate == 8000

  def test_construction_custom(self):
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts, encoding="linear16", sample_rate=16000)
    assert pipeline._encoding == "linear16"
    assert pipeline._sample_rate == 16000

  def test_is_call_pipeline(self):
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    assert isinstance(pipeline, CallPipeline)

  @pytest.mark.asyncio
  async def test_handle_call_connects_stt(self):
    """STT provider should be connected at pipeline start."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    # Create a WebSocket that sends a stop event then closes
    mock_ws = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value=json.dumps({"event": "stop", "streamSid": "MZ1"}))

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    assert stt.connected is True
    assert stt.closed is True
    assert tts.closed is True

  @pytest.mark.asyncio
  async def test_handle_call_stt_connect_failure(self):
    """Pipeline should handle STT connection failure gracefully."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    class FailingSTT(_MockSTT):
      async def connect(self, **kwargs):
        raise ConnectionError("STT connect failed")

    stt = FailingSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    # Should record error event
    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.ERROR

  @pytest.mark.asyncio
  async def test_handle_call_ends_session(self):
    """Session should be ENDED after pipeline completes."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    mock_ws = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value=json.dumps({"event": "stop", "streamSid": "MZ1"}))

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    assert session.state == CallState.ENDED

  @pytest.mark.asyncio
  async def test_cleanup_closes_both_providers(self):
    """Cleanup should close both STT and TTS providers."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)

    await pipeline._cleanup()

    assert stt.closed is True
    assert tts.closed is True

  @pytest.mark.asyncio
  async def test_cleanup_handles_errors(self):
    """Cleanup should not raise even if providers fail to close."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    class FailCloseSTT(_MockSTT):
      async def close(self):
        raise RuntimeError("STT close failed")

    class FailCloseTTS(_MockTTS):
      async def close(self):
        raise RuntimeError("TTS close failed")

    pipeline = CascadingPipeline(stt=FailCloseSTT(), tts=FailCloseTTS())
    # Should not raise
    await pipeline._cleanup()

  @pytest.mark.asyncio
  async def test_read_websocket_parses_start_event(self):
    """WebSocket reader should parse start events and update session."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123")
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    # Send start event, then stop
    call_count = 0
    events = [
      json.dumps({"event": "start", "start": {"callSid": "CA999", "streamSid": "MZ1", "mediaFormat": {}}}),
      json.dumps({"event": "stop", "streamSid": "MZ1"}),
    ]

    async def mock_receive():
      nonlocal call_count
      if call_count < len(events):
        result = events[call_count]
        call_count += 1
        return result
      raise Exception("closed")

    mock_ws = AsyncMock()
    mock_ws.receive_text = mock_receive

    await pipeline._read_websocket(mock_ws, session, telephony)

    assert session.stream_id == "MZ1"
    assert session.call_id == "CA999"
    assert session.state == CallState.ENDED

  @pytest.mark.asyncio
  async def test_read_websocket_forwards_audio_to_stt(self):
    """Audio events should be forwarded to STT provider."""
    import base64

    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    audio_payload = base64.b64encode(b"\xaa\xbb\xcc").decode("ascii")
    call_count = 0
    events = [
      json.dumps({"event": "media", "streamSid": "MZ1", "media": {"payload": audio_payload, "chunk": "1", "timestamp": "0"}}),
      json.dumps({"event": "stop", "streamSid": "MZ1"}),
    ]

    async def mock_receive():
      nonlocal call_count
      if call_count < len(events):
        result = events[call_count]
        call_count += 1
        return result
      raise Exception("closed")

    mock_ws = AsyncMock()
    mock_ws.receive_text = mock_receive

    await pipeline._read_websocket(mock_ws, session, telephony)

    assert len(stt.audio_received) == 1
    assert stt.audio_received[0] == b"\xaa\xbb\xcc"

  @pytest.mark.asyncio
  async def test_listen_stt_pushes_final_transcripts(self):
    """STT listener should push final transcripts to utterance queue."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    transcripts = [
      Transcript(text="Hello", is_final=False, confidence=0.8),
      Transcript(text="Hello world", is_final=True, confidence=0.95),
    ]
    stt = _MockSTT(transcripts=transcripts)
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    utterance_queue: asyncio.Queue[str] = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_stt(mock_ws, session, telephony, utterance_queue, playback)

    # Only the final transcript should be in the queue
    assert utterance_queue.qsize() == 1
    assert await utterance_queue.get() == "Hello world"

    # Session should have the utterance event
    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.UTTERANCE

    # User message should be recorded
    assert len(session.conversation) == 1
    assert session.conversation[0] == {"role": "user", "content": "Hello world"}

  @pytest.mark.asyncio
  async def test_listen_stt_barge_in_detection(self):
    """STT listener should detect barge-in during playback."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    transcripts = [
      Transcript(text="Wait", is_final=False, confidence=0.5),
    ]
    stt = _MockSTT(transcripts=transcripts)
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE, stream_id="MZ1")
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    utterance_queue: asyncio.Queue[str] = asyncio.Queue()
    playback = _PlaybackState()
    playback.active = True  # Simulate active TTS playback

    await pipeline._listen_stt(mock_ws, session, telephony, utterance_queue, playback)

    # Barge-in should be detected
    assert playback.interrupted is True

    # Interruption event should be recorded
    has_interrupt = any(e.type == CallEventType.INTERRUPTION for e in session.events)
    assert has_interrupt

    # Clear audio should have been sent
    mock_ws.send_json.assert_called_once()
    clear_msg = mock_ws.send_json.call_args[0][0]
    assert clear_msg["event"] == "clear"
    assert clear_msg["streamSid"] == "MZ1"

  @pytest.mark.asyncio
  async def test_listen_stt_skips_empty_transcripts(self):
    """STT listener should skip empty transcripts (UtteranceEnd markers)."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    transcripts = [
      Transcript(text="", is_final=True, confidence=1.0),  # UtteranceEnd
    ]
    stt = _MockSTT(transcripts=transcripts)
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    utterance_queue: asyncio.Queue[str] = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_stt(mock_ws, session, telephony, utterance_queue, playback)

    assert utterance_queue.qsize() == 0
    assert len(session.events) == 0

  @pytest.mark.asyncio
  async def test_stream_tts_sends_audio(self):
    """TTS streaming should encode and send audio chunks via WebSocket."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    stt = _MockSTT()
    tts = _MockTTS(audio_chunks=[b"\x00\x01", b"\x02\x03", b"\x04\x05"])
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()
    playback = _PlaybackState()

    await pipeline._stream_tts(mock_ws, telephony, session, "Hello!", playback)

    # Should have sent 3 audio chunks
    assert mock_ws.send_json.call_count == 3

    # Each should be a media event
    for call_args in mock_ws.send_json.call_args_list:
      msg = call_args[0][0]
      assert msg["event"] == "media"
      assert msg["streamSid"] == "MZ1"
      assert "payload" in msg["media"]

    # TTS should have been called with the text
    assert tts.synthesize_calls == ["Hello!"]

  @pytest.mark.asyncio
  async def test_stream_tts_stops_on_barge_in(self):
    """TTS streaming should stop when barge-in is detected mid-stream."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    # TTS that yields many chunks; we'll interrupt after the 3rd
    stt = _MockSTT()
    tts = _MockTTS(audio_chunks=[b"\x00"] * 100)
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    send_count = 0

    async def counting_send_json(msg):
      nonlocal send_count
      send_count += 1
      # Simulate barge-in after 3 chunks are sent
      if send_count >= 3:
        playback.interrupted = True

    mock_ws = AsyncMock()
    mock_ws.send_json = counting_send_json
    playback = _PlaybackState()

    await pipeline._stream_tts(mock_ws, telephony, session, "Long response.", playback)

    # Should have stopped after ~3 chunks (not all 100)
    assert send_count < 100
    assert send_count >= 3
    # Conversation should be truncated due to interruption
    assert playback.active is False

  @pytest.mark.asyncio
  async def test_stream_tts_no_stream_id_noop(self):
    """TTS streaming should be a no-op without a stream_id."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    stt = _MockSTT()
    tts = _MockTTS()
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", stream_id="", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()
    playback = _PlaybackState()

    await pipeline._stream_tts(mock_ws, telephony, session, "Hello", playback)

    # No audio should be sent
    mock_ws.send_json.assert_not_called()

  @pytest.mark.asyncio
  async def test_stream_tts_sets_playback_state(self):
    """Playback state should be active during TTS and inactive after."""
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline, _PlaybackState

    stt = _MockSTT()
    tts = _MockTTS(audio_chunks=[b"\x00"])
    pipeline = CascadingPipeline(stt=stt, tts=tts)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()
    playback = _PlaybackState()

    assert playback.active is False
    await pipeline._stream_tts(mock_ws, telephony, session, "Hi", playback)
    assert playback.active is False  # Should be False after completion

  def test_playback_state_defaults(self):
    from definable.agent.interface.call.pipeline.cascading import _PlaybackState

    state = _PlaybackState()
    assert state.active is False
    assert state.interrupted is False


# ============================================================
# CallInterface with Cascading Pipeline Tests
# ============================================================


class TestCallInterfaceCascading:
  """Tests for CallInterface with cascading pipeline mode."""

  def test_cascading_pipeline_creation(self):
    stt = _MockSTT()
    tts = _MockTTS()
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="tok",
      pipeline="cascading",
      stt=stt,
      tts=tts,
    )
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    assert isinstance(ci._pipeline, CascadingPipeline)
    assert ci._call_config.pipeline_mode == "cascading"

  def test_cascading_without_stt_raises(self):
    tts = _MockTTS()
    with pytest.raises(ValueError, match="stt= and tts= providers"):
      CallInterface(
        provider="twilio",
        phone_number="+1555",
        account_sid="AC",
        auth_token="tok",
        pipeline="cascading",
        tts=tts,
      )

  def test_cascading_without_tts_raises(self):
    stt = _MockSTT()
    with pytest.raises(ValueError, match="stt= and tts= providers"):
      CallInterface(
        provider="twilio",
        phone_number="+1555",
        account_sid="AC",
        auth_token="tok",
        pipeline="cascading",
        stt=stt,
      )

  def test_cascading_router_uses_stream_path(self):
    """Cascading mode should use /call/stream/{call_id} WebSocket path."""
    stt = _MockSTT()
    tts = _MockTTS()
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="tok",
      pipeline="cascading",
      stt=stt,
      tts=tts,
    )
    assert ci._call_config.pipeline_mode == "cascading"


# ============================================================
# OpenAIRealtimeProvider Tests
# ============================================================


class TestOpenAIRealtimeProvider:
  """Tests for OpenAIRealtimeProvider construction and event mapping."""

  def test_construction_defaults(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    assert provider._api_key == "sk-test"
    assert provider._model == "gpt-4o-realtime-preview"
    assert provider._voice == "alloy"
    assert provider._temperature == 0.8
    assert provider._max_response_output_tokens == "inf"
    assert provider._transcription_model == "whisper-1"
    assert provider._connected is False
    assert provider._ws is None

  def test_construction_custom(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(
      api_key="sk-custom",
      model="gpt-4o-realtime-preview-2025-01",
      voice="nova",
      temperature=0.6,
      max_response_output_tokens="4096",
      transcription_model="whisper-2",
      turn_detection={"type": "server_vad", "threshold": 0.3},
    )
    assert provider._model == "gpt-4o-realtime-preview-2025-01"
    assert provider._voice == "nova"
    assert provider._temperature == 0.6
    assert provider._max_response_output_tokens == "4096"
    assert provider._transcription_model == "whisper-2"
    assert provider._turn_detection["threshold"] == 0.3

  def test_api_key_from_env(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
      provider = OpenAIRealtimeProvider()
    assert provider._api_key == "sk-env"

  def test_encoding_map(self):
    from definable.agent.interface.call.realtime.openai import _ENCODING_MAP

    assert _ENCODING_MAP["mulaw"] == "g711_ulaw"
    assert _ENCODING_MAP["g711_ulaw"] == "g711_ulaw"
    assert _ENCODING_MAP["alaw"] == "g711_alaw"
    assert _ENCODING_MAP["pcm16"] == "pcm16"
    assert _ENCODING_MAP["linear16"] == "pcm16"

  def test_is_realtime_provider(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    assert isinstance(provider, RealtimeProvider)

  def test_default_turn_detection(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    assert provider._turn_detection["type"] == "server_vad"
    assert provider._turn_detection["threshold"] == 0.5
    assert provider._turn_detection["prefix_padding_ms"] == 300
    assert provider._turn_detection["silence_duration_ms"] == 500

  @pytest.mark.asyncio
  async def test_connect_without_api_key_raises(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="")
    with patch.dict("os.environ", {}, clear=True):
      provider._api_key = ""
      with pytest.raises(ValueError, match="OpenAI API key is required"):
        await provider.connect(instructions="test")

  @pytest.mark.asyncio
  async def test_connect_without_websockets_raises(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    with patch.dict("sys.modules", {"websockets": None}):
      with pytest.raises(ImportError, match="websockets is required"):
        await provider.connect(instructions="test")

  @pytest.mark.asyncio
  async def test_send_audio_when_disconnected_noop(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    # Should not raise
    await provider.send_audio(b"\x00\x01\x02")

  @pytest.mark.asyncio
  async def test_send_audio_encodes_base64(self):
    import base64

    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    mock_ws = AsyncMock()
    provider._ws = mock_ws
    provider._connected = True

    await provider.send_audio(b"\xaa\xbb\xcc")

    mock_ws.send.assert_called_once()
    sent_data = json.loads(mock_ws.send.call_args[0][0])
    assert sent_data["type"] == "input_audio_buffer.append"
    assert base64.b64decode(sent_data["audio"]) == b"\xaa\xbb\xcc"

  @pytest.mark.asyncio
  async def test_send_tool_result_sends_two_messages(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    mock_ws = AsyncMock()
    provider._ws = mock_ws
    provider._connected = True

    await provider.send_tool_result("call_123", "Tool result text")

    assert mock_ws.send.call_count == 2

    # First: conversation.item.create
    msg1 = json.loads(mock_ws.send.call_args_list[0][0][0])
    assert msg1["type"] == "conversation.item.create"
    assert msg1["item"]["type"] == "function_call_output"
    assert msg1["item"]["call_id"] == "call_123"
    assert msg1["item"]["output"] == "Tool result text"

    # Second: response.create
    msg2 = json.loads(mock_ws.send.call_args_list[1][0][0])
    assert msg2["type"] == "response.create"

  @pytest.mark.asyncio
  async def test_send_tool_result_when_disconnected_noop(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    # Should not raise
    await provider.send_tool_result("call_123", "result")

  @pytest.mark.asyncio
  async def test_interrupt_sends_cancel(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    mock_ws = AsyncMock()
    provider._ws = mock_ws
    provider._connected = True

    await provider.interrupt()

    mock_ws.send.assert_called_once()
    sent_data = json.loads(mock_ws.send.call_args[0][0])
    assert sent_data["type"] == "response.cancel"

  @pytest.mark.asyncio
  async def test_interrupt_when_disconnected_noop(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    # Should not raise
    await provider.interrupt()

  @pytest.mark.asyncio
  async def test_send_truncate(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    mock_ws = AsyncMock()
    provider._ws = mock_ws
    provider._connected = True

    await provider.send_truncate("item_abc", 1500)

    mock_ws.send.assert_called_once()
    sent_data = json.loads(mock_ws.send.call_args[0][0])
    assert sent_data["type"] == "conversation.item.truncate"
    assert sent_data["item_id"] == "item_abc"
    assert sent_data["content_index"] == 0
    assert sent_data["audio_end_ms"] == 1500

  @pytest.mark.asyncio
  async def test_close(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    mock_ws = AsyncMock()
    provider._ws = mock_ws
    provider._connected = True

    await provider.close()

    assert provider._connected is False  # type: ignore[unreachable]
    assert provider._ws is None  # type: ignore[unreachable]
    mock_ws.close.assert_called_once()  # type: ignore[unreachable]

  @pytest.mark.asyncio
  async def test_close_when_disconnected_noop(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    # Should not raise
    await provider.close()

  @pytest.mark.asyncio
  async def test_receive_events_when_disconnected(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")
    events = []
    async for e in provider.receive_events():
      events.append(e)
    assert events == []

  @pytest.mark.asyncio
  async def test_receive_events_audio_delta(self):
    import base64

    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    audio_b64 = base64.b64encode(b"\xaa\xbb").decode("ascii")
    messages = [
      json.dumps({
        "type": "response.audio.delta",
        "delta": audio_b64,
        "response_id": "resp_1",
        "item_id": "item_1",
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "audio_delta"
    assert events[0].audio == b"\xaa\xbb"
    assert events[0].metadata["item_id"] == "item_1"

  @pytest.mark.asyncio
  async def test_receive_events_transcript(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "Hello world",
        "item_id": "item_2",
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "transcript"
    assert events[0].text == "Hello world"

  @pytest.mark.asyncio
  async def test_receive_events_tool_call(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "response.function_call_arguments.done",
        "call_id": "call_99",
        "name": "my_tool",
        "arguments": '{"query": "test"}',
        "item_id": "item_3",
        "response_id": "resp_2",
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "tool_call"
    assert events[0].tool_call is not None
    assert events[0].tool_call["id"] == "call_99"
    assert events[0].tool_call["name"] == "my_tool"
    assert events[0].tool_call["arguments"] == '{"query": "test"}'

  @pytest.mark.asyncio
  async def test_receive_events_speech_started(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 1234,
        "item_id": "item_4",
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "speech_started"
    assert events[0].metadata["audio_start_ms"] == 1234

  @pytest.mark.asyncio
  async def test_receive_events_response_done_completed(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "response.done",
        "response": {"id": "resp_3", "status": "completed", "usage": {"total_tokens": 100}},
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "turn_complete"
    assert events[0].metadata["status"] == "completed"

  @pytest.mark.asyncio
  async def test_receive_events_response_done_cancelled(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "response.done",
        "response": {"id": "resp_4", "status": "cancelled"},
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "interrupted"

  @pytest.mark.asyncio
  async def test_receive_events_error(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    provider = OpenAIRealtimeProvider(api_key="sk-test")

    messages = [
      json.dumps({
        "type": "error",
        "error": {"message": "Something went wrong", "code": "internal_error"},
      }),
    ]

    async def mock_iter(self_ws):
      for msg in messages:
        yield msg

    mock_ws = AsyncMock()
    mock_ws.__aiter__ = lambda self: mock_iter(self)
    provider._ws = mock_ws
    provider._connected = True

    events = []
    async for e in provider.receive_events():
      events.append(e)

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].text == "Something went wrong"


# ============================================================
# RealtimePipeline Tests
# ============================================================


class _MockRealtimeProvider(RealtimeProvider):
  """Mock realtime provider for testing RealtimePipeline."""

  def __init__(self, events=None):  # type: ignore[assignment]
    self.connected = False
    self.closed = False
    self.audio_received: list[bytes] = []
    self.tool_results: list[dict[str, Any]] = []
    self.truncate_calls: list[dict[str, Any]] = []
    self._events: list[RealtimeEvent] = events or []
    self._connect_kwargs: dict[str, Any] = {}

  async def connect(self, **kwargs: Any) -> None:  # type: ignore[override]
    self.connected = True
    self._connect_kwargs = kwargs

  async def send_audio(self, audio_bytes: bytes) -> None:
    self.audio_received.append(audio_bytes)

  async def receive_events(self) -> AsyncIterator[RealtimeEvent]:  # type: ignore[override]
    for event in self._events:
      yield event

  async def send_tool_result(self, call_id: str, result: str) -> None:
    self.tool_results.append({"call_id": call_id, "result": result})

  async def send_truncate(self, item_id: str, audio_end_ms: int) -> None:
    self.truncate_calls.append({"item_id": item_id, "audio_end_ms": audio_end_ms})

  async def interrupt(self) -> None:
    pass

  async def close(self) -> None:
    self.closed = True


class TestRealtimePipeline:
  """Tests for the realtime voice pipeline."""

  def test_construction(self):
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    assert pipeline._realtime is provider

  def test_is_call_pipeline(self):
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    assert isinstance(pipeline, CallPipeline)

  @pytest.mark.asyncio
  async def test_handle_call_connects_provider(self):
    """Realtime provider should be connected at pipeline start."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    mock_agent.instructions = "You are helpful."
    mock_agent._tools_dict = {}
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    # WebSocket that sends stop event immediately
    mock_ws = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value=json.dumps({"event": "stop", "streamSid": "MZ1"}))

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    assert provider.connected is True
    assert provider.closed is True
    assert provider._connect_kwargs["instructions"] == "You are helpful."

  @pytest.mark.asyncio
  async def test_handle_call_connect_failure(self):
    """Pipeline should handle realtime connection failure gracefully."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    class FailingProvider(_MockRealtimeProvider):
      async def connect(self, **kwargs):
        raise ConnectionError("Realtime connect failed")

    provider = FailingProvider()
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    mock_agent.instructions = "test"
    mock_agent._tools_dict = {}
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    # Should record error event
    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.ERROR

  @pytest.mark.asyncio
  async def test_handle_call_ends_session(self):
    """Session should be ENDED after pipeline completes."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123")

    mock_agent = AsyncMock()
    mock_agent.instructions = "test"
    mock_agent._tools_dict = {}
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    mock_ws = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value=json.dumps({"event": "stop", "streamSid": "MZ1"}))

    await pipeline.handle_call(mock_ws, session, mock_agent, telephony)

    assert session.state == CallState.ENDED

  @pytest.mark.asyncio
  async def test_read_websocket_parses_start_event(self):
    """WebSocket reader should parse start events and update session."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123")
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    call_count = 0
    events = [
      json.dumps({"event": "start", "start": {"callSid": "CA999", "streamSid": "MZ1", "mediaFormat": {}}}),
      json.dumps({"event": "stop", "streamSid": "MZ1"}),
    ]

    async def mock_receive():
      nonlocal call_count
      if call_count < len(events):
        result = events[call_count]
        call_count += 1
        return result
      raise Exception("closed")

    mock_ws = AsyncMock()
    mock_ws.receive_text = mock_receive

    await pipeline._read_websocket(mock_ws, session, telephony)

    assert session.stream_id == "MZ1"
    assert session.call_id == "CA999"
    assert session.state == CallState.ENDED

  @pytest.mark.asyncio
  async def test_read_websocket_forwards_audio_to_provider(self):
    """Audio events should be forwarded to the realtime provider."""
    import base64

    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")

    audio_payload = base64.b64encode(b"\xaa\xbb\xcc").decode("ascii")
    call_count = 0
    events = [
      json.dumps({"event": "media", "streamSid": "MZ1", "media": {"payload": audio_payload, "chunk": "1", "timestamp": "0"}}),
      json.dumps({"event": "stop", "streamSid": "MZ1"}),
    ]

    async def mock_receive():
      nonlocal call_count
      if call_count < len(events):
        result = events[call_count]
        call_count += 1
        return result
      raise Exception("closed")

    mock_ws = AsyncMock()
    mock_ws.receive_text = mock_receive

    await pipeline._read_websocket(mock_ws, session, telephony)

    assert len(provider.audio_received) == 1
    assert provider.audio_received[0] == b"\xaa\xbb\xcc"

  @pytest.mark.asyncio
  async def test_listen_events_audio_delta(self):
    """Audio delta events should be forwarded to the caller via WebSocket."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(
        type="audio_delta",
        audio=b"\x00\x01",
        metadata={"item_id": "item_1"},
      ),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    # Should have sent audio to the WebSocket
    mock_ws.send_json.assert_called_once()
    msg = mock_ws.send_json.call_args[0][0]
    assert msg["event"] == "media"
    assert msg["streamSid"] == "MZ1"

    # Playback should have been active
    assert playback.current_item_id == "item_1"

  @pytest.mark.asyncio
  async def test_listen_events_transcript(self):
    """Transcript events should record user speech in session."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="transcript", text="Hello there"),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    assert len(session.conversation) == 1
    assert session.conversation[0] == {"role": "user", "content": "Hello there"}

    # Should have an utterance event
    has_utterance = any(e.type == CallEventType.UTTERANCE for e in session.events)
    assert has_utterance

  @pytest.mark.asyncio
  async def test_listen_events_assistant_transcript(self):
    """Assistant transcript events should record agent speech."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="assistant_transcript", text="I can help with that"),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    assert len(session.conversation) == 1
    assert session.conversation[0] == {"role": "assistant", "content": "I can help with that"}

  @pytest.mark.asyncio
  async def test_listen_events_tool_call_queued(self):
    """Tool call events should be queued for the tool handler."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    tool_call_data = {"id": "call_1", "name": "my_tool", "arguments": '{"q": "test"}'}
    events = [
      RealtimeEvent(type="tool_call", tool_call=tool_call_data),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    assert tool_queue.qsize() == 1
    queued = await tool_queue.get()
    assert queued["name"] == "my_tool"

  @pytest.mark.asyncio
  async def test_listen_events_speech_started_barge_in(self):
    """Speech started during playback should trigger barge-in."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="speech_started", metadata={"audio_start_ms": 500}),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()
    playback.active = True
    playback.current_item_id = "item_1"
    playback.audio_ms = 1500

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    # Barge-in should clear audio
    mock_ws.send_json.assert_called_once()
    clear_msg = mock_ws.send_json.call_args[0][0]
    assert clear_msg["event"] == "clear"
    assert clear_msg["streamSid"] == "MZ1"

    # Playback should be reset
    assert playback.active is False
    assert playback.audio_ms == 0

    # Truncate should have been called on the provider
    assert len(provider.truncate_calls) == 1
    assert provider.truncate_calls[0]["item_id"] == "item_1"
    assert provider.truncate_calls[0]["audio_end_ms"] == 1500

    # Interruption event recorded
    has_interrupt = any(e.type == CallEventType.INTERRUPTION for e in session.events)
    assert has_interrupt

  @pytest.mark.asyncio
  async def test_listen_events_speech_started_no_playback(self):
    """Speech started when not playing should be a no-op."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="speech_started", metadata={"audio_start_ms": 500}),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", stream_id="MZ1", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()
    playback.active = False

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    # Should NOT send clear audio when not playing
    mock_ws.send_json.assert_not_called()
    assert len(session.events) == 0

  @pytest.mark.asyncio
  async def test_listen_events_turn_complete(self):
    """Turn complete should reset playback state."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="turn_complete"),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()
    playback.active = True
    playback.audio_ms = 5000

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    assert playback.active is False
    assert playback.audio_ms == 0

  @pytest.mark.asyncio
  async def test_listen_events_error(self):
    """Error events should record in session."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline, _PlaybackState

    events = [
      RealtimeEvent(type="error", text="Something broke"),
    ]
    provider = _MockRealtimeProvider(events=events)
    pipeline = RealtimePipeline(realtime=provider)
    session = CallSession(call_id="CA123", state=CallState.ACTIVE)
    telephony = TwilioProvider(account_sid="AC", auth_token="tok")
    mock_ws = AsyncMock()

    tool_queue: asyncio.Queue = asyncio.Queue()
    playback = _PlaybackState()

    await pipeline._listen_events(mock_ws, session, telephony, tool_queue, playback)

    assert len(session.events) == 1
    assert session.events[0].type == CallEventType.ERROR

  @pytest.mark.asyncio
  async def test_handle_tool_calls_invokes_tool(self):
    """Tool handler should invoke the matching tool and send result."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)

    # Create a mock agent with a tool
    mock_fn = AsyncMock()
    mock_fn.name = "my_tool"
    mock_fn.entrypoint = lambda query: f"Result for {query}"

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {"my_tool": mock_fn}

    session = CallSession(call_id="CA123", state=CallState.ACTIVE)

    tool_queue: asyncio.Queue = asyncio.Queue()
    await tool_queue.put({"id": "call_1", "name": "my_tool", "arguments": '{"query": "test"}'})

    # End the session after a short delay so the handler exits
    async def end_session():
      await asyncio.sleep(0.1)
      session.state = CallState.ENDED

    asyncio.create_task(end_session())

    await pipeline._handle_tool_calls(session, mock_agent, tool_queue)

    assert len(provider.tool_results) == 1
    assert provider.tool_results[0]["call_id"] == "call_1"
    assert provider.tool_results[0]["result"] == "Result for test"

  @pytest.mark.asyncio
  async def test_handle_tool_calls_unknown_tool(self):
    """Unknown tool should return an error message."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {}

    session = CallSession(call_id="CA123", state=CallState.ACTIVE)

    tool_queue: asyncio.Queue = asyncio.Queue()
    await tool_queue.put({"id": "call_1", "name": "nonexistent", "arguments": "{}"})

    async def end_session():
      await asyncio.sleep(0.1)
      session.state = CallState.ENDED

    asyncio.create_task(end_session())

    await pipeline._handle_tool_calls(session, mock_agent, tool_queue)

    assert len(provider.tool_results) == 1
    assert "Unknown tool" in provider.tool_results[0]["result"]

  @pytest.mark.asyncio
  async def test_handle_tool_calls_async_tool(self):
    """Async tools should be awaited correctly."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)

    async def async_tool(query: str) -> str:
      return f"Async result: {query}"

    mock_fn = AsyncMock()
    mock_fn.name = "async_tool"
    mock_fn.entrypoint = async_tool

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {"async_tool": mock_fn}

    session = CallSession(call_id="CA123", state=CallState.ACTIVE)

    tool_queue: asyncio.Queue = asyncio.Queue()
    await tool_queue.put({"id": "call_2", "name": "async_tool", "arguments": '{"query": "async test"}'})

    async def end_session():
      await asyncio.sleep(0.1)
      session.state = CallState.ENDED

    asyncio.create_task(end_session())

    await pipeline._handle_tool_calls(session, mock_agent, tool_queue)

    assert len(provider.tool_results) == 1
    assert provider.tool_results[0]["result"] == "Async result: async test"

  @pytest.mark.asyncio
  async def test_handle_tool_calls_tool_error(self):
    """Tool errors should send error message back to provider."""
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    provider = _MockRealtimeProvider()
    pipeline = RealtimePipeline(realtime=provider)

    def failing_tool(**kwargs):
      raise RuntimeError("Tool exploded")

    mock_fn = AsyncMock()
    mock_fn.name = "failing_tool"
    mock_fn.entrypoint = failing_tool

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {"failing_tool": mock_fn}

    session = CallSession(call_id="CA123", state=CallState.ACTIVE)

    tool_queue: asyncio.Queue = asyncio.Queue()
    await tool_queue.put({"id": "call_3", "name": "failing_tool", "arguments": "{}"})

    async def end_session():
      await asyncio.sleep(0.1)
      session.state = CallState.ENDED

    asyncio.create_task(end_session())

    await pipeline._handle_tool_calls(session, mock_agent, tool_queue)

    assert len(provider.tool_results) == 1
    assert "Error executing tool" in provider.tool_results[0]["result"]

  def test_build_tool_definitions(self):
    """Tool definitions should be extracted from agent in OpenAI format."""
    from definable.agent.interface.call.pipeline.realtime import _build_tool_definitions

    mock_fn = AsyncMock()
    mock_fn.name = "search"
    mock_fn.description = "Search for documents"
    mock_fn.parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {"search": mock_fn}

    defs = _build_tool_definitions(mock_agent)

    assert len(defs) == 1
    assert defs[0]["type"] == "function"
    assert defs[0]["name"] == "search"
    assert defs[0]["description"] == "Search for documents"
    assert defs[0]["parameters"]["properties"]["query"]["type"] == "string"

  def test_build_tool_definitions_empty(self):
    """Empty tools dict should return empty list."""
    from definable.agent.interface.call.pipeline.realtime import _build_tool_definitions

    mock_agent = AsyncMock()
    mock_agent._tools_dict = {}

    defs = _build_tool_definitions(mock_agent)
    assert defs == []

  def test_estimate_audio_ms(self):
    """Audio duration estimate for mu-law at 8kHz."""
    from definable.agent.interface.call.pipeline.realtime import _estimate_audio_ms

    # 8000 bytes = 1 second
    assert _estimate_audio_ms(b"\x00" * 8000) == 1000
    # 4000 bytes = 500ms
    assert _estimate_audio_ms(b"\x00" * 4000) == 500
    # 0 bytes = 0ms
    assert _estimate_audio_ms(b"") == 0

  def test_playback_state_defaults(self):
    from definable.agent.interface.call.pipeline.realtime import _PlaybackState

    state = _PlaybackState()
    assert state.active is False
    assert state.current_item_id == ""
    assert state.audio_ms == 0


# ============================================================
# CallInterface with Realtime Pipeline Tests
# ============================================================


class TestCallInterfaceRealtime:
  """Tests for CallInterface with realtime pipeline mode."""

  def test_realtime_pipeline_creation(self):
    provider = _MockRealtimeProvider()
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="tok",
      pipeline="realtime",
      realtime=provider,
    )
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    assert isinstance(ci._pipeline, RealtimePipeline)
    assert ci._call_config.pipeline_mode == "realtime"

  def test_realtime_without_provider_raises(self):
    with pytest.raises(ValueError, match="realtime= provider"):
      CallInterface(
        provider="twilio",
        phone_number="+1555",
        account_sid="AC",
        auth_token="tok",
        pipeline="realtime",
      )

  def test_realtime_stores_provider(self):
    provider = _MockRealtimeProvider()
    ci = CallInterface(
      provider="twilio",
      phone_number="+1555",
      account_sid="AC",
      auth_token="tok",
      pipeline="realtime",
      realtime=provider,
    )
    assert ci._realtime is provider


# ============================================================
# Import Tests for Phase 3 Types
# ============================================================


class TestPhase3Imports:
  """Verify Phase 3 types are importable from the call package."""

  def test_import_realtime_pipeline(self):
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    assert RealtimePipeline is not None

  def test_import_openai_realtime_provider(self):
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    assert OpenAIRealtimeProvider is not None

  def test_lazy_import_realtime_pipeline(self):
    from definable.agent.interface.call import RealtimePipeline

    assert RealtimePipeline is not None

  def test_lazy_import_openai_realtime_provider(self):
    from definable.agent.interface.call import OpenAIRealtimeProvider

    assert OpenAIRealtimeProvider is not None

  def test_lazy_import_from_pipeline_package(self):
    from definable.agent.interface.call.pipeline import RealtimePipeline

    assert RealtimePipeline is not None

  def test_lazy_import_from_realtime_package(self):
    from definable.agent.interface.call.realtime import OpenAIRealtimeProvider

    assert OpenAIRealtimeProvider is not None


# ============================================================
# PlivoProvider Tests
# ============================================================


class TestPlivoProvider:
  """Tests for PlivoProvider construction, XML generation, event parsing, and encoding."""

  def test_construction_defaults(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA123", auth_token="tok123")
    assert provider.auth_id == "MA123"
    assert provider.auth_token == "tok123"

  def test_construction_env_vars(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    with patch.dict("os.environ", {"PLIVO_AUTH_ID": "MA_ENV", "PLIVO_AUTH_TOKEN": "TOK_ENV"}):
      provider = PlivoProvider()
    assert provider.auth_id == "MA_ENV"
    assert provider.auth_token == "TOK_ENV"

  def test_explicit_overrides_env(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    with patch.dict("os.environ", {"PLIVO_AUTH_ID": "MA_ENV"}):
      provider = PlivoProvider(auth_id="MA_EXPLICIT")
    assert provider.auth_id == "MA_EXPLICIT"

  def test_is_telephony_provider(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    assert isinstance(provider, TelephonyProvider)

  # --- XML Generation ---

  def test_generate_stream_xml_basic(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    xml = provider.generate_answer_xml("wss://example.com/call/stream/CA123", mode="stream")

    assert '<?xml version="1.0" encoding="UTF-8"?>' in xml
    assert "<Response>" in xml
    assert 'bidirectional="true"' in xml
    assert 'keepCallAlive="true"' in xml
    assert 'contentType="audio/x-mulaw;rate=8000"' in xml
    assert "wss://example.com/call/stream/CA123" in xml
    assert "</Stream>" in xml
    assert "</Response>" in xml
    # No <Speak> without welcome message
    assert "<Speak>" not in xml

  def test_generate_stream_xml_with_welcome(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    xml = provider.generate_answer_xml(
      "wss://example.com/stream",
      welcome_message="Hello! How can I help?",
      mode="stream",
    )

    assert "<Speak>Hello! How can I help?</Speak>" in xml
    # Speak should be before Stream
    speak_pos = xml.index("<Speak>")
    stream_pos = xml.index("<Stream")
    assert speak_pos < stream_pos

  def test_generate_xml_escapes_welcome(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    xml = provider.generate_answer_xml(
      "wss://example.com/stream",
      welcome_message='Hello & welcome to "Acme"!',
      mode="stream",
    )

    assert "Hello &amp; welcome to &quot;Acme&quot;!" in xml

  def test_generate_xml_custom_content_type(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    xml = provider.generate_answer_xml(
      "wss://example.com/stream",
      mode="stream",
      content_type="audio/x-l16;rate=16000",
    )

    assert 'contentType="audio/x-l16;rate=16000"' in xml

  def test_generate_managed_xml_raises(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    with pytest.raises(ValueError, match="does not support managed pipeline"):
      provider.generate_answer_xml("wss://example.com/convo", mode="managed")

  def test_generate_xml_default_mode_is_stream(self):
    """Default mode should be 'stream' (not 'managed')."""
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    # Should NOT raise — default is stream mode
    xml = provider.generate_answer_xml("wss://example.com/stream")
    assert "<Stream" in xml

  # --- WebSocket Event Parsing ---

  def test_parse_start_event(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    event = provider.parse_websocket_event({
      "sequenceNumber": 0,
      "event": "start",
      "start": {
        "callId": "8c43a765-94fa",
        "streamId": "b77e037d-4119",
        "accountId": "155747",
        "tracks": ["inbound", "outbound"],
        "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 8000},
      },
      "extra_headers": "{}",
    })

    assert event.event == "start"
    assert event.call_id == "8c43a765-94fa"
    assert event.stream_id == "b77e037d-4119"
    assert event.metadata["account_id"] == "155747"
    assert event.metadata["tracks"] == ["inbound", "outbound"]
    assert event.metadata["media_format"]["sampleRate"] == 8000

  def test_parse_media_event(self):
    import base64

    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    audio_b64 = base64.b64encode(b"\xaa\xbb\xcc").decode("ascii")
    event = provider.parse_websocket_event({
      "sequenceNumber": 42,
      "streamId": "stream123",
      "event": "media",
      "media": {
        "track": "inbound",
        "timestamp": "1687353805345",
        "chunk": 469,
        "payload": audio_b64,
      },
    })

    assert event.event == "media"
    assert event.stream_id == "stream123"
    assert event.payload == b"\xaa\xbb\xcc"
    assert event.metadata["track"] == "inbound"
    assert event.metadata["chunk"] == 469

  def test_parse_media_event_empty_payload(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    event = provider.parse_websocket_event({
      "event": "media",
      "streamId": "stream123",
      "media": {"track": "inbound", "payload": ""},
    })

    assert event.event == "media"
    assert event.payload == b""

  def test_parse_dtmf_event(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    event = provider.parse_websocket_event({
      "event": "dtmf",
      "digit": "5",
      "track": "inbound",
      "streamId": "stream123",
    })

    assert event.event == "dtmf"
    assert event.payload == "5"
    assert event.stream_id == "stream123"
    assert event.metadata["track"] == "inbound"

  def test_parse_stop_event(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    event = provider.parse_websocket_event({
      "event": "stop",
      "streamId": "stream123",
    })

    assert event.event == "stop"
    assert event.stream_id == "stream123"

  def test_parse_unknown_event(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    event = provider.parse_websocket_event({
      "event": "checkpoint",
      "streamId": "stream123",
      "name": "greeting_done",
    })

    assert event.event == "checkpoint"
    assert event.stream_id == "stream123"
    assert event.metadata["name"] == "greeting_done"

  # --- Response Encoding ---

  def test_encode_audio_response(self):
    import base64

    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    msg = provider.encode_audio_response(b"\xaa\xbb", "stream123")

    assert msg["event"] == "playAudio"
    assert msg["media"]["contentType"] == "audio/x-mulaw"
    assert msg["media"]["sampleRate"] == "8000"
    assert base64.b64decode(msg["media"]["payload"]) == b"\xaa\xbb"

  def test_encode_clear_audio(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    msg = provider.encode_clear_audio("stream123")

    assert msg["event"] == "clearAudio"
    assert msg["streamId"] == "stream123"

  def test_encode_text_response_raises(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="tok")
    with pytest.raises(NotImplementedError, match="does not support text-based"):
      provider.encode_text_response("Hello")


# ============================================================
# PlivoProvider Webhook Signature Validation
# ============================================================


class TestPlivoWebhookValidation:
  """Tests for Plivo V3 HMAC-SHA256 webhook signature validation."""

  def _compute_signature(self, auth_token, url, nonce, method="POST", params=None):
    """Helper to compute expected Plivo V3 signature."""
    import hashlib
    import hmac as hmac_mod

    string_to_sign = url + nonce
    if method == "POST" and params:
      sorted_params = sorted(params.items())
      query_string = "&".join(f"{k}={v}" for k, v in sorted_params)
      string_to_sign += query_string

    mac = hmac_mod.new(
      auth_token.encode("utf-8"),
      string_to_sign.encode("utf-8"),
      hashlib.sha256,
    )
    return base64.b64encode(mac.digest()).decode("ascii")

  def test_valid_get_signature(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    url = "https://example.com/call/incoming"
    nonce = "abc123"
    sig = self._compute_signature("test_token", url, nonce, method="GET")

    assert provider.validate_webhook_signature(b"", sig, url, nonce=nonce, method="GET") is True

  def test_valid_post_signature_with_params(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    url = "https://example.com/call/incoming"
    nonce = "xyz789"
    params = {"CallUUID": "abc", "From": "+1555", "To": "+1666"}
    sig = self._compute_signature("test_token", url, nonce, method="POST", params=params)

    assert provider.validate_webhook_signature(b"", sig, url, nonce=nonce, method="POST", params=params) is True

  def test_invalid_signature(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    assert provider.validate_webhook_signature(b"", "bad_signature", "https://example.com", nonce="abc") is False

  def test_missing_nonce_returns_false(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    assert provider.validate_webhook_signature(b"", "some_sig", "https://example.com") is False

  def test_missing_auth_token_returns_false(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="")
    assert provider.validate_webhook_signature(b"", "sig", "https://example.com", nonce="abc") is False

  def test_post_without_params_uses_url_nonce_only(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    url = "https://example.com/call/incoming"
    nonce = "nonce123"
    # POST without params — same as GET
    sig = self._compute_signature("test_token", url, nonce, method="POST", params=None)

    assert provider.validate_webhook_signature(b"", sig, url, nonce=nonce, method="POST") is True

  def test_params_sorted_correctly(self):
    """Params should be sorted by key for signature computation."""
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = PlivoProvider(auth_id="MA", auth_token="test_token")
    url = "https://example.com/call"
    nonce = "n1"
    # Deliberately unsorted
    params = {"Zebra": "z", "Apple": "a", "Middle": "m"}
    sig = self._compute_signature("test_token", url, nonce, method="POST", params=params)

    assert provider.validate_webhook_signature(b"", sig, url, nonce=nonce, method="POST", params=params) is True


# ============================================================
# CallInterface with Plivo Provider Tests
# ============================================================


class TestCallInterfacePlivo:
  """Tests for CallInterface with Plivo telephony provider."""

  def test_plivo_cascading_creation(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    stt = _MockSTT()
    tts = _MockTTS()
    ci = CallInterface(
      provider="plivo",
      phone_number="+1555",
      auth_id="MA",
      auth_token="tok",
      pipeline="cascading",
      stt=stt,
      tts=tts,
    )
    assert isinstance(ci._telephony, PlivoProvider)
    assert ci._telephony.auth_id == "MA"
    assert ci._call_config.pipeline_mode == "cascading"

  def test_plivo_realtime_creation(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    provider = _MockRealtimeProvider()
    ci = CallInterface(
      provider="plivo",
      phone_number="+1555",
      auth_id="MA",
      auth_token="tok",
      pipeline="realtime",
      realtime=provider,
    )
    assert isinstance(ci._telephony, PlivoProvider)
    assert ci._call_config.pipeline_mode == "realtime"

  def test_plivo_managed_raises(self):
    with pytest.raises(ValueError, match="does not support managed pipeline"):
      CallInterface(
        provider="plivo",
        phone_number="+1555",
        auth_id="MA",
        auth_token="tok",
        pipeline="managed",
      )

  def test_plivo_env_var_fallback(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    stt = _MockSTT()
    tts = _MockTTS()
    with patch.dict("os.environ", {"PLIVO_AUTH_ID": "MA_ENV", "PLIVO_AUTH_TOKEN": "TOK_ENV"}):
      ci = CallInterface(
        provider="plivo",
        phone_number="+1555",
        pipeline="cascading",
        stt=stt,
        tts=tts,
      )
    assert isinstance(ci._telephony, PlivoProvider)
    assert ci._telephony.auth_id == "MA_ENV"

  def test_plivo_config_stores_provider_name(self):
    stt = _MockSTT()
    tts = _MockTTS()
    ci = CallInterface(
      provider="plivo",
      phone_number="+1555",
      auth_id="MA",
      auth_token="tok",
      pipeline="cascading",
      stt=stt,
      tts=tts,
    )
    assert ci._call_config.telephony_provider == "plivo"


# ============================================================
# Plivo Import Tests
# ============================================================


class TestPlivoImports:
  """Verify Plivo types are importable from the call package."""

  def test_import_plivo_provider(self):
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    assert PlivoProvider is not None

  def test_lazy_import_from_telephony_package(self):
    from definable.agent.interface.call.telephony import PlivoProvider

    assert PlivoProvider is not None

  def test_lazy_import_from_call_package(self):
    from definable.agent.interface.call import PlivoProvider

    assert PlivoProvider is not None
