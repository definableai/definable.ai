"""Cascading voice pipeline — raw audio → STT → Agent → TTS → audio.

In cascading mode, we handle the full audio processing chain:

  1. Receive raw mu-law audio from Twilio Media Streams WebSocket
  2. Forward audio to STT provider (e.g. Deepgram) for transcription
  3. When a complete utterance is detected, invoke Agent.arun()
  4. Stream the agent's response through TTS provider (e.g. Cartesia)
  5. Send synthesized audio back through the WebSocket to Twilio

Three concurrent tasks cooperate:

  - **WebSocket reader** — receives Twilio events, forwards audio to STT
  - **STT listener** — receives transcripts, detects barge-in during playback,
    pushes final utterances to a queue
  - **Response handler** — pops utterances, runs the agent, streams TTS audio back

Barge-in detection uses the STT transcript stream: if the STT provider
reports speech while TTS audio is playing, we interrupt playback and
clear Twilio's audio buffer.

Flow::

  Caller speaks → Twilio Media Stream → us → STT → text
  → Agent.arun() → TTS → audio → Twilio → Caller hears
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from definable.agent.interface.call.call import CallEventType, CallSession, CallState
from definable.agent.interface.call.pipeline.base import CallPipeline
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.call.stt.base import STTProvider
  from definable.agent.interface.call.telephony.base import TelephonyProvider
  from definable.agent.interface.call.tts.base import TTSProvider


class CascadingPipeline(CallPipeline):
  """Cascading voice pipeline — STT → Agent → TTS.

  Full control over each component. Supports any combination of
  pluggable STT and TTS providers.

  Higher latency than managed (~800-1200ms) but maximum flexibility
  and complete provider independence.

  Args:
    stt: Speech-to-text provider (e.g. DeepgramSTT).
    tts: Text-to-speech provider (e.g. CartesiaTTS).
    encoding: Audio encoding for STT/TTS ("mulaw", "linear16").
    sample_rate: Audio sample rate in Hz (8000 for telephony).
  """

  def __init__(
    self,
    *,
    stt: "STTProvider",
    tts: "TTSProvider",
    encoding: str = "mulaw",
    sample_rate: int = 8000,
  ) -> None:
    self._stt = stt
    self._tts = tts
    self._encoding = encoding
    self._sample_rate = sample_rate

  async def handle_call(
    self,
    websocket: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
  ) -> None:
    """Handle a cascading-mode call over WebSocket.

    Connects to the STT provider, then runs three concurrent tasks
    until the call ends or a task fails.
    """
    log_info(f"[call] Cascading pipeline started for call {call_session.call_id}")

    # Queue for completed utterances (STT listener → response handler)
    utterance_queue: asyncio.Queue[str] = asyncio.Queue()

    # Shared playback state for barge-in detection
    playback = _PlaybackState()

    # Connect STT
    try:
      await self._stt.connect(
        sample_rate=self._sample_rate,
        encoding=self._encoding,
      )
    except Exception as e:
      log_error(f"[call] Failed to connect STT: {e}")
      call_session.add_event(CallEventType.ERROR, error=f"STT connection failed: {e}")
      return

    try:
      ws_reader = asyncio.create_task(
        self._read_websocket(websocket, call_session, telephony),
        name=f"ws-reader-{call_session.call_id}",
      )
      stt_listener = asyncio.create_task(
        self._listen_stt(websocket, call_session, telephony, utterance_queue, playback),
        name=f"stt-listener-{call_session.call_id}",
      )
      response_handler = asyncio.create_task(
        self._handle_responses(websocket, call_session, agent, telephony, utterance_queue, playback),
        name=f"response-handler-{call_session.call_id}",
      )

      done, pending = await asyncio.wait(
        [ws_reader, stt_listener, response_handler],
        return_when=asyncio.FIRST_COMPLETED,
      )

      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)

      # Log errors from completed tasks
      for task in done:
        if not task.cancelled():
          exc = task.exception()
          if exc is not None and not isinstance(exc, asyncio.CancelledError):
            log_error(f"[call] Pipeline task {task.get_name()} error: {exc}")

    except asyncio.CancelledError:
      log_info(f"[call] Cascading pipeline cancelled for call {call_session.call_id}")
    except Exception as e:
      log_error(f"[call] Cascading pipeline error: {e}")
      call_session.add_event(CallEventType.ERROR, error=str(e))
    finally:
      await self._cleanup()
      if call_session.state != CallState.ENDED:
        call_session.state = CallState.ENDED
        call_session.add_event(CallEventType.CALL_ENDED)

  # --- Task 1: WebSocket reader ---

  async def _read_websocket(
    self,
    websocket: Any,
    call_session: "CallSession",
    telephony: "TelephonyProvider",
  ) -> None:
    """Read events from Twilio Media Streams and forward audio to STT.

    Handles ``start``, ``media``, ``dtmf``, and ``stop`` events.
    Audio chunks from ``media`` events are forwarded to the STT
    provider for transcription.
    """
    try:
      while call_session.state != CallState.ENDED:
        try:
          raw = await websocket.receive_text()
        except Exception:
          break

        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          continue

        event = telephony.parse_websocket_event(data)

        if event.event == "start":
          call_session.stream_id = event.stream_id
          call_session.call_id = event.call_id or call_session.call_id
          call_session.state = CallState.ACTIVE
          call_session.add_event(CallEventType.CALL_STARTED)
          log_info(f"[call] Media stream started: stream={event.stream_id}, call={event.call_id}")

        elif event.event == "media":
          audio_bytes = event.payload
          if isinstance(audio_bytes, bytes) and audio_bytes:
            await self._stt.send_audio(audio_bytes)

        elif event.event == "dtmf":
          call_session.add_event(CallEventType.DTMF, digit=event.payload)
          log_debug(f"[call] DTMF: {event.payload}")

        elif event.event == "stop":
          call_session.state = CallState.ENDED
          call_session.add_event(CallEventType.CALL_ENDED)
          log_info(f"[call] Media stream stopped: {call_session.call_id}")
          break

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] WebSocket read error: {e}")

  # --- Task 2: STT listener ---

  async def _listen_stt(
    self,
    websocket: Any,
    call_session: "CallSession",
    telephony: "TelephonyProvider",
    utterance_queue: "asyncio.Queue[str]",
    playback: "_PlaybackState",
  ) -> None:
    """Listen for transcription results and handle barge-in.

    When a final transcript arrives, it's pushed to the utterance
    queue for the response handler to process.

    If any speech is detected during TTS playback (``playback.active``
    is True), triggers barge-in: clears Twilio's audio buffer and
    signals the TTS streamer to stop.
    """
    try:
      async for transcript in self._stt.receive_transcripts():
        if call_session.state == CallState.ENDED:
          break

        # Empty text = UtteranceEnd marker (silence signal)
        if not transcript.text:
          continue

        # Barge-in: speech detected while TTS is playing
        if playback.active and not playback.interrupted:
          playback.interrupted = True
          call_session.add_event(CallEventType.INTERRUPTION, spoken_text="")
          log_debug("[call] Barge-in — speech during playback")

          # Clear Twilio's audio buffer
          if call_session.stream_id:
            clear_msg = telephony.encode_clear_audio(call_session.stream_id)
            try:
              await websocket.send_json(clear_msg)
            except Exception as e:
              log_warning(f"[call] Failed to send clear audio: {e}")

        if transcript.is_final:
          log_debug(f"[call] Final transcript: {transcript.text[:100]}")
          call_session.add_user_message(transcript.text)
          call_session.add_event(CallEventType.UTTERANCE, text=transcript.text)
          await utterance_queue.put(transcript.text)

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] STT listen error: {e}")

  # --- Task 3: Response handler ---

  async def _handle_responses(
    self,
    websocket: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
    utterance_queue: "asyncio.Queue[str]",
    playback: "_PlaybackState",
  ) -> None:
    """Process utterances: invoke agent and stream TTS audio back.

    Waits for completed utterances from the STT listener, runs the
    agent, synthesizes the response, and streams audio chunks back
    through the WebSocket to Twilio.
    """
    try:
      while call_session.state != CallState.ENDED:
        try:
          user_text = await asyncio.wait_for(utterance_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
          continue

        if not user_text.strip():
          continue

        log_debug(f"[call] Processing: {user_text[:100]}")

        try:
          # Invoke the agent
          session = call_session.interface_session
          run_output = await agent.arun(
            instruction=user_text,
            messages=session.messages if session else None,
            session_id=f"call:{call_session.call_id}",
            user_id=call_session.from_number or call_session.call_id,
          )

          response_text = str(run_output.content) if run_output.content else ""

          if response_text:
            call_session.add_assistant_message(response_text)

            # Synthesize and stream audio back
            await self._stream_tts(
              websocket,
              telephony,
              call_session,
              response_text,
              playback,
            )

          # Update session history
          if session and run_output.messages:
            session.messages = list(run_output.messages)
            session.last_run_output = run_output
            session.touch()

        except Exception as e:
          log_error(f"[call] Agent/TTS error: {e}")
          call_session.add_event(CallEventType.ERROR, error=str(e))

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] Response handler error: {e}")

  # --- TTS streaming ---

  async def _stream_tts(
    self,
    websocket: Any,
    telephony: "TelephonyProvider",
    call_session: "CallSession",
    text: str,
    playback: "_PlaybackState",
  ) -> None:
    """Synthesize text to speech and stream audio chunks to Twilio.

    Marks playback as active so barge-in detection works.
    Stops early if barge-in is detected.
    """
    stream_id = call_session.stream_id
    if not stream_id:
      log_warning("[call] No stream_id — cannot send audio")
      return

    playback.active = True
    playback.interrupted = False

    try:
      async for audio_chunk in self._tts.synthesize_stream(
        text,
        encoding=self._encoding,
        sample_rate=self._sample_rate,
      ):
        # Check for barge-in or call end
        if playback.interrupted or call_session.state == CallState.ENDED:
          log_debug("[call] TTS playback stopped")
          break

        msg = telephony.encode_audio_response(audio_chunk, stream_id)
        await websocket.send_json(msg)

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] TTS stream error: {e}")
    finally:
      playback.active = False

      # If interrupted, truncate conversation history
      if playback.interrupted:
        call_session.truncate_last_assistant("")

  # --- Cleanup ---

  async def _cleanup(self) -> None:
    """Close STT and TTS provider connections."""
    try:
      await self._stt.close()
    except Exception as e:
      log_debug(f"[call] STT close error: {e}")

    try:
      await self._tts.close()
    except Exception as e:
      log_debug(f"[call] TTS close error: {e}")


class _PlaybackState:
  """Tracks TTS playback state for barge-in coordination.

  Shared between the STT listener (sets ``interrupted``) and the
  TTS streamer (checks ``interrupted``, sets ``active``).

  Thread-safe in asyncio since all access is within a single event loop.
  """

  __slots__ = ("active", "interrupted")

  def __init__(self) -> None:
    self.active: bool = False
    self.interrupted: bool = False
