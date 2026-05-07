"""Realtime voice pipeline — audio proxied to speech-to-speech model.

In realtime mode, audio flows directly between Twilio Media Streams
and a speech-to-speech provider (e.g. OpenAI Realtime API).

The provider handles STT, LLM reasoning, and TTS internally.
Function calls are emitted as events — the pipeline invokes the
agent's tools and returns results to the provider.

Three concurrent tasks cooperate:

  - **WebSocket reader** — forwards Twilio audio to the realtime provider
  - **Event listener** — dispatches provider events (audio deltas back to
    Twilio, tool calls to queue, barge-in handling)
  - **Tool handler** — invokes agent tools and sends results back

This is the lowest-latency pipeline mode (~200-300ms TTFB) since
the model processes speech natively — no separate STT/TTS roundtrip.

Flow::

  Caller speaks → Twilio → us → Provider (STT+LLM+TTS) → audio → us → Twilio → Caller hears
  Tool calls:  Provider → tool_call event → Pipeline invokes tool → send_tool_result → Provider continues
"""

import asyncio
import contextlib
import inspect
import json
from typing import TYPE_CHECKING, Any, Dict, List

from definable.agent.interface.call.call import CallEventType, CallSession, CallState
from definable.agent.interface.call.pipeline.base import CallPipeline
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.call.realtime.base import RealtimeProvider
  from definable.agent.interface.call.telephony.base import TelephonyProvider


class RealtimePipeline(CallPipeline):
  """Realtime voice pipeline — speech-to-speech proxy.

  Proxies audio directly between Twilio and a realtime provider
  like the OpenAI Realtime API. The provider handles STT, LLM
  reasoning, and TTS internally.

  Function calling is handled by the pipeline: when the provider
  emits a ``tool_call`` event, the pipeline invokes the matching
  tool from the agent's tool registry and sends the result back.

  Args:
    realtime: Speech-to-speech provider (e.g. OpenAIRealtimeProvider).
  """

  def __init__(self, *, realtime: "RealtimeProvider") -> None:
    self._realtime = realtime

  async def handle_call(
    self,
    websocket: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
  ) -> None:
    """Handle a realtime-mode call over WebSocket.

    Connects to the realtime provider with the agent's instructions
    and tools, then runs three concurrent tasks until the call ends.
    """
    log_info(f"[call] Realtime pipeline started for call {call_session.call_id}")

    # Build tool definitions from the agent's registry
    tool_defs = _build_tool_definitions(agent)

    # Connect to the realtime provider
    try:
      await self._realtime.connect(
        instructions=agent.instructions or "",
        tools=tool_defs,
        voice="alloy",
        input_encoding="mulaw",
        input_sample_rate=8000,
        output_encoding="mulaw",
        output_sample_rate=8000,
      )
    except Exception as e:
      log_error(f"[call] Failed to connect realtime provider: {e}")
      call_session.add_event(CallEventType.ERROR, error=f"Realtime connection failed: {e}")
      return

    # Shared state
    tool_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    playback = _PlaybackState()

    try:
      ws_reader = asyncio.create_task(
        self._read_websocket(websocket, call_session, telephony),
        name=f"ws-reader-{call_session.call_id}",
      )
      event_listener = asyncio.create_task(
        self._listen_events(websocket, call_session, telephony, tool_queue, playback),
        name=f"event-listener-{call_session.call_id}",
      )
      tool_handler = asyncio.create_task(
        self._handle_tool_calls(call_session, agent, tool_queue),
        name=f"tool-handler-{call_session.call_id}",
      )

      done, pending = await asyncio.wait(
        [ws_reader, event_listener, tool_handler],
        return_when=asyncio.FIRST_COMPLETED,
      )

      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)

      for task in done:
        if not task.cancelled():
          exc = task.exception()
          if exc is not None and not isinstance(exc, asyncio.CancelledError):
            log_error(f"[call] Pipeline task {task.get_name()} error: {exc}")

    except asyncio.CancelledError:
      log_info(f"[call] Realtime pipeline cancelled for call {call_session.call_id}")
    except Exception as e:
      log_error(f"[call] Realtime pipeline error: {e}")
      call_session.add_event(CallEventType.ERROR, error=str(e))
    finally:
      await self._realtime.close()
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
    """Read Twilio Media Stream events and forward audio to the realtime provider.

    Audio from ``media`` events is forwarded directly (g711_ulaw → g711_ulaw,
    zero transcoding). Handles ``start``, ``dtmf``, and ``stop`` events.
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
            await self._realtime.send_audio(audio_bytes)

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

  # --- Task 2: Event listener ---

  async def _listen_events(
    self,
    websocket: Any,
    call_session: "CallSession",
    telephony: "TelephonyProvider",
    tool_queue: "asyncio.Queue[Dict[str, Any]]",
    playback: "_PlaybackState",
  ) -> None:
    """Listen for events from the realtime provider and dispatch them.

    - ``audio_delta``: Forward audio to Twilio.
    - ``transcript``: Record user speech in conversation history.
    - ``assistant_transcript``: Record agent speech.
    - ``tool_call``: Queue for the tool handler.
    - ``speech_started``: Barge-in — clear Twilio audio, truncate.
    - ``turn_complete`` / ``interrupted``: Reset playback state.
    - ``error``: Log and record.
    """
    try:
      async for event in self._realtime.receive_events():
        if call_session.state == CallState.ENDED:
          break

        if event.type == "audio_delta":
          # Forward audio to caller via Twilio
          if event.audio and call_session.stream_id:
            playback.active = True
            playback.current_item_id = event.metadata.get("item_id", "")
            playback.audio_ms += _estimate_audio_ms(event.audio)
            msg = telephony.encode_audio_response(event.audio, call_session.stream_id)
            await websocket.send_json(msg)

        elif event.type == "transcript":
          # User speech transcript (from Whisper)
          if event.text:
            call_session.add_user_message(event.text)
            call_session.add_event(CallEventType.UTTERANCE, text=event.text)
            log_debug(f"[call] User said: {event.text[:100]}")

        elif event.type == "assistant_transcript":
          # Assistant speech transcript
          if event.text:
            call_session.add_assistant_message(event.text)
            log_debug(f"[call] Agent said: {event.text[:100]}")

        elif event.type == "tool_call":
          if event.tool_call:
            await tool_queue.put(event.tool_call)

        elif event.type == "speech_started":
          # Barge-in: user started speaking during playback
          if playback.active:
            log_debug("[call] Barge-in detected — clearing audio")
            # Clear Twilio's audio buffer
            if call_session.stream_id:
              clear_msg = telephony.encode_clear_audio(call_session.stream_id)
              try:
                await websocket.send_json(clear_msg)
              except Exception as e:
                log_warning(f"[call] Failed to send clear audio: {e}")

            # Tell provider how much audio was actually played
            if hasattr(self._realtime, "send_truncate") and playback.current_item_id:
              try:
                await self._realtime.send_truncate(playback.current_item_id, playback.audio_ms)
              except Exception as e:
                log_debug(f"[call] Truncate failed: {e}")

            playback.active = False
            playback.audio_ms = 0
            call_session.add_event(CallEventType.INTERRUPTION)

        elif event.type == "speech_stopped":
          log_debug("[call] Speech stopped")

        elif event.type == "turn_complete":
          playback.active = False
          playback.audio_ms = 0

        elif event.type == "interrupted":
          playback.active = False
          playback.audio_ms = 0
          log_debug("[call] Response interrupted by VAD")

        elif event.type == "error":
          log_error(f"[call] Realtime error: {event.text}")
          call_session.add_event(CallEventType.ERROR, error=event.text or "Unknown error")

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] Event listener error: {e}")

  # --- Task 3: Tool handler ---

  async def _handle_tool_calls(
    self,
    call_session: "CallSession",
    agent: "Agent",
    tool_queue: "asyncio.Queue[Dict[str, Any]]",
  ) -> None:
    """Invoke tools when the realtime provider requests them.

    Pops tool_call dicts from the queue, looks up the tool in
    ``agent._tools_dict``, invokes it, and sends the result back
    to the provider via ``send_tool_result()``.
    """
    try:
      while call_session.state != CallState.ENDED:
        try:
          tool_call = await asyncio.wait_for(tool_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
          continue

        call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name", "")
        arguments_str = tool_call.get("arguments", "{}")

        log_debug(f"[call] Tool call: {tool_name}({arguments_str[:100]})")

        try:
          # Parse arguments
          try:
            args = json.loads(arguments_str)
          except json.JSONDecodeError:
            args = {}

          # Look up the tool
          tools_dict = getattr(agent, "_tools_dict", {})
          fn = tools_dict.get(tool_name)

          if fn is None:
            result = f"Error: Unknown tool '{tool_name}'"
            log_warning(f"[call] Unknown tool: {tool_name}")
          elif fn.entrypoint is None:
            result = f"Error: Tool '{tool_name}' has no entrypoint"
          else:
            # Invoke the tool (handles both sync and async)
            raw_result = fn.entrypoint(**args)
            if inspect.isawaitable(raw_result):
              raw_result = await raw_result
            result = str(raw_result)

          # Send result back to the provider
          await self._realtime.send_tool_result(call_id, result)
          log_debug(f"[call] Tool result: {tool_name} → {result[:100]}")

        except Exception as e:
          error_msg = f"Error executing tool '{tool_name}': {e}"
          log_error(f"[call] {error_msg}")
          with contextlib.suppress(Exception):
            await self._realtime.send_tool_result(call_id, error_msg)

    except asyncio.CancelledError:
      raise
    except Exception as e:
      log_warning(f"[call] Tool handler error: {e}")


def _build_tool_definitions(agent: "Agent") -> List[Dict[str, Any]]:
  """Extract tool definitions from the agent in OpenAI function format.

  Converts the agent's registered tools into the format expected
  by the OpenAI Realtime API's ``session.update`` event.

  Returns:
    List of tool definition dicts with type, name, description, parameters.
  """
  tools_dict = getattr(agent, "_tools_dict", {})
  if not tools_dict:
    return []

  tool_defs = []
  for name, fn in tools_dict.items():
    tool_defs.append({
      "type": "function",
      "name": getattr(fn, "name", name),
      "description": getattr(fn, "description", "") or "",
      "parameters": getattr(fn, "parameters", {"type": "object", "properties": {}}),
    })

  return tool_defs


def _estimate_audio_ms(audio_bytes: bytes) -> int:
  """Estimate audio duration in milliseconds for mu-law at 8kHz.

  For mu-law encoding: 1 byte = 1 sample at 8000 Hz = 0.125ms.
  """
  return len(audio_bytes) * 1000 // 8000


class _PlaybackState:
  """Tracks audio playback state for barge-in coordination.

  Stores the current response item ID and cumulative audio duration
  so that ``conversation.item.truncate`` can accurately report how
  much audio the caller actually heard before interruption.
  """

  __slots__ = ("active", "current_item_id", "audio_ms")

  def __init__(self) -> None:
    self.active: bool = False
    self.current_item_id: str = ""
    self.audio_ms: int = 0
