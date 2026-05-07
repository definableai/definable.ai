"""Managed voice pipeline — provider handles STT/TTS (e.g. Twilio ConversationRelay).

In managed mode, the telephony provider transcribes caller speech and
synthesizes agent responses. We receive and send **text** over WebSocket,
which maps directly to Agent.arun().

Flow:
  Caller speaks → Provider STT → {"type": "prompt", "voicePrompt": "..."} → us
  us → Agent.arun(text) → stream tokens → {"type": "text", "token": "..."} → Provider TTS → Caller hears
"""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from definable.agent.interface.call.call import CallEventType, CallSession, CallState
from definable.agent.interface.call.pipeline.base import CallPipeline
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.call.telephony.base import TelephonyProvider


class ManagedPipeline(CallPipeline):
  """Managed voice pipeline using provider-handled STT/TTS.

  Works with Twilio ConversationRelay. The provider sends transcribed
  caller speech as text, and we send back text tokens that the provider
  synthesizes into audio.

  This is the simplest and lowest-latency pipeline — it maps directly
  to Agent.arun() with no audio processing on our side.
  """

  async def handle_call(
    self,
    websocket: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
  ) -> None:
    """Handle a managed-mode call over WebSocket.

    Loops over incoming WebSocket messages, dispatching each
    event type to the appropriate handler.
    """
    log_info(f"[call] Managed pipeline started for call {call_session.call_id}")

    try:
      while call_session.state != CallState.ENDED:
        try:
          raw = await websocket.receive_text()
        except Exception:
          # WebSocket closed
          break

        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          log_warning(f"[call] Non-JSON WebSocket message: {raw[:100]}")
          continue

        event = telephony.parse_websocket_event(data)

        if event.event == "setup":
          await self._handle_setup(event, call_session)

        elif event.event == "prompt":
          await self._handle_prompt(event, call_session, agent, telephony, websocket)

        elif event.event == "interrupt":
          await self._handle_interrupt(event, call_session)

        elif event.event == "dtmf":
          call_session.add_event(CallEventType.DTMF, digit=event.payload)
          log_debug(f"[call] DTMF digit: {event.payload}")

        elif event.event in ("stop", "hangup"):
          call_session.state = CallState.ENDED
          call_session.add_event(CallEventType.CALL_ENDED)
          log_info(f"[call] Call ended: {call_session.call_id}")

    except asyncio.CancelledError:
      log_info(f"[call] Pipeline cancelled for call {call_session.call_id}")
    except Exception as e:
      log_error(f"[call] Pipeline error for call {call_session.call_id}: {e}")
      call_session.add_event(CallEventType.ERROR, error=str(e))
    finally:
      if call_session.state != CallState.ENDED:
        call_session.state = CallState.ENDED
        call_session.add_event(CallEventType.CALL_ENDED)

  async def _handle_setup(self, event: Any, call_session: "CallSession") -> None:
    """Handle the initial setup event when the WebSocket connects."""
    if event.call_id:
      call_session.call_id = event.call_id
    call_session.state = CallState.ACTIVE
    call_session.add_event(CallEventType.CALL_STARTED)
    log_info(f"[call] Call connected: {call_session.call_id}")

  async def _handle_prompt(
    self,
    event: Any,
    call_session: "CallSession",
    agent: "Agent",
    telephony: "TelephonyProvider",
    websocket: Any,
  ) -> None:
    """Handle a transcribed caller utterance — invoke agent and stream response."""
    user_text = event.payload
    if not user_text or not user_text.strip():
      return

    log_debug(f"[call] User said: {user_text[:100]}")
    call_session.add_user_message(user_text)
    call_session.add_event(CallEventType.UTTERANCE, text=user_text)

    try:
      # Build the message history from the call session for multi-turn
      session = call_session.interface_session

      # Run the agent
      run_output = await agent.arun(
        instruction=user_text,
        messages=session.messages if session else None,
        session_id=f"call:{call_session.call_id}",
        user_id=call_session.from_number or call_session.call_id,
      )

      response_text = str(run_output.content) if run_output.content else ""

      if response_text:
        # Stream the response as tokens to the provider
        # For now, send the full response. Streaming integration
        # with arun_stream will be added when the pipeline matures.
        await self._send_text_response(websocket, telephony, response_text)
        call_session.add_assistant_message(response_text)

      # Update the interface session with the run output
      if session and run_output.messages:
        session.messages = list(run_output.messages)
        session.last_run_output = run_output
        session.touch()

    except Exception as e:
      log_error(f"[call] Agent error during call {call_session.call_id}: {e}")
      # Send an error response to the caller
      error_msg = "Sorry, I encountered an error. Let me try again."
      await self._send_text_response(websocket, telephony, error_msg)

  async def _handle_interrupt(self, event: Any, call_session: "CallSession") -> None:
    """Handle caller interrupting the agent's response.

    Truncates the last assistant message to what was actually spoken
    before the interruption.
    """
    spoken_text = event.payload or ""
    call_session.truncate_last_assistant(spoken_text)
    call_session.add_event(CallEventType.INTERRUPTION, spoken_text=spoken_text)
    log_debug(f"[call] Interrupted — spoken so far: {spoken_text[:80]}")

  async def _send_text_response(
    self,
    websocket: Any,
    telephony: "TelephonyProvider",
    text: str,
  ) -> None:
    """Send a text response through the WebSocket.

    Sends the response as a sequence of tokens. For managed pipelines,
    the provider synthesizes these into audio.

    Chunks the text into sentence-sized pieces for more natural
    speech pacing.
    """
    chunks = _split_into_speech_chunks(text)

    for i, chunk in enumerate(chunks):
      is_last = i == len(chunks) - 1
      msg = telephony.encode_text_response(chunk, last=is_last)
      await websocket.send_json(msg)

    # If the text was empty or had no chunks, send an empty last token
    if not chunks:
      msg = telephony.encode_text_response("", last=True)
      await websocket.send_json(msg)


def _split_into_speech_chunks(text: str) -> list:
  """Split text into speech-friendly chunks at sentence boundaries.

  ConversationRelay streams tokens to TTS. Sending at sentence
  boundaries produces more natural speech pacing.

  Args:
    text: Full response text.

  Returns:
    List of text chunks.
  """
  if not text:
    return []

  # Split on sentence-ending punctuation followed by space
  chunks = []
  current = ""
  for char in text:
    current += char
    if char in ".!?" and len(current) > 10:
      chunks.append(current)
      current = ""

  if current.strip():
    chunks.append(current)

  return chunks
