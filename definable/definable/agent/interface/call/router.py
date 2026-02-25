"""FastAPI router factory for the call interface.

Creates the HTTP endpoints that telephony providers connect to:
  - POST /call/incoming — webhook for incoming calls (returns XML)
  - POST /call/status — call status callbacks
  - WS   /call/stream/{call_id} — WebSocket for audio (Media Streams)
  - WS   /call/convo/{call_id} — WebSocket for text (ConversationRelay)
"""

from typing import TYPE_CHECKING, Any

from definable.utils.log import log_debug, log_info

if TYPE_CHECKING:
  from definable.agent.interface.call.interface import CallInterface


def create_call_router(call_interface: "CallInterface") -> Any:
  """Create a FastAPI APIRouter with call interface endpoints.

  This router is mounted on the AgentServer's FastAPI app
  during runtime startup.

  Args:
    call_interface: The CallInterface instance to route requests to.

  Returns:
    A FastAPI APIRouter.

  Raises:
    ImportError: If fastapi is not installed.
  """
  try:
    from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import Response
  except ImportError as e:
    raise ImportError("fastapi is required for the call interface. Install it with: pip install 'definable[call]'") from e

  router = APIRouter(tags=["call"])
  config = call_interface._call_config

  # --- Incoming call webhook ---

  @router.api_route(config.webhook_path, methods=["GET", "POST"])
  async def handle_incoming_call(request: Request) -> Response:
    """Handle incoming call webhook from telephony provider.

    Returns XML (TwiML/Plivo XML) instructing the provider to
    connect the call to our WebSocket endpoint.
    """
    # Extract call metadata from the webhook body
    try:
      if request.method == "POST":
        body = await request.form()
        call_data = dict(body)
      else:
        call_data = dict(request.query_params)
    except Exception:
      call_data = {}

    # Determine the WebSocket URL based on pipeline mode
    host = request.headers.get("host", "localhost")
    scheme = "wss" if request.url.scheme == "https" else "ws"

    call_id = call_data.get("CallSid", call_data.get("CallUUID", "unknown"))

    if config.pipeline_mode == "managed":
      ws_path = f"/call/convo/{call_id}"
    else:
      ws_path = f"/call/stream/{call_id}"

    websocket_url = f"{scheme}://{host}{ws_path}"

    log_info(f"[call] Incoming call {call_id} → {websocket_url}")

    # Generate provider-specific XML
    xml = call_interface._telephony.generate_answer_xml(
      websocket_url,
      welcome_message=config.welcome_message,
      mode="managed" if config.pipeline_mode == "managed" else "stream",
      tts_provider=config.tts_provider,
      stt_provider=config.stt_provider,
      voice=config.voice,
      language=config.language,
      interruptible=config.interruptible,
      interrupt_sensitivity=config.interrupt_sensitivity,
    )

    return Response(content=xml, media_type="application/xml")

  # --- Call status callback ---

  @router.post("/call/status")
  async def handle_call_status(request: Request) -> dict:
    """Handle call status updates from the telephony provider."""
    try:
      body = await request.form()
      status_data = dict(body)
    except Exception:
      status_data = {}

    call_id = str(status_data.get("CallSid", status_data.get("CallUUID", "unknown")))
    status = str(status_data.get("CallStatus", status_data.get("Status", "unknown")))
    log_debug(f"[call] Status update: {call_id} → {status}")

    await call_interface._handle_status_update(call_id, status, {k: str(v) for k, v in status_data.items()})
    return {"status": "ok"}

  # --- WebSocket for ConversationRelay (managed mode) ---

  @router.websocket("/call/convo/{call_id}")
  async def handle_conversation_stream(websocket: WebSocket, call_id: str) -> None:
    """WebSocket endpoint for managed (text-based) pipelines.

    Twilio ConversationRelay connects here. Text flows both directions:
    provider sends transcribed speech, we send response tokens.
    """
    await websocket.accept()
    log_info(f"[call] ConversationRelay WebSocket connected: {call_id}")

    try:
      await call_interface._handle_websocket_call(websocket, call_id)
    except WebSocketDisconnect:
      log_info(f"[call] ConversationRelay WebSocket disconnected: {call_id}")
    except Exception as e:
      log_debug(f"[call] ConversationRelay WebSocket error: {call_id}: {e}")

  # --- WebSocket for Media Streams (cascading/realtime mode) ---

  @router.websocket("/call/stream/{call_id}")
  async def handle_audio_stream(websocket: WebSocket, call_id: str) -> None:
    """WebSocket endpoint for cascading and realtime pipelines.

    Twilio Media Streams or Plivo Audio Streaming connects here.
    Raw audio (mu-law 8kHz) flows both directions.
    """
    await websocket.accept()
    log_info(f"[call] Audio stream WebSocket connected: {call_id}")

    try:
      await call_interface._handle_websocket_call(websocket, call_id)
    except WebSocketDisconnect:
      log_info(f"[call] Audio stream WebSocket disconnected: {call_id}")
    except Exception as e:
      log_debug(f"[call] Audio stream WebSocket error: {call_id}: {e}")

  return router
