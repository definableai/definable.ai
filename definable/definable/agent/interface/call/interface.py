"""CallInterface — voice calling interface for connecting agents to phone calls.

Supports multiple telephony providers (Twilio, Plivo) and voice pipeline
strategies (managed, cascading, realtime).

Example (managed mode with Twilio ConversationRelay)::

    from definable.agent import Agent
    from definable.agent.interface.call import CallInterface

    agent = Agent(model="openai/gpt-4o", instructions="You are a phone agent.")
    call = CallInterface(
      agent=agent,
      provider="twilio",
      account_sid="AC...",
      auth_token="...",
      phone_number="+15551234567",
      pipeline="managed",
    )
    async with call:
      await call.serve_forever()
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from definable.agent.interface.base import Interface as BaseInterface
from definable.agent.interface.call.call import CallEventType, CallSession, CallState
from definable.agent.interface.call.config import CallConfig
from definable.agent.interface.call.pipeline.base import CallPipeline
from definable.agent.interface.call.telephony.base import TelephonyProvider
from definable.agent.interface.hooks import InterfaceHook
from definable.agent.interface.message import InterfaceMessage, InterfaceResponse
from definable.agent.interface.session import SessionManager
from definable.utils.log import log_debug, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.interface.call.realtime.base import RealtimeProvider
  from definable.agent.interface.call.stt.base import STTProvider
  from definable.agent.interface.call.tts.base import TTSProvider
  from definable.agent.interface.identity import IdentityResolver


class CallInterface(BaseInterface):
  """Voice calling interface for connecting agents to phone calls.

  Supports three pipeline modes:
    - **managed**: Telephony provider handles STT/TTS (Twilio ConversationRelay).
      Simplest, ~500ms latency.
    - **cascading**: Raw audio → STT → Agent → TTS → audio. Full control,
      pluggable providers. ~800-1200ms latency.
    - **realtime**: Speech-to-speech proxy (OpenAI Realtime API). Lowest
      latency (~200-300ms), but model-locked.

  Routes are mounted on the AgentRuntime's FastAPI server automatically
  when the interface is registered.

  Args:
    provider: Telephony provider ("twilio" or "plivo").
    phone_number: Phone number to receive calls on.
    pipeline: Pipeline mode ("managed", "cascading", or "realtime").
    account_sid: Twilio account SID (or env var TWILIO_ACCOUNT_SID).
    auth_token: Twilio/Plivo auth token (or env var).
    auth_id: Plivo auth ID (or env var PLIVO_AUTH_ID).
    welcome_message: Greeting spoken when a call connects.
    voice: Voice name/ID for TTS.
    language: BCP-47 language code.
    interruptible: When the caller can interrupt ("none", "dtmf", "speech", "any").
    interrupt_sensitivity: Barge-in sensitivity ("low", "medium", "high").
    stt_provider: STT provider name for managed mode.
    tts_provider: TTS provider name for managed mode.
    stt: STTProvider instance for cascading mode.
    tts: TTSProvider instance for cascading mode.
    realtime: RealtimeProvider instance for realtime mode.
    webhook_path: URL path for incoming call webhook.
    stream_path: URL path for WebSocket streams.
    max_call_duration_seconds: Max call duration before hangup.
    agent: Agent instance (or set later via bind()).
    hooks: Interface hooks.
    auth: Auth provider.

  Example::

      call = CallInterface(
        provider="twilio",
        account_sid="AC...",
        auth_token="...",
        phone_number="+15551234567",
        pipeline="managed",
        welcome_message="Hello! How can I help?",
      )
      agent = Agent(model="openai/gpt-4o", interfaces=[call])
  """

  def __init__(
    self,
    *,
    # Telephony
    provider: str = "twilio",
    phone_number: str = "",
    account_sid: str = "",
    auth_token: str = "",
    auth_id: str = "",  # Plivo
    # Pipeline
    pipeline: Literal["managed", "cascading", "realtime"] = "managed",
    # Voice settings
    welcome_message: Optional[str] = None,
    voice: str = "en-US-Standard-A",
    language: str = "en-US",
    interruptible: Literal["none", "dtmf", "speech", "any"] = "any",
    interrupt_sensitivity: Literal["low", "medium", "high"] = "medium",
    # Managed mode
    stt_provider: str = "deepgram",
    tts_provider: str = "google",
    # Cascading mode (pluggable providers)
    stt: Optional["STTProvider"] = None,
    tts: Optional["TTSProvider"] = None,
    # Realtime mode
    realtime: Optional["RealtimeProvider"] = None,
    # Server paths
    webhook_path: str = "/call/incoming",
    stream_path: str = "/call/stream",
    # Call settings
    max_call_duration_seconds: int = 3600,
    # Base config
    max_session_history: int = 50,
    session_ttl_seconds: int = 7200,
    max_concurrent_requests: int = 50,
    error_message: str = "Sorry, I encountered an error. Please try again.",
    # BaseInterface params
    agent: Optional["Agent"] = None,
    session_manager: Optional[SessionManager] = None,
    hooks: Optional[List[InterfaceHook]] = None,
    identity_resolver: Optional["IdentityResolver"] = None,
    auth: Optional[object] = None,
  ) -> None:
    resolved_config = CallConfig(
      telephony_provider=provider,
      phone_number=phone_number,
      pipeline_mode=pipeline,
      welcome_message=welcome_message,
      voice=voice,
      language=language,
      interruptible=interruptible,
      interrupt_sensitivity=interrupt_sensitivity,
      stt_provider=stt_provider,
      tts_provider=tts_provider,
      webhook_path=webhook_path,
      stream_path=stream_path,
      max_call_duration_seconds=max_call_duration_seconds,
      max_session_history=max_session_history,
      session_ttl_seconds=session_ttl_seconds,
      max_concurrent_requests=max_concurrent_requests,
      error_message=error_message,
    )

    super().__init__(
      agent=agent,
      config=resolved_config,
      session_manager=session_manager,
      hooks=hooks,
      identity_resolver=identity_resolver,
      auth=auth,
    )

    self._call_config: CallConfig = self.config  # type: ignore[assignment]
    self._stt = stt
    self._tts = tts
    self._realtime = realtime

    # Validate: Plivo does not support managed mode
    if resolved_config.telephony_provider == "plivo" and resolved_config.pipeline_mode == "managed":
      raise ValueError(
        "Plivo does not support managed pipeline mode (no ConversationRelay equivalent). Use pipeline='cascading' or pipeline='realtime' instead."
      )

    # Active calls tracked by call_id
    self._active_calls: Dict[str, CallSession] = {}

    # Telephony provider (created lazily)
    self._telephony: TelephonyProvider = self._create_telephony_provider(
      provider=resolved_config.telephony_provider,
      account_sid=account_sid,
      auth_token=auth_token,
      auth_id=auth_id,
    )

    # Pipeline (created lazily)
    self._pipeline: CallPipeline = self._create_pipeline(resolved_config.pipeline_mode)

    # The FastAPI router (created on demand by create_router)
    self._router: Optional[Any] = None

  # --- Factory methods ---

  def _create_telephony_provider(
    self,
    provider: str,
    account_sid: str,
    auth_token: str,
    auth_id: str,
  ) -> TelephonyProvider:
    """Create the telephony provider instance."""
    if provider == "twilio":
      from definable.agent.interface.call.telephony.twilio import TwilioProvider

      return TwilioProvider(account_sid=account_sid, auth_token=auth_token)

    if provider == "plivo":
      from definable.agent.interface.call.telephony.plivo import PlivoProvider

      return PlivoProvider(auth_id=auth_id, auth_token=auth_token)

    raise ValueError(f"Unknown telephony provider: {provider!r}")

  def _create_pipeline(self, mode: str) -> CallPipeline:
    """Create the voice pipeline for the configured mode."""
    if mode == "managed":
      from definable.agent.interface.call.pipeline.managed import ManagedPipeline

      return ManagedPipeline()

    if mode == "cascading":
      if self._stt is None or self._tts is None:
        raise ValueError(
          "Cascading pipeline requires stt= and tts= providers. "
          "Example: CallInterface(pipeline='cascading', stt=DeepgramSTT(...), tts=CartesiaTTS(...))"
        )
      from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

      return CascadingPipeline(stt=self._stt, tts=self._tts)

    if mode == "realtime":
      if self._realtime is None:
        raise ValueError(
          "Realtime pipeline requires realtime= provider. Example: CallInterface(pipeline='realtime', realtime=OpenAIRealtimeProvider(...))"
        )
      from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

      return RealtimePipeline(realtime=self._realtime)

    raise ValueError(f"Unknown pipeline mode: {mode!r}")

  # --- Router integration ---

  def create_router(self) -> Any:
    """Create the FastAPI router for this interface.

    Called by AgentServer during app creation to mount call
    endpoints on the shared HTTP server.

    Returns:
      A FastAPI APIRouter.
    """
    if self._router is None:
      from definable.agent.interface.call.router import create_call_router

      self._router = create_call_router(self)
    return self._router

  # --- BaseInterface abstract methods ---

  async def _start_receiver(self) -> None:
    """Start receiving calls.

    For CallInterface, the HTTP routes are registered on the
    AgentRuntime's server. This method logs the readiness state.
    """
    log_info(
      f"[call] CallInterface ready — "
      f"provider={self._call_config.telephony_provider}, "
      f"pipeline={self._call_config.pipeline_mode}, "
      f"phone={self._call_config.phone_number}"
    )

  async def _stop_receiver(self) -> None:
    """Stop receiving calls and clean up active calls."""
    # End all active calls gracefully
    for call_id, call_session in list(self._active_calls.items()):
      if call_session.state != CallState.ENDED:
        call_session.state = CallState.ENDED
        call_session.add_event(CallEventType.CALL_ENDED)
    self._active_calls.clear()
    log_info("[call] CallInterface stopped")

  async def _convert_inbound(self, raw_message: Any) -> Optional[InterfaceMessage]:
    """Convert a call event to an InterfaceMessage.

    For managed pipelines, the raw_message is already a text string.
    This is called by the pipeline when an utterance is ready.
    """
    if isinstance(raw_message, str):
      # Text from managed pipeline
      return InterfaceMessage(
        platform="call",
        platform_user_id=raw_message,
        platform_chat_id=raw_message,
        platform_message_id=raw_message,
        text=raw_message,
      )
    return None

  async def _send_response(
    self,
    original_msg: InterfaceMessage,
    response: InterfaceResponse,
    raw_message: Any,
  ) -> None:
    """Send a response back to the caller.

    For managed pipelines, the pipeline handles sending directly.
    This method is a no-op since the pipeline streams tokens itself.
    """
    pass

  # --- WebSocket call handling ---

  async def _handle_websocket_call(self, websocket: Any, call_id: str) -> None:
    """Handle a WebSocket connection for a call.

    Creates a CallSession, runs the pipeline, and cleans up.

    Args:
      websocket: The FastAPI WebSocket connection.
      call_id: The call identifier from the URL path.
    """
    # Create a call session
    session = self.session_manager.get_or_create(
      platform="call",
      user_id=call_id,
      chat_id=call_id,
    )

    call_session = CallSession(
      call_id=call_id,
      interface_session=session,
    )
    self._active_calls[call_id] = call_session

    # Run hooks
    for hook in self._hooks:
      if hasattr(hook, "on_call_started"):
        try:
          result = await hook.on_call_started(call_session)
          if result is False:
            log_info(f"[call] Hook vetoed call {call_id}")
            return
        except Exception as e:
          log_warning(f"[call] Hook on_call_started error: {e}")

    try:
      assert self.agent is not None
      await self._pipeline.handle_call(
        websocket=websocket,
        call_session=call_session,
        agent=self.agent,
        telephony=self._telephony,
      )
    finally:
      # Run end hooks
      for hook in self._hooks:
        if hasattr(hook, "on_call_ended"):
          try:
            await hook.on_call_ended(call_session)
          except Exception as e:
            log_warning(f"[call] Hook on_call_ended error: {e}")

      # Cleanup
      self._active_calls.pop(call_id, None)

  async def _handle_status_update(
    self,
    call_id: str,
    status: str,
    data: Dict[str, Any],
  ) -> None:
    """Handle a call status callback from the telephony provider.

    Args:
      call_id: The call identifier.
      status: Status string from the provider.
      data: Full status callback data.
    """
    call_session = self._active_calls.get(call_id)
    if call_session is None:
      log_debug(f"[call] Status update for unknown call {call_id}: {status}")
      return

    log_debug(f"[call] Status: {call_id} → {status}")

    if status in ("completed", "failed", "busy", "no-answer", "canceled"):
      call_session.state = CallState.ENDED
      call_session.add_event(CallEventType.CALL_ENDED, status=status)

  # --- Properties ---

  @property
  def active_calls(self) -> Dict[str, CallSession]:
    """Currently active calls indexed by call_id."""
    return dict(self._active_calls)

  @property
  def telephony(self) -> TelephonyProvider:
    """The telephony provider instance."""
    return self._telephony

  @property
  def pipeline(self) -> CallPipeline:
    """The voice pipeline instance."""
    return self._pipeline
