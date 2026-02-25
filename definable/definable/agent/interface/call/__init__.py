"""
Definable Call Interface — Connect agents to voice calls.

Supports multiple telephony providers (Twilio, Plivo) and voice
pipeline strategies (managed, cascading, realtime) for production-grade
voice AI agents.

Quick Start (Twilio ConversationRelay)::

    from definable.agent import Agent
    from definable.agent.interface.call import CallInterface

    agent = Agent(model="openai/gpt-4o", instructions="You are a phone agent.")
    call = CallInterface(
      agent=agent,
      provider="twilio",
      account_sid="AC...",
      auth_token="...",
      phone_number="+15551234567",
      welcome_message="Hello! How can I help?",
    )

    # Run with AgentRuntime (recommended — shared HTTP server):
    from definable.agent.runtime import AgentRuntime
    runtime = AgentRuntime(agent, interfaces=[call])
    await runtime.start()
"""

from definable.agent.interface.call.call import CallEvent, CallEventType, CallSession, CallState
from definable.agent.interface.call.config import CallConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.interface.call.interface import CallInterface
  from definable.agent.interface.call.pipeline.base import CallPipeline
  from definable.agent.interface.call.pipeline.cascading import CascadingPipeline
  from definable.agent.interface.call.pipeline.managed import ManagedPipeline
  from definable.agent.interface.call.pipeline.realtime import RealtimePipeline
  from definable.agent.interface.call.realtime.base import RealtimeEvent, RealtimeProvider
  from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider
  from definable.agent.interface.call.stt.base import STTProvider, Transcript
  from definable.agent.interface.call.stt.deepgram import DeepgramSTT
  from definable.agent.interface.call.telephony.base import CallInfo, TelephonyEvent, TelephonyProvider
  from definable.agent.interface.call.telephony.plivo import PlivoProvider
  from definable.agent.interface.call.tts.base import TTSProvider
  from definable.agent.interface.call.tts.cartesia import CartesiaTTS


def __getattr__(name: str):
  if name == "CallInterface":
    from definable.agent.interface.call.interface import CallInterface

    return CallInterface
  if name == "TelephonyProvider":
    from definable.agent.interface.call.telephony.base import TelephonyProvider

    return TelephonyProvider
  if name == "TelephonyEvent":
    from definable.agent.interface.call.telephony.base import TelephonyEvent

    return TelephonyEvent
  if name == "CallInfo":
    from definable.agent.interface.call.telephony.base import CallInfo

    return CallInfo
  if name == "TwilioProvider":
    from definable.agent.interface.call.telephony.twilio import TwilioProvider

    return TwilioProvider
  if name == "PlivoProvider":
    from definable.agent.interface.call.telephony.plivo import PlivoProvider

    return PlivoProvider
  if name == "CallPipeline":
    from definable.agent.interface.call.pipeline.base import CallPipeline

    return CallPipeline
  if name == "ManagedPipeline":
    from definable.agent.interface.call.pipeline.managed import ManagedPipeline

    return ManagedPipeline
  if name == "CascadingPipeline":
    from definable.agent.interface.call.pipeline.cascading import CascadingPipeline

    return CascadingPipeline
  if name == "RealtimePipeline":
    from definable.agent.interface.call.pipeline.realtime import RealtimePipeline

    return RealtimePipeline
  if name == "OpenAIRealtimeProvider":
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    return OpenAIRealtimeProvider
  if name == "DeepgramSTT":
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    return DeepgramSTT
  if name == "CartesiaTTS":
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    return CartesiaTTS
  if name == "STTProvider":
    from definable.agent.interface.call.stt.base import STTProvider

    return STTProvider
  if name == "Transcript":
    from definable.agent.interface.call.stt.base import Transcript

    return Transcript
  if name == "TTSProvider":
    from definable.agent.interface.call.tts.base import TTSProvider

    return TTSProvider
  if name == "RealtimeProvider":
    from definable.agent.interface.call.realtime.base import RealtimeProvider

    return RealtimeProvider
  if name == "RealtimeEvent":
    from definable.agent.interface.call.realtime.base import RealtimeEvent

    return RealtimeEvent
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "CallInterface",
  "CallConfig",
  "CallSession",
  "CallState",
  "CallEvent",
  "CallEventType",
  # Telephony
  "TelephonyProvider",
  "TelephonyEvent",
  "CallInfo",
  "PlivoProvider",
  # Pipeline
  "CallPipeline",
  "ManagedPipeline",
  "CascadingPipeline",
  "RealtimePipeline",
  # STT
  "STTProvider",
  "Transcript",
  "DeepgramSTT",
  # TTS
  "TTSProvider",
  "CartesiaTTS",
  # Realtime
  "RealtimeProvider",
  "RealtimeEvent",
  "OpenAIRealtimeProvider",
]
