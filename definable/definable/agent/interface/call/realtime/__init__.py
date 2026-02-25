"""Real-time speech-to-speech providers for the call interface."""

from definable.agent.interface.call.realtime.base import RealtimeEvent, RealtimeProvider

__all__ = [
  "RealtimeProvider",
  "RealtimeEvent",
]


def __getattr__(name: str):
  if name == "OpenAIRealtimeProvider":
    from definable.agent.interface.call.realtime.openai import OpenAIRealtimeProvider

    return OpenAIRealtimeProvider
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
