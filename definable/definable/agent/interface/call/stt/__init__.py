"""Speech-to-text providers for the call interface."""

from definable.agent.interface.call.stt.base import STTProvider, Transcript

__all__ = [
  "STTProvider",
  "Transcript",
]


def __getattr__(name: str):
  if name == "DeepgramSTT":
    from definable.agent.interface.call.stt.deepgram import DeepgramSTT

    return DeepgramSTT
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
