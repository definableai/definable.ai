"""Text-to-speech providers for the call interface."""

from definable.agent.interface.call.tts.base import TTSProvider

__all__ = [
  "TTSProvider",
]


def __getattr__(name: str):
  if name == "CartesiaTTS":
    from definable.agent.interface.call.tts.cartesia import CartesiaTTS

    return CartesiaTTS
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
