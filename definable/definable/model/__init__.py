"""
Definable Models — LLM provider models and message types.

Usage:
    from definable.model import OpenAIChat, Message
    from definable.model import DeepSeekChat, MoonshotChat, xAI
"""

from typing import TYPE_CHECKING

from definable.model.message import Citations, Message, MessageReferences
from definable.model.metrics import Metrics
from definable.model.response import ModelResponse, ToolExecution

if TYPE_CHECKING:
  from definable.model.base import Model
  from definable.model.deepseek import DeepSeekChat
  from definable.model.moonshot import MoonshotChat
  from definable.model.openai import OpenAIChat, OpenAILike
  from definable.model.xai import xAI


def __getattr__(name: str):
  if name == "Model":
    from definable.model.base import Model

    return Model
  if name == "OpenAIChat":
    from definable.model.openai import OpenAIChat

    return OpenAIChat
  if name == "OpenAILike":
    from definable.model.openai import OpenAILike

    return OpenAILike
  if name == "DeepSeekChat":
    from definable.model.deepseek import DeepSeekChat

    return DeepSeekChat
  if name == "MoonshotChat":
    from definable.model.moonshot import MoonshotChat

    return MoonshotChat
  if name == "xAI":
    from definable.model.xai import xAI

    return xAI
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Eager — Message types
  "Message",
  "Citations",
  "MessageReferences",
  "Metrics",
  "ModelResponse",
  "ToolExecution",
  # Lazy — Base
  "Model",
  # Lazy — Providers
  "OpenAIChat",
  "OpenAILike",
  "DeepSeekChat",
  "MoonshotChat",
  "xAI",
]
