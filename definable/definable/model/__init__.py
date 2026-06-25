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
  from definable.model.anthropic import Claude
  from definable.model.base import Model
  from definable.model.deepseek import DeepSeekChat
  from definable.model.google import Gemini
  from definable.model.moonshot import MoonshotChat
  from definable.model.openai import OpenAIChat, OpenAILike
  from definable.model.openrouter import OpenRouter
  from definable.model.perplexity import Perplexity
  from definable.model.utils import resolve_model_string as resolve_model_string
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
  if name == "Claude":
    from definable.model.anthropic import Claude

    return Claude

  if name == "Gemini":
    from definable.model.google import Gemini

    return Gemini
  if name == "Perplexity":
    from definable.model.perplexity import Perplexity

    return Perplexity
  if name == "OpenRouter":
    from definable.model.openrouter import OpenRouter

    return OpenRouter

  if name == "resolve_model_string":
    from definable.model.utils import resolve_model_string

    return resolve_model_string
  # --- Test guard ---
  if name == "override_allow_model_requests":
    from definable.model.base import override_allow_model_requests

    return override_allow_model_requests
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
  "Claude",
  "Gemini",
  "Perplexity",
  "OpenRouter",
  # Lazy — Utilities
  "resolve_model_string",
  # Lazy — Test guard
  "override_allow_model_requests",  # noqa: F822
  # Lazy — Resilience
  "ResilientModel",  # noqa: F822
  "KeyPool",  # noqa: F822
  "FailoverChain",  # noqa: F822
  "FailoverEntry",  # noqa: F822
]
