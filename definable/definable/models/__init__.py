from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.models.deepseek import DeepSeekChat
  from definable.models.moonshot import MoonshotChat
  from definable.models.openai import OpenAIChat, OpenAILike
  from definable.models.xai import xAI

__all__ = [
  "OpenAIChat",
  "OpenAILike",
  "DeepSeekChat",
  "MoonshotChat",
  "xAI",
]


def __getattr__(name: str):
  if name == "OpenAIChat":
    from definable.models.openai import OpenAIChat

    return OpenAIChat
  if name == "OpenAILike":
    from definable.models.openai import OpenAILike

    return OpenAILike
  if name == "DeepSeekChat":
    from definable.models.deepseek import DeepSeekChat

    return DeepSeekChat
  if name == "MoonshotChat":
    from definable.models.moonshot import MoonshotChat

    return MoonshotChat
  if name == "xAI":
    from definable.models.xai import xAI

    return xAI
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
