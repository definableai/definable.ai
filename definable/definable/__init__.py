"""Definable — minimal harness for production agents.

Quick start::

    from definable import Agent, tool

    @tool
    def my_tool(x: int) -> int:
      return x * 2

    agent = Agent(name="t", model="anthropic/claude-sonnet-4-6", tools=[my_tool])

    async with agent:
      result = await agent.arun("hello")
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("definable")


# --- Eager exports — the harness public surface ---

from definable.agent.agent import Agent
from definable.agent.core import (
  Event,
  EventBus,
  MemoryAccessed,
  ModelResponded,
  RunCompleted,
  RunErrored,
  RunResult,
  StreamChunkEvent,
  ToolCall,
  ToolCallCompleted,
  ToolCallFailed,
  ToolCallStarted,
  ToolRegistry,
  ToolResult,
  TurnSnapshot,
  TurnStarted,
)
from definable.agent.memory import FileMemory
from definable.agent.toolkit import AsyncToolkit, Function, Toolkit, tool
from definable.exceptions import AgentRunException, ControlFlowException, RetryAgentRun, StopAgentRun, UserError
from definable.media import Audio, File, Image, Video
from definable.model.message import Message


# --- Lazy exports — model providers loaded on first access ---

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
  "OpenAIChat": ("definable.model.openai", "OpenAIChat"),
  "OpenAILike": ("definable.model.openai", "OpenAILike"),
  "DeepSeekChat": ("definable.model.deepseek", "DeepSeekChat"),
  "MoonshotChat": ("definable.model.moonshot", "MoonshotChat"),
  "xAI": ("definable.model.xai", "xAI"),
  "Claude": ("definable.model.anthropic", "Claude"),
  "MistralChat": ("definable.model.mistral", "MistralChat"),
  "Gemini": ("definable.model.google", "Gemini"),
  "Perplexity": ("definable.model.perplexity", "Perplexity"),
  "Ollama": ("definable.model.ollama", "Ollama"),
  "OpenRouter": ("definable.model.openrouter", "OpenRouter"),
  "MCPToolkit": ("definable.agent.mcp", "MCPToolkit"),
  "MCPConfig": ("definable.agent.mcp", "MCPConfig"),
  "MCPServerConfig": ("definable.agent.mcp", "MCPServerConfig"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, attr_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "__version__",
  # Harness
  "Agent",
  "AsyncToolkit",
  "Audio",
  "Event",
  "EventBus",
  "File",
  "FileMemory",
  "Function",
  "Image",
  "MemoryAccessed",
  "Message",
  "ModelResponded",
  "RunCompleted",
  "RunErrored",
  "RunResult",
  "StreamChunkEvent",
  "ToolCall",
  "ToolCallCompleted",
  "ToolCallFailed",
  "ToolCallStarted",
  "ToolRegistry",
  "ToolResult",
  "Toolkit",
  "TurnSnapshot",
  "TurnStarted",
  "Video",
  # Exceptions
  "AgentRunException",
  "ControlFlowException",
  "RetryAgentRun",
  "StopAgentRun",
  "UserError",
  # Tools
  "tool",
  # Lazy — model providers
  "OpenAIChat",  # noqa: F822
  "OpenAILike",  # noqa: F822
  "DeepSeekChat",  # noqa: F822
  "MoonshotChat",  # noqa: F822
  "xAI",  # noqa: F822
  "Claude",  # noqa: F822
  "MistralChat",  # noqa: F822
  "Gemini",  # noqa: F822
  "Perplexity",  # noqa: F822
  "Ollama",  # noqa: F822
  "OpenRouter",  # noqa: F822
  # Lazy — MCP
  "MCPToolkit",  # noqa: F822
  "MCPConfig",  # noqa: F822
  "MCPServerConfig",  # noqa: F822
]
