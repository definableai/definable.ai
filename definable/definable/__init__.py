"""
Definable — Production-grade agentic framework.

Lego-style DX: composable blocks that snap together.

Quick Start:
    from definable import Agent, tool, OpenAIChat

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[my_tool],
        instructions="You are a helpful assistant.",
    )
    output = agent.run("Hello!")

    # Or with string model shorthand:
    agent = Agent(model="gpt-4o-mini", instructions="Hello")

All Models:
    from definable.model import OpenAIChat, DeepSeekChat, MoonshotChat, xAI

Lego Blocks:
    from definable.knowledge import Knowledge, Document
    from definable.embedder import OpenAIEmbedder, VoyageAIEmbedder
    from definable.chunker import RecursiveChunker
    from definable.reranker import CohereReranker
    from definable.vectordb import InMemoryVectorDB, PgVector
    from definable.memory import Memory, SQLiteStore
    from definable.tool import tool, Function
    from definable.toolkit import Toolkit

Agent-scoped:
    from definable.agent import Agent, AgentConfig
    from definable.agent.tracing import Tracing, JSONLExporter
    from definable.agent.guardrail import Guardrails
    from definable.agent.interface import TelegramInterface
    from definable.agent.reasoning import Thinking
    from definable.agent.research import DeepResearch

Events:
    from definable.agent.events import RunContentEvent, ToolCallStartedEvent, RunCompletedEvent
"""

from typing import TYPE_CHECKING

# --- Eager exports (always loaded — core classes used by every consumer) ---

from definable.agent.agent import Agent
from definable.agent.config import (
  AgentConfig,
  CompressionConfig,
  ReadersConfig,
)
from definable.agent.research.config import DeepResearchConfig
from definable.agent.toolkit import Toolkit
from definable.skill.base import Skill
from definable.tool.decorator import tool
from definable.tool.function import Function
from definable.model.message import Message
from definable.agent.events import RunOutput
from definable.media import Audio, File, Image, Video
from definable.exceptions import AgentRunException, RetryAgentRun, StopAgentRun


if TYPE_CHECKING:
  from definable.agent.guardrail import Guardrails
  from definable.agent.reasoning import Thinking
  from definable.agent.tracing import Tracing
  from definable.knowledge import Document, Knowledge
  from definable.mcp import MCPConfig, MCPServerConfig, MCPToolkit
  from definable.memory import Memory
  from definable.model.deepseek import DeepSeekChat
  from definable.model.moonshot import MoonshotChat
  from definable.model.openai import OpenAIChat, OpenAILike
  from definable.model.xai import xAI
  from definable.model.anthropic import Claude
  from definable.model.mistral import MistralChat
  from definable.model.google import Gemini
  from definable.model.perplexity import Perplexity
  from definable.model.ollama import Ollama
  from definable.model.openrouter import OpenRouter
  from definable.claude_code import ClaudeCodeAgent
  from definable.skill.registry import SkillRegistry


# --- Lazy exports (loaded on first access via __getattr__) ---

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
  # Models
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
  # MCP
  "MCPToolkit": ("definable.mcp", "MCPToolkit"),
  "MCPConfig": ("definable.mcp", "MCPConfig"),
  "MCPServerConfig": ("definable.mcp", "MCPServerConfig"),
  # Memory
  "Memory": ("definable.memory", "Memory"),
  # Knowledge
  "Knowledge": ("definable.knowledge", "Knowledge"),
  "Document": ("definable.knowledge", "Document"),
  # Guardrails
  "Guardrails": ("definable.agent.guardrail", "Guardrails"),
  # Skills
  "SkillRegistry": ("definable.skill.registry", "SkillRegistry"),
  # Claude Code
  "ClaudeCodeAgent": ("definable.claude_code", "ClaudeCodeAgent"),
  # New blocks
  "Thinking": ("definable.agent.reasoning", "Thinking"),
  "Tracing": ("definable.agent.tracing", "Tracing"),
}


def __getattr__(name: str):
  if name in _LAZY_IMPORTS:
    module_path, attr_name = _LAZY_IMPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "Agent",
  "AgentConfig",
  "CompressionConfig",
  "ReadersConfig",
  "DeepResearchConfig",
  "Toolkit",
  "Skill",
  # Tools
  "tool",
  "Function",
  # Messages & Media
  "Message",
  "Image",
  "Audio",
  "Video",
  "File",
  # Run
  "RunOutput",
  # Exceptions
  "AgentRunException",
  "StopAgentRun",
  "RetryAgentRun",
  # Lazy — Models
  "OpenAIChat",
  "OpenAILike",
  "DeepSeekChat",
  "MoonshotChat",
  "xAI",
  "Claude",
  "MistralChat",
  "Gemini",
  "Perplexity",
  "Ollama",
  "OpenRouter",
  # Lazy — MCP
  "MCPToolkit",
  "MCPConfig",
  "MCPServerConfig",
  # Lazy — Memory
  "Memory",
  # Lazy — Knowledge
  "Knowledge",
  "Document",
  # Lazy — Guardrails
  "Guardrails",
  # Lazy — Skills
  "SkillRegistry",
  # Lazy — Claude Code
  "ClaudeCodeAgent",
  # Lazy — New blocks
  "Thinking",
  "Tracing",
]
