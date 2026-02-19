"""
Definable Agent — Production-grade agent framework.

Quick Start:
    from definable.agent import Agent, AgentConfig
    from definable.model import OpenAIChat

    agent = Agent(
        model=OpenAIChat(id="gpt-4"),
        tools=[my_tool],
        instructions="You are a helpful assistant.",
    )
    output = agent.run("Hello!")

    # String model shorthand:
    agent = Agent(model="gpt-4o-mini", instructions="Hello")

Lego-style blocks snap directly into Agent:
    from definable.knowledge import Knowledge
    from definable.memory import Memory, SQLiteStore
    from definable.vectordb import InMemoryVectorDB

    agent = Agent(
        model="gpt-4o",
        knowledge=Knowledge(vector_db=InMemoryVectorDB(), top_k=5),
        memory=Memory(store=SQLiteStore("./memory.db")),
    )

With Tracing:
    from definable.agent.tracing import Tracing, JSONLExporter

    agent = Agent(
        model="gpt-4o",
        tracing=Tracing(exporters=[JSONLExporter("./traces")]),
    )

With Middleware:
    from definable.agent.middleware import LoggingMiddleware, RetryMiddleware

    agent = Agent(model="gpt-4o")
    agent.use(LoggingMiddleware(logger))
    agent.use(RetryMiddleware(max_retries=3))
"""

from definable.agent.agent import Agent
from definable.agent.cancellation import AgentCancelled, CancellationToken
from definable.agent.config import AgentConfig, CompressionConfig, ReadersConfig
from definable.agent.event_bus import EventBus
from definable.agent.research.config import DeepResearchConfig
from definable.agent.middleware import (
  KnowledgeMiddleware,
  LoggingMiddleware,
  MetricsMiddleware,
  Middleware,
  RetryMiddleware,
  StreamingMiddleware,
)
from definable.agent.testing import AgentTestCase, MockModel, create_test_agent
from definable.agent.toolkit import Toolkit
from definable.agent.toolkits import KnowledgeToolkit
from definable.agent.tracing import (
  JSONLExporter,
  NoOpExporter,
  Tracing,
  TraceExporter,
  TraceWriter,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.guardrail import GuardrailResult, Guardrails
  from definable.agent.reasoning import Thinking
  from definable.mcp.toolkit import MCPToolkit
  from definable.memory import Memory
  from definable.reader import BaseReader as FileReaderRegistry
  from definable.agent.replay import Replay, ReplayComparison


# Lazy import to avoid circular dependency
def __getattr__(name: str):
  if name == "MCPToolkit":
    from definable.mcp.toolkit import MCPToolkit

    return MCPToolkit
  if name == "Memory":
    from definable.memory import Memory

    return Memory
  if name == "FileReaderRegistry":
    from definable.reader import BaseReader

    return BaseReader
  if name == "Replay":
    from definable.agent.replay import Replay

    return Replay
  if name == "ReplayComparison":
    from definable.agent.replay import ReplayComparison

    return ReplayComparison
  if name == "Guardrails":
    from definable.agent.guardrail import Guardrails

    return Guardrails
  if name == "GuardrailResult":
    from definable.agent.guardrail import GuardrailResult

    return GuardrailResult
  if name == "Thinking":
    from definable.agent.reasoning import Thinking

    return Thinking
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "Agent",
  "AgentConfig",
  "AgentCancelled",
  "CancellationToken",
  "EventBus",
  "Tracing",
  "CompressionConfig",
  "ReadersConfig",
  "DeepResearchConfig",
  "FileReaderRegistry",
  "Toolkit",
  "KnowledgeToolkit",
  "MCPToolkit",
  "Memory",
  "Thinking",
  "Replay",
  "ReplayComparison",
  "Guardrails",
  "GuardrailResult",
  # Middleware
  "Middleware",
  "StreamingMiddleware",
  "LoggingMiddleware",
  "RetryMiddleware",
  "MetricsMiddleware",
  "KnowledgeMiddleware",
  # Tracing
  "TraceExporter",
  "TraceWriter",
  "JSONLExporter",
  "NoOpExporter",
  # Testing
  "MockModel",
  "AgentTestCase",
  "create_test_agent",
]
