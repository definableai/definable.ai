"""Definable Agent — Production-grade agent framework.

Quick Start::

    from definable.agent import Agent
    agent = Agent(model="gpt-4o-mini", instructions="You are helpful.")
    output = agent.run("Hello!")
"""

from definable.agent.agent import Agent
from definable.agent.compression import Compression
from definable.agent.config import AgentConfig, ReadersConfig
from definable.agent.event_bus import EventBus
from definable.agent.loop import AgentCancelled, CancelToken, CancellationToken, Cancelled
from definable.agent.testing import AgentTestCase, MockModel, create_test_agent
from definable.agent.toolkit import Toolkit
from definable.agent.tracing import (
  DebugExporter,
  JSONLExporter,
  NoOpExporter,
  Tracing,
  TraceExporter,
  TraceWriter,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.observability.config import ObservabilityConfig
  from definable.mcp.toolkit import MCPToolkit
  from definable.memory import Memory
  from definable.reader import BaseReader as FileReaderRegistry


# Lazy imports
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
  if name == "ObservabilityConfig":
    from definable.agent.observability.config import ObservabilityConfig

    return ObservabilityConfig
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "Agent",
  "AgentConfig",
  "AgentCancelled",
  "Cancelled",
  "CancelToken",
  "CancellationToken",
  "EventBus",
  "Tracing",
  "Compression",
  "ReadersConfig",
  "ObservabilityConfig",
  "FileReaderRegistry",
  "Toolkit",
  "MCPToolkit",
  "Memory",
  # Tracing
  "TraceExporter",
  "TraceWriter",
  "JSONLExporter",
  "NoOpExporter",
  "DebugExporter",
  # Testing
  "MockModel",
  "AgentTestCase",
  "create_test_agent",
]
