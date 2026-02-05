"""
Definable Agents - Production-grade agent framework.

This module provides the Agent class for building LLM-powered agents
with tools, middleware, tracing, and knowledge base support.

Quick Start:
    from definable.agents import Agent, AgentConfig
    from definable.models import OpenAIChat

    agent = Agent(
        model=OpenAIChat(id="gpt-4"),
        tools=[my_tool],
        instructions="You are a helpful assistant.",
    )

    output = agent.run("Hello!")
    print(output.content)

Multi-turn Conversations:
    output1 = agent.run("What is 2+2?")
    output2 = agent.run("And 3+3?", messages=output1.messages)

With Toolkits:
    from definable.agents import Toolkit

    class MyToolkit(Toolkit):
        @property
        def tools(self):
            return [tool1, tool2]

    agent = Agent(model=model, toolkits=[MyToolkit()])

With Knowledge Base (RAG):
    from definable.agents import Agent, AgentConfig, KnowledgeConfig
    from definable.knowledge import Knowledge, Document, InMemoryVectorDB

    # Setup knowledge base
    kb = Knowledge(
        vector_db=InMemoryVectorDB(),
        embedder=VoyageAIEmbedder(api_key="..."),
    )
    kb.add(Document(content="Company policy: Employees get 20 days PTO."))

    # Agent with automatic RAG
    agent = Agent(
        model=model,
        instructions="You are a helpful HR assistant.",
        config=AgentConfig(
            knowledge=KnowledgeConfig(
                knowledge=kb,
                top_k=5,
                rerank=True,
            ),
        ),
    )

    # Agent automatically retrieves relevant context
    response = agent.run("How many PTO days do I get?")

With KnowledgeToolkit (Explicit Search):
    from definable.agents import Agent, KnowledgeToolkit

    agent = Agent(
        model=model,
        toolkits=[KnowledgeToolkit(knowledge=kb, top_k=5)],
        instructions="Search the knowledge base when you need information.",
    )

With Tracing:
    from definable.agents import AgentConfig
    from definable.agents.tracing import TracingConfig, JSONLExporter

    agent = Agent(
        model=model,
        config=AgentConfig(
            tracing=TracingConfig(
                exporters=[JSONLExporter("./traces")]
            )
        )
    )

With Middleware:
    from definable.agents.middleware import LoggingMiddleware, RetryMiddleware

    agent = Agent(model=model)
    agent.use(LoggingMiddleware(logger))
    agent.use(RetryMiddleware(max_retries=3))
"""

from definable.agents.agent import Agent
from definable.agents.config import AgentConfig, KnowledgeConfig, TracingConfig
from definable.agents.middleware import (
  KnowledgeMiddleware,
  LoggingMiddleware,
  MetricsMiddleware,
  Middleware,
  RetryMiddleware,
)
from definable.agents.testing import AgentTestCase, MockModel, create_test_agent
from definable.agents.toolkit import Toolkit
from definable.agents.toolkits import KnowledgeToolkit
from definable.agents.tracing import (
  JSONLExporter,
  NoOpExporter,
  TraceExporter,
  TraceWriter,
)


# Lazy import to avoid circular dependency
def __getattr__(name: str):
  if name == "MCPToolkit":
    from definable.mcp.toolkit import MCPToolkit

    return MCPToolkit
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define MCPToolkit for static analysis (actual import is lazy)
MCPToolkit: type

__all__ = [
  # Core
  "Agent",
  "AgentConfig",
  "TracingConfig",
  "KnowledgeConfig",
  "Toolkit",
  "KnowledgeToolkit",
  "MCPToolkit",
  # Middleware
  "Middleware",
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
