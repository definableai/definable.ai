"""
Definable Agents - Production-grade agent framework.

This module provides the Agent class for building LLM-powered agents
with tools, middleware, and tracing support.

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
from definable.agents.config import AgentConfig, TracingConfig
from definable.agents.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    RetryMiddleware,
)
from definable.agents.testing import AgentTestCase, MockModel, create_test_agent
from definable.agents.toolkit import Toolkit
from definable.agents.tracing import (
    JSONLExporter,
    NoOpExporter,
    TraceExporter,
    TraceWriter,
)

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "TracingConfig",
    "Toolkit",
    # Middleware
    "Middleware",
    "LoggingMiddleware",
    "RetryMiddleware",
    "MetricsMiddleware",
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
