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
from definable.agent.compression import Compression
from definable.agent.config import AgentConfig, ReadersConfig
from definable.agent.context import Context, TokenBudget
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
  DebugExporter,
  JSONLExporter,
  NoOpExporter,
  Tracing,
  TraceExporter,
  TraceWriter,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.guardrail import GuardrailResult, Guardrails
  from definable.agent.observability.config import ObservabilityConfig
  from definable.agent.pipeline import (
    BasePhase,
    DebugConfig,
    LoopState,
    LoopStatus,
    Phase,
    Pipeline,
    SubAgentPolicy,
    ToolRetry,
  )
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
  if name == "ObservabilityConfig":
    from definable.agent.observability.config import ObservabilityConfig

    return ObservabilityConfig
  if name in ("Pipeline", "Phase", "BasePhase", "LoopState", "LoopStatus", "ToolRetry", "DebugConfig", "SubAgentPolicy"):
    from definable.agent import pipeline as _pipeline

    return getattr(_pipeline, name)
  if name in ("Team", "TeamMode"):
    from definable.agent import team as _team

    return getattr(_team, name)
  if name in ("Workflow", "Step", "Steps", "Parallel", "Loop", "Condition", "Router"):
    from definable.agent import workflow as _workflow

    return getattr(_workflow, name)
  _eval_names = (
    "BaseEval",
    "EvalCase",
    "EvalSuite",
    "AccuracyEval",
    "PerformanceEval",
    "ReliabilityEval",
    "AgentAsJudgeEval",
    "EvalResult",
    "AccuracyResult",
    "PerformanceResult",
    "ReliabilityResult",
    "JudgeResult",
  )
  if name in _eval_names:
    from definable.agent import eval as _eval

    return getattr(_eval, name)
  if name in ("SecurityConfig", "ToolPolicy", "SecurityReport", "SecurityFinding", "SecuritySeverity"):
    from definable.agent import security as _security

    return getattr(_security, name)
  if name == "UsageTracker":
    from definable.agent.usage import UsageTracker

    return UsageTracker
  if name == "UsageSnapshot":
    from definable.agent.usage import UsageSnapshot

    return UsageSnapshot
  # --- Scheduler ---
  if name in ("Scheduler", "ScheduledJob", "JobStatus"):
    from definable.agent import scheduler as _scheduler

    return getattr(_scheduler, name)
  if name in ("Interval", "OneShot"):
    from definable.agent import trigger as _trigger

    return getattr(_trigger, name)
  # --- Plugins ---
  if name in ("Plugin", "PluginRegistry"):
    from definable.agent import plugin as _plugin

    return getattr(_plugin, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  # Core
  "Agent",
  "AgentConfig",
  "AgentCancelled",
  "CancellationToken",
  "EventBus",
  "Tracing",
  "Compression",
  "Context",
  "TokenBudget",
  "ReadersConfig",
  "DeepResearchConfig",
  "ObservabilityConfig",
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
  "DebugExporter",
  # Testing
  "MockModel",
  "AgentTestCase",
  "create_test_agent",
  # Pipeline
  "Pipeline",
  "Phase",
  "BasePhase",
  "LoopState",
  "LoopStatus",
  "ToolRetry",
  "DebugConfig",
  "SubAgentPolicy",
  # Team
  "Team",  # noqa: F822
  "TeamMode",  # noqa: F822
  # Workflow
  "Workflow",  # noqa: F822
  "Step",  # noqa: F822
  "Steps",  # noqa: F822
  "Parallel",  # noqa: F822
  "Loop",  # noqa: F822
  "Condition",  # noqa: F822
  "Router",  # noqa: F822
  # Eval
  "BaseEval",  # noqa: F822
  "EvalCase",  # noqa: F822
  "EvalSuite",  # noqa: F822
  "AccuracyEval",  # noqa: F822
  "PerformanceEval",  # noqa: F822
  "ReliabilityEval",  # noqa: F822
  "AgentAsJudgeEval",  # noqa: F822
  "EvalResult",  # noqa: F822
  "AccuracyResult",  # noqa: F822
  "PerformanceResult",  # noqa: F822
  "ReliabilityResult",  # noqa: F822
  "JudgeResult",  # noqa: F822
  # Security
  "SecurityConfig",  # noqa: F822
  "ToolPolicy",  # noqa: F822
  "SecurityReport",  # noqa: F822
  "SecurityFinding",  # noqa: F822
  "SecuritySeverity",  # noqa: F822
  # Usage tracking
  "UsageTracker",  # noqa: F822
  "UsageSnapshot",  # noqa: F822
  # Scheduler
  "Scheduler",  # noqa: F822
  "ScheduledJob",  # noqa: F822
  "JobStatus",  # noqa: F822
  # Triggers
  "Interval",  # noqa: F822
  "OneShot",  # noqa: F822
  # Plugins
  "Plugin",  # noqa: F822
  "PluginRegistry",  # noqa: F822
]
