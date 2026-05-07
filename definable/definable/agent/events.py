"""
Definable Events — All agent run event types in one place.

Usage:
    from definable.agent.events import RunContentEvent, ToolCallStartedEvent, RunCompletedEvent
    from definable.agent.events import BrowserActionEvent
"""

from definable.agent.run.base import BaseRunOutputEvent, RunContext, RunStatus
from definable.browser.events import BrowserActionEvent
from definable.agent.run.agent import (
  BaseAgentRunEvent,
  CompressionCompletedEvent,
  CompressionStartedEvent,
  CustomEvent,
  DeepResearchCompletedEvent,
  DeepResearchProgressEvent,
  DeepResearchStartedEvent,
  FileReadCompletedEvent,
  FileReadStartedEvent,
  IntermediateRunContentEvent,
  KnowledgeRetrievalCompletedEvent,
  KnowledgeRetrievalStartedEvent,
  MemoryRecallCompletedEvent,
  MemoryRecallStartedEvent,
  MemoryUpdateCompletedEvent,
  MemoryUpdateStartedEvent,
  ModelCallCompletedEvent,
  ModelCallStartedEvent,
  OutputModelResponseCompletedEvent,
  OutputModelResponseStartedEvent,
  ParserModelResponseCompletedEvent,
  ParserModelResponseStartedEvent,
  PhaseCompletedEvent,
  PhaseStartedEvent,
  PostHookCompletedEvent,
  PostHookStartedEvent,
  PreHookCompletedEvent,
  PreHookStartedEvent,
  ReasoningCompletedEvent,
  ReasoningContentDeltaEvent,
  ReasoningStartedEvent,
  ReasoningStepEvent,
  RunCancelledEvent,
  RunCompletedEvent,
  RunContentCompletedEvent,
  RunContentEvent,
  RunErrorEvent,
  RunEvent,
  RunInput,
  RunOutput,
  RunOutputEvent,
  RunPausedEvent,
  RunStartedEvent,
  SessionSummaryCompletedEvent,
  SessionSummaryStartedEvent,
  SubAgentCompletedEvent,
  SubAgentFailedEvent,
  SubAgentKilledEvent,
  SubAgentSpawnedEvent,
  ToolCallCompletedEvent,
  ToolCallErrorEvent,
  ToolCallStartedEvent,
  ToolContentEvent,
)


__all__ = [
  # Base
  "BaseRunOutputEvent",
  "RunContext",
  "RunStatus",
  # Enums & Types
  "RunEvent",
  "BaseAgentRunEvent",
  "RunOutputEvent",
  "RunInput",
  "RunOutput",
  # Run lifecycle
  "RunStartedEvent",
  "RunContentEvent",
  "RunContentCompletedEvent",
  "IntermediateRunContentEvent",
  "RunCompletedEvent",
  "RunErrorEvent",
  "RunCancelledEvent",
  "RunPausedEvent",
  # Reasoning
  "ReasoningStartedEvent",
  "ReasoningStepEvent",
  "ReasoningContentDeltaEvent",
  "ReasoningCompletedEvent",
  # Tool calls
  "ToolCallStartedEvent",
  "ToolCallCompletedEvent",
  "ToolCallErrorEvent",
  "ToolContentEvent",
  # Knowledge
  "KnowledgeRetrievalStartedEvent",
  "KnowledgeRetrievalCompletedEvent",
  # Memory
  "MemoryRecallStartedEvent",
  "MemoryRecallCompletedEvent",
  "MemoryUpdateStartedEvent",
  "MemoryUpdateCompletedEvent",
  # Model calls
  "ModelCallStartedEvent",
  "ModelCallCompletedEvent",
  # File reads
  "FileReadStartedEvent",
  "FileReadCompletedEvent",
  # Deep research
  "DeepResearchStartedEvent",
  "DeepResearchProgressEvent",
  "DeepResearchCompletedEvent",
  # Session summary
  "SessionSummaryStartedEvent",
  "SessionSummaryCompletedEvent",
  # Hooks
  "PreHookStartedEvent",
  "PreHookCompletedEvent",
  "PostHookStartedEvent",
  "PostHookCompletedEvent",
  # Parser/Output model responses
  "ParserModelResponseStartedEvent",
  "ParserModelResponseCompletedEvent",
  "OutputModelResponseStartedEvent",
  "OutputModelResponseCompletedEvent",
  # Pipeline phases
  "PhaseStartedEvent",
  "PhaseCompletedEvent",
  # Sub-agent
  "SubAgentSpawnedEvent",
  "SubAgentCompletedEvent",
  "SubAgentFailedEvent",
  "SubAgentKilledEvent",
  # Compression
  "CompressionStartedEvent",
  "CompressionCompletedEvent",
  # Custom
  "CustomEvent",
  # Browser
  "BrowserActionEvent",
  # Desktop bridge (lazy-loaded via __getattr__ to avoid circular import)
  "BridgeCallEvent",  # noqa: F822
  "DesktopActionEvent",  # noqa: F822
]


# Lazy imports to avoid circular dependency:
# agent/events → agent/interface/desktop/events → agent/interface/__init__ → agent/events
_LAZY_IMPORTS = {
  "BridgeCallEvent": "definable.agent.interface.desktop.events",
  "DesktopActionEvent": "definable.agent.interface.desktop.events",
}


def __getattr__(name: str):  # noqa: F822
  if name in _LAZY_IMPORTS:
    import importlib

    module = importlib.import_module(_LAZY_IMPORTS[name])
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
