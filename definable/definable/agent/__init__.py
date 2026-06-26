"""Definable Agent — minimal harness for production agents.

Quick start::

    from definable import Agent

    agent = Agent(
      name="my_agent",
      model="anthropic/claude-sonnet-4-6",
      instructions="Be concise.",
      tools=[my_tool],
      memory=True,
    )

    async with agent:
      result = await agent.arun("hello")
      async for event in await agent.arun("hi", stream=True):
        print(event)

Public surface: Agent + the harness primitives in `agent.core`.
Composition is via Hooks (control plane — mutate/abort) and the EventBus
(observe-only step events; `agent.events.on(EventType)`).
"""

from definable.agent.agent import Agent
from definable.agent.core import (
  AbortRun,
  AgentBegin,
  AgentEnd,
  AgentError,
  Event,
  EventBus,
  Hook,
  ModelHookContext,
  RunResult,
  SkipTool,
  StepBegin,
  StepDelta,
  StepEnd,
  StepType,
  ToolCall,
  ToolHookContext,
  ToolRegistry,
  ToolResult,
)
from definable.agent.memory import FileMemory

__all__ = [
  "AbortRun",
  "Agent",
  "AgentBegin",
  "AgentEnd",
  "AgentError",
  "Event",
  "EventBus",
  "FileMemory",
  "Hook",
  "ModelHookContext",
  "RunResult",
  "SkipTool",
  "StepBegin",
  "StepDelta",
  "StepEnd",
  "StepType",
  "ToolCall",
  "ToolHookContext",
  "ToolRegistry",
  "ToolResult",
]
