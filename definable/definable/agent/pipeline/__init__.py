"""Definable Pipeline — phase-based agent execution pipeline.

The pipeline decomposes the agent execution flow into discrete,
hookable phases that developers can add, remove, replace, reorder,
and intercept.

Quick Start::

    from definable.agent.pipeline import Pipeline, LoopState, ToolRetry

    # Access pipeline from an agent
    agent.pipeline.add_phase(MyPhase(), after="recall")
    agent.pipeline.remove_phase("think")

    # Register hooks
    @agent.hook("before:invoke_loop")
    async def my_hook(state):
        print(f"Messages: {len(state.invoke_messages)}")
        return state

    # Tool retry
    @tool
    def search(query: str) -> str:
        if len(query) < 3:
            raise ToolRetry("Query too short.")
        return do_search(query)

    # Sub-agent spawning
    agent = Agent(model="openai/gpt-4o", sub_agents=True)
"""

from definable.agent.pipeline.debug import DebugConfig
from definable.agent.pipeline.event_stream import EventStream
from definable.agent.pipeline.phase import BasePhase, Phase
from definable.agent.pipeline.pipeline import Pipeline
from definable.agent.pipeline.state import LoopState, LoopStatus, PhaseMetric
from definable.agent.pipeline.sub_agent import SubAgentConfig, SubAgentPolicy, ThreadControlBlock
from definable.agent.pipeline.tool_retry import ToolRetry

__all__ = [
  "BasePhase",
  "DebugConfig",
  "EventStream",
  "LoopState",
  "LoopStatus",
  "Phase",
  "PhaseMetric",
  "Pipeline",
  "SubAgentConfig",
  "SubAgentPolicy",
  "ThreadControlBlock",
  "ToolRetry",
]
