"""RecallPhase — memory + knowledge + readers + research (parallel where possible)."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class RecallPhase(BasePhase):
  """Execute pre-execution pipeline: readers, knowledge, research, memory.

  Wraps Agent._run_pre_execution_pipeline() which handles:
    - Agent._readers_extract()
    - Agent._knowledge_retrieve()
    - Agent._deep_research()
    - Agent._memory_recall()

  Each sub-step populates context fields consumed by later phases.
  """

  _name = "recall"
  _provides: set[str] = {"knowledge_context", "memory_context", "research_context", "readers_context"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  def should_run(self, state: LoopState) -> bool:
    """Only run if agent has knowledge, memory, researcher, or readers."""
    return bool(self._agent._knowledge or self._agent.memory or getattr(self._agent, "_researcher", None) or self._agent.readers)

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    # Delegate to existing pre-execution pipeline
    events = await self._agent._run_pre_execution_pipeline(state.context, state.new_messages, state.all_messages)

    # Propagate context results to state
    state.knowledge_context = state.context.knowledge_context
    state.memory_context = state.context.memory_context
    state.research_context = state.context.research_context
    state.readers_context = state.context.readers_context

    for evt in events:
      yield state, evt
