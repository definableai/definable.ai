"""StorePhase — memory store (awaited, reliable)."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.agent.run.base import BaseRunOutputEvent
from definable.model.message import Message

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class StorePhase(BasePhase):
  """Store messages in memory after execution.

  Wraps Agent._memory_store() but **awaited** instead of fire-and-forget,
  ensuring reliable persistence.

  The phase stores both user message(s) and the assistant response
  for full conversational context.
  """

  _name = "store"
  _requires: set[str] = {"content"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  def should_run(self, state: LoopState) -> bool:
    """Only run if agent has memory configured and enabled."""
    return self._agent._should_store_memory()

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    # Build messages to store: user message(s) + assistant response
    store_messages = list(state.new_messages)
    if state.content:
      store_messages.append(Message(role="assistant", content=state.content))

    events = self._agent._memory_store(store_messages, state.context)
    for evt in events:
      yield state, evt

    # Await pending memory tasks for reliable persistence
    await self._agent._drain_memory_tasks()
