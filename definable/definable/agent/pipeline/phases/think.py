"""ThinkPhase — reasoning/planning (optional, trigger-based)."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.agent.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class ThinkPhase(BasePhase):
  """Execute thinking/reasoning before main model invocation.

  Wraps:
    - Agent._thinking_should_run()
    - Agent._execute_thinking()

  The thinking output is stored on state and consumed by ComposePhase
  to inject reasoning context into the system prompt. If removed or
  replaced via hooks, ComposePhase falls back to computing thinking
  inline (backward compat).
  """

  _name = "think"
  _requires: set[str] = {"tools"}
  _provides: set[str] = {"thinking_output", "reasoning_steps", "reasoning_messages"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  def should_run(self, state: LoopState) -> bool:
    """Only run if thinking is configured on the agent."""
    return self._agent._thinking is not None

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    # Check trigger conditions (always/auto/never)
    if not await self._agent._thinking_should_run(state.all_messages):
      yield state, None
      return

    # Execute thinking
    thinking_output, reasoning_steps, reasoning_messages = await self._agent._execute_thinking(
      state.context,
      state.all_messages,
      state.tools,
    )

    state.thinking_output = thinking_output
    state.reasoning_steps = reasoning_steps
    state.reasoning_messages = reasoning_messages

    yield state, None
