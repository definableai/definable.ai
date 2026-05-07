"""GuardInputPhase and GuardOutputPhase — input/output guardrails."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState, LoopStatus
from definable.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class GuardInputPhase(BasePhase):
  """Run input guardrails (after recall, before execution).

  Wraps Agent._run_input_guardrails().

  If blocked, sets state.status = LoopStatus.blocked and
  state.content to the block reason. The pipeline will exit early.
  """

  _name = "guard_input"

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  def should_run(self, state: LoopState) -> bool:
    """Only run if agent has input guardrails."""
    return bool(self._agent.guardrails and self._agent.guardrails.input)

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    input_block = await self._agent._run_input_guardrails(state.context, state.new_messages)
    if input_block is not None:
      state.status = LoopStatus.blocked
      state.content = input_block.content
      yield state, None
      return

    yield state, None


class GuardOutputPhase(BasePhase):
  """Run output guardrails (after execution, before memory store).

  Wraps Agent._run_output_guardrails().

  If blocked, replaces state.content with the block reason.
  """

  _name = "guard_output"
  _requires: set[str] = {"content"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  def should_run(self, state: LoopState) -> bool:
    """Only run if agent has output guardrails."""
    return bool(self._agent.guardrails and self._agent.guardrails.output)

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    if not state.content:
      yield state, None
      return

    from definable.agent.events import RunOutput

    temp_output = RunOutput(
      run_id=state.run_id,
      session_id=state.session_id,
      content=state.content,
    )
    output_block = await self._agent._run_output_guardrails(state.context, temp_output)
    if output_block is not None:
      state.content = output_block.content

    yield state, None
