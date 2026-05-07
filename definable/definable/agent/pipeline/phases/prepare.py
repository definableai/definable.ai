"""PreparePhase — normalize input, build RunContext, prepare tools."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class PreparePhase(BasePhase):
  """Normalize instruction into messages, build RunContext, prepare tools.

  Wraps:
    - Agent._normalize_instruction()
    - Agent._prepare_tools_for_run()
  """

  _name = "prepare"
  _provides: set[str] = {"tools"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    # Ensure async toolkits (e.g. MCPToolkit) are initialized — idempotent with lock
    await self._agent._ensure_toolkits_initialized()

    # Prepare tools with injected context
    assert state.context is not None
    state.tools = self._agent._prepare_tools_for_run(state.context)

    yield state, None
