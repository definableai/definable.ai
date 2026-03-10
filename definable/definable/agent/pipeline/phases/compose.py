"""ComposePhase — build system prompt and invoke messages."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.agent.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class ComposePhase(BasePhase):
  """Build invoke messages from system prompt, layers, thinking.

  Wraps Agent._build_invoke_messages() which:
    - Assembles system content (instructions + skills + layer guide)
    - Injects pre-computed thinking output (from ThinkPhase) or
      computes inline (backward compat when ThinkPhase is removed)
    - Appends knowledge/research/memory context
    - Injects readers context into user messages
  """

  _name = "compose"
  _requires: set[str] = {"tools"}
  _provides: set[str] = {"invoke_messages"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None

    invoke_messages, reasoning_steps, reasoning_messages = await self._agent._build_invoke_messages(
      state.context,
      state.all_messages,
      state.tools,
      thinking_output=state.thinking_output,
      thinking_text=state.thinking_text,
      reasoning_steps=state.reasoning_steps,
      reasoning_messages=state.reasoning_messages,
    )

    state.invoke_messages = invoke_messages
    # Update reasoning state — may have been computed inline if ThinkPhase was skipped
    state.reasoning_steps = reasoning_steps
    state.reasoning_messages = reasoning_messages

    yield state, None
