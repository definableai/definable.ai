"""ThinkPhase — reasoning/planning (optional, trigger-based)."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState
from definable.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class ThinkPhase(BasePhase):
  """Execute thinking/reasoning before main model invocation.

  Wraps:
    - Agent._thinking_should_run()
    - Agent._execute_thinking() (unified async generator — Definable fallback layer)
    - Agent._enable_native_thinking() (native model thinking)

  For native thinking models (Claude, DeepSeek, Gemini), this phase
  configures the model's thinking parameters. The actual reasoning
  content is emitted by the AgentLoop during model invocation.

  For non-native models, this phase runs Definable's fallback thinking
  layer (a separate LLM call) and stores the output on state for
  ComposePhase to inject into the system prompt. Reasoning tokens are
  yielded as ReasoningContentDelta events token-by-token.
  """

  _name = "think"
  _requires: set[str] = {"tools"}
  _provides: set[str] = {"thinking_output", "thinking_text", "reasoning_steps", "reasoning_messages"}

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

    thinking = self._agent._thinking
    assert thinking is not None

    if thinking.should_use_native(self._agent.model):
      # Native thinking: configure the model, let the loop handle events.
      self._agent._enable_native_thinking()
      yield state, None
    else:
      # Fallback: unified generator handles both streaming and non-streaming
      async for s, e in self._execute_fallback(state):
        yield s, e

  async def _execute_fallback(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    """Execute Definable's fallback thinking via the unified async generator."""
    from definable.agent.events import (
      ReasoningCompletedEvent,
      ReasoningContentDeltaEvent,
      ReasoningStartedEvent,
    )

    ctx = state.context
    assert ctx is not None
    agent = self._agent

    # Yield ReasoningStarted
    yield (
      state,
      ReasoningStartedEvent(
        run_id=ctx.run_id,
        session_id=ctx.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
      ),
    )

    # Stream thinking tokens from the unified generator
    async for item in agent._execute_thinking(ctx, state.all_messages, state.tools):
      if isinstance(item, str):
        # Content delta — yield as ReasoningContentDelta
        yield (
          state,
          ReasoningContentDeltaEvent(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
            agent_id=agent.agent_id,
            agent_name=agent.agent_name,
            reasoning_content=item,
          ),
        )
      elif isinstance(item, tuple):
        # Final result tuple: (thinking_output, thinking_text, reasoning_steps, reasoning_messages)
        thinking_output, thinking_text, reasoning_steps, reasoning_messages = item
        state.thinking_output = thinking_output
        state.thinking_text = thinking_text
        state.reasoning_steps = reasoning_steps
        state.reasoning_messages = reasoning_messages

    # Yield ReasoningCompleted
    yield (
      state,
      ReasoningCompletedEvent(
        run_id=ctx.run_id,
        session_id=ctx.session_id,
        agent_id=agent.agent_id,
        agent_name=agent.agent_name,
      ),
    )
