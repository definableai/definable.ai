"""InvokeLoopPhase — THE LOOP: model -> tools -> repeat."""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.events import RunCompletedEvent, RunPausedEvent, RunStartedEvent
from definable.agent.loop import AgentLoop
from definable.agent.pipeline.phase import BasePhase
from definable.agent.pipeline.state import LoopState, LoopStatus
from definable.agent.run.base import BaseRunOutputEvent

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class InvokeLoopPhase(BasePhase):
  """Run the agent loop (model -> tools -> repeat).

  Creates an AgentLoop and consumes its event generator. This wraps
  the existing proven while-loop-with-tools pattern.

  The phase:
    1. Emits RunStartedEvent
    2. Runs AgentLoop (streaming or non-streaming based on state.streaming)
    3. Collects final content, metrics, tool executions
    4. Handles HITL pause
    5. Yields all events from the loop

  Events are yielded to the pipeline, which dispatches them to the
  EventStream (trace_writer + event_bus). The phase does NOT call
  ``_emit()`` or ``_event_bus.emit()`` directly to avoid double-emission.
  """

  _name = "invoke_loop"
  _requires: set[str] = {"invoke_messages", "tools"}
  _provides: set[str] = {"content", "metrics", "output_messages", "tool_executions"}

  def __init__(self, agent: "Agent") -> None:
    self._agent = agent

  async def execute(self, state: LoopState) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
    assert state.context is not None
    assert state.model is not None

    # Emit RunStarted — yielded to pipeline EventStream
    started_event = RunStartedEvent(
      run_id=state.context.run_id,
      session_id=state.context.session_id,
      agent_id=state.agent_id,
      agent_name=state.agent_name,
      model=state.model.id,
      model_provider=state.model.provider,  # type: ignore[arg-type]
      run_input=state.run_input,
    )
    yield state, started_event

    # Create the loop — streaming and cancellation come from state
    loop = AgentLoop(
      model=state.model,
      tools=state.tools,
      messages=state.invoke_messages,
      context=state.context,
      config=state.config or self._agent.config,
      streaming=state.streaming,
      cancellation_token=state.cancellation_token,
      compression_manager=self._agent._compression_manager,
      guardrails=self._agent.guardrails,
      emit_fn=self._agent._emit,
      agent_id=state.agent_id,
      agent_name=state.agent_name,
    )

    # Use unified run() — streaming is controlled by state.streaming
    # Events are yielded to the pipeline for EventStream dispatch.
    # The loop's emit_fn still handles internal trace writes for events
    # that the loop itself emits (tool calls, etc.) — but the pipeline
    # EventStream is the primary dispatch for all phase-yielded events.
    async for event in loop.run():
      if isinstance(event, RunCompletedEvent):
        state.content = event.content
        state.metrics = event.metrics

      elif isinstance(event, RunPausedEvent):
        state.status = LoopStatus.paused
        state.requirements = event.requirements

      yield state, event

    # Capture output messages (excluding system)
    state.output_messages = [m for m in loop.messages if m.role != "system"]
    state.tool_executions = loop.tool_executions or []
