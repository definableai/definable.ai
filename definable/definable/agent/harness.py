"""Agent harness — flat execution flow replacing the Pipeline + 8 phases.

A single async generator that runs the complete agent execution as sequential
steps. Each step is a plain function call to layers.py or prompt.py.

Usage from Agent::

    async for state, event in execute_run(self, state):
        ...
"""

from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from definable.agent.events import (
  ReasoningCompletedEvent,
  ReasoningContentDeltaEvent,
  ReasoningStartedEvent,
  RunCompletedEvent,
  RunOutput,
  RunStartedEvent,
)
from definable.agent.loop import AgentLoop
from definable.agent.pipeline.state import LoopState, LoopStatus
from definable.run.base import BaseRunOutputEvent
from definable.model.message import Message

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.cancellation import CancellationToken


async def execute_run(
  agent: "Agent",
  state: LoopState,
  *,
  cancellation_token: "Optional[CancellationToken]" = None,
) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
  """Execute the complete agent run — flat, readable, sequential.

  Replaces Pipeline.execute() and all 8 phase classes. Each step is a
  plain function call. Override by subclassing Agent. Customize with
  before_request/after_response hooks.

  Yields (state, event) tuples compatible with the old pipeline interface.
  """
  state.status = LoopStatus.running

  # ── Step 1: Prepare tools ──────────────────────────────────
  from definable.agent.lifecycle import ensure_toolkits_initialized

  await ensure_toolkits_initialized(agent)
  assert state.context is not None
  state.tools = agent._prepare_tools_for_run(state.context)

  # ── Step 2: Recall context (memory + knowledge + research + readers) ──
  if agent._knowledge or agent.memory or getattr(agent, "_researcher", None) or agent.readers:
    from definable.agent.layers import run_pre_execution_pipeline

    events = await run_pre_execution_pipeline(agent, state.context, state.new_messages, state.all_messages)

    # Propagate context results to state
    state.knowledge_context = state.context.knowledge_context
    state.memory_context = state.context.memory_context
    state.research_context = state.context.research_context
    state.readers_context = state.context.readers_context

    for evt in events:
      yield state, evt

  # ── Step 3: Think (native or fallback) ─────────────────────
  if agent._thinking is not None:
    async for s, e in _run_thinking(agent, state):
      yield s, e

  # ── Step 4: Guard input ────────────────────────────────────
  if agent.guardrails and agent.guardrails.input:
    from definable.agent.layers import run_input_guardrails

    input_block = await run_input_guardrails(agent, state.context, state.new_messages)
    if input_block is not None:
      state.status = LoopStatus.blocked
      state.content = input_block.content
      yield state, None
      return

  # ── Step 5: Build invoke messages ──────────────────────────
  from definable.agent.prompt import build_invoke_messages

  state.invoke_messages, state.reasoning_steps, state.reasoning_messages = await build_invoke_messages(
    agent,
    state.context,
    state.all_messages,
    state.tools,
    thinking_output=state.thinking_output,
    thinking_text=state.thinking_text,
    reasoning_steps=state.reasoning_steps,
    reasoning_messages=state.reasoning_messages,
  )

  # ── Step 6: Execute agent loop (model → tools → repeat) ───
  async for s, e in _run_agent_loop(agent, state):
    yield s, e

  # Early exit on terminal states (paused, cancelled, error)
  if state.status in (LoopStatus.paused, LoopStatus.cancelled, LoopStatus.error):
    return

  # ── Step 7: Guard output ───────────────────────────────────
  if agent.guardrails and agent.guardrails.output and state.content:
    from definable.agent.layers import run_output_guardrails

    temp_output = RunOutput(
      run_id=state.run_id,
      session_id=state.session_id,
      content=state.content,
    )
    output_block = await run_output_guardrails(agent, state.context, temp_output)
    if output_block is not None:
      state.content = output_block.content

  # ── Step 8: Store memory ───────────────────────────────────
  from definable.agent.layers import should_store_memory

  if should_store_memory(agent):
    from definable.agent.layers import memory_store
    from definable.agent.lifecycle import drain_memory_tasks

    store_messages = list(state.new_messages)
    if state.content:
      store_messages.append(Message(role="assistant", content=state.content))

    memory_events = memory_store(agent, store_messages, state.context)
    for evt in memory_events:
      yield state, evt

    await drain_memory_tasks(agent)

  # ── Done ───────────────────────────────────────────────────
  if state.status == LoopStatus.running:
    state.status = LoopStatus.completed


# ---------------------------------------------------------------------------
# Step 3: Thinking (native or Definable fallback)
# ---------------------------------------------------------------------------


async def _run_thinking(
  agent: "Agent",
  state: LoopState,
) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
  """Execute thinking layer: check trigger, then native or fallback path."""
  from definable.agent.layers import thinking_should_run

  if not await thinking_should_run(agent, state.all_messages):
    return

  thinking = agent._thinking
  assert thinking is not None

  if thinking.should_use_native(agent.model):
    # Native thinking: configure the model, let the loop handle events.
    from definable.agent.layers import enable_native_thinking

    enable_native_thinking(agent)
  else:
    # Definable fallback: separate LLM call, stream reasoning tokens
    async for s, e in _execute_thinking_fallback(agent, state):
      yield s, e


async def _execute_thinking_fallback(
  agent: "Agent",
  state: LoopState,
) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
  """Execute Definable's fallback thinking via the unified async generator."""
  ctx = state.context
  assert ctx is not None

  yield (
    state,
    ReasoningStartedEvent(
      run_id=ctx.run_id,
      session_id=ctx.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
    ),
  )

  from definable.agent.layers import execute_thinking

  async for item in execute_thinking(agent, ctx, state.all_messages, state.tools):
    if isinstance(item, str):
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
      thinking_output, thinking_text, reasoning_steps, reasoning_messages = item
      state.thinking_output = thinking_output
      state.thinking_text = thinking_text
      state.reasoning_steps = reasoning_steps
      state.reasoning_messages = reasoning_messages

  yield (
    state,
    ReasoningCompletedEvent(
      run_id=ctx.run_id,
      session_id=ctx.session_id,
      agent_id=agent.agent_id,
      agent_name=agent.agent_name,
    ),
  )


# ---------------------------------------------------------------------------
# Step 6: Agent Loop (model → tools → repeat)
# ---------------------------------------------------------------------------


async def _run_agent_loop(
  agent: "Agent",
  state: LoopState,
) -> AsyncGenerator[Tuple[LoopState, Optional[BaseRunOutputEvent]], None]:
  """Run the agent loop: emit RunStarted, execute loop, capture results."""
  assert state.context is not None
  assert state.model is not None

  # Emit RunStarted
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

  # Detect native thinking
  _native_thinking = bool(agent._thinking and agent._thinking.enabled and agent._thinking.should_use_native(agent.model))

  # Deferred tools: swap full tool set with just load_tools + loaded tools
  _loop_tools = state.tools
  _deferred = agent._deferred_tool_manager
  if _deferred is not None:
    _loop_tools = _deferred.get_active_tools()

  # Create the loop
  loop = AgentLoop(
    model=state.model,
    tools=_loop_tools,
    messages=state.invoke_messages,
    context=state.context,
    config=state.config or agent.config,
    streaming=state.streaming,
    native_thinking=_native_thinking,
    cancellation_token=state.cancellation_token,
    compression_manager=agent._compression_manager,
    guardrails=agent.guardrails,
    emit_fn=agent._emit,
    agent_id=state.agent_id,
    agent_name=state.agent_name,
    deferred_tool_manager=_deferred,
    permission_service=getattr(agent, "_permission_service", None),
  )

  # Run the loop — streaming is controlled by state.streaming
  async for event in loop.run():
    if isinstance(event, RunCompletedEvent):
      state.content = event.content
      state.parsed = event.parsed
      state.metrics = event.metrics
    yield state, event

  # Capture output messages and tool executions
  state.output_messages = [m for m in loop.messages if m.role != "system"]
  state.tool_executions = loop.tool_executions or []

  # Capture native reasoning content
  if loop.native_reasoning_content:
    state.native_reasoning_content = loop.native_reasoning_content
