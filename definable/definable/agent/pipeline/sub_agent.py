"""Sub-agent spawning — Self-Manager pattern with Thread Control Blocks."""

import asyncio
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Callable, List, Literal, Optional
from uuid import uuid4

from definable.utils.log import log_debug, log_error, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.run.agent import RunOutput
  from definable.knowledge import Knowledge
  from definable.model.base import Model
  from definable.model.metrics import Metrics
  from definable.tool.function import Function


@dataclass
class SubAgentConfig:
  """Configuration for a spawned sub-agent."""

  instructions: str
  tools: Optional[List["Function"]] = None  # Tool subset (None = inherit parent's)
  model: "str | Model | None" = None  # Override (None = inherit parent's)
  max_tool_rounds: int = 15  # Tighter limits for sub-agents
  knowledge: Optional["Knowledge"] = None  # Optional knowledge injection
  context: Optional[str] = None  # Prefix context from parent


@dataclass
class ThreadControlBlock:
  """Tracks a spawned sub-agent (inspired by Self-Manager paper).

  Each TCB represents one sub-agent execution. Multiple TCBs can
  exist in parallel when the parent agent spawns multiple sub-agents
  in a single model turn.
  """

  id: str
  goal: str
  state: Literal["running", "completed", "failed", "killed"] = "running"
  agent_config: Optional[SubAgentConfig] = None
  start_time: float = field(default_factory=time)
  result: Optional[str] = None
  error: Optional[str] = None
  metrics: Optional["Metrics"] = None
  run_output: Optional["RunOutput"] = None  # RunOutput for inspection


@dataclass
class SubAgentPolicy:
  """Developer controls for sub-agent spawning.

  Example::

      agent = Agent(
          model="openai/gpt-4o",
          sub_agents=SubAgentPolicy(
              max_concurrent=5,
              inherit_tools=True,
              allowed_models=["openai/gpt-4o-mini"],
          ),
      )
  """

  max_concurrent: int = 5
  max_tool_rounds: int = 15
  inherit_tools: bool = True
  inherit_knowledge: bool = False
  allowed_models: Optional[List[str]] = None
  on_spawn: Optional[Callable] = None
  on_complete: Optional[Callable] = None


def _build_spawn_agent_function(
  parent_agent: "Agent",
  policy: SubAgentPolicy,
) -> "Function":
  """Build a spawn_agent Function that creates child Agents at runtime.

  Each call returns a fresh Function with its own semaphore and TCB list.
  The entrypoint is an async callable that:
    1. Creates a ThreadControlBlock
    2. Emits SubAgentSpawnedEvent
    3. Creates a child Agent (minimal config — no memory/middleware/tracing)
    4. Runs child.arun(goal)
    5. Emits SubAgentCompletedEvent or SubAgentFailedEvent
    6. Returns the child's content as a string (tool result for parent)
  """
  from definable.tool.function import Function

  # Shared state across all spawn_agent calls within one parent run
  semaphore = asyncio.Semaphore(policy.max_concurrent)
  tcb_list: List[ThreadControlBlock] = []

  async def spawn_agent(
    goal: str,
    instructions: str = "",
    context: str = "",
  ) -> str:
    """Spawn a sub-agent to handle a subtask autonomously.

    The sub-agent gets its own execution loop, runs to completion,
    and returns its final answer. Multiple spawn_agent calls in one
    turn execute in parallel.

    Args:
        goal: The task for the sub-agent to accomplish.
        instructions: Optional system instructions for the sub-agent.
        context: Optional context to prepend to the sub-agent's prompt.
    """
    thread_id = str(uuid4())
    tcb = ThreadControlBlock(id=thread_id, goal=goal)
    tcb_list.append(tcb)

    # Emit spawned event
    try:
      from definable.agent.run.agent import SubAgentSpawnedEvent

      await parent_agent._event_bus.emit(
        SubAgentSpawnedEvent(
          run_id=thread_id,
          thread_id=thread_id,
          goal=goal,
          instructions=instructions or None,
        )
      )
    except Exception:
      pass  # Event emission is best-effort

    # Fire on_spawn callback
    if policy.on_spawn is not None:
      try:
        result = policy.on_spawn(tcb)
        if asyncio.iscoroutine(result):
          await result
      except Exception as cb_err:
        log_warning(f"on_spawn callback error: {cb_err}")

    async with semaphore:
      try:
        # Lazy import to avoid circular deps
        from definable.agent.agent import Agent
        from definable.agent.config import AgentConfig

        # Resolve tools for child
        child_tools: Optional[List["Function"]] = None
        if policy.inherit_tools:
          # Copy parent's original tool definitions (not prepared copies)
          child_tools = list(parent_agent._tools_dict.values())

        # Resolve knowledge for child
        child_knowledge = parent_agent._knowledge if policy.inherit_knowledge else None

        # Build child instructions
        child_instructions = instructions or "You are a helpful sub-agent. Complete the given task thoroughly."
        if context:
          child_instructions = f"{context}\n\n{child_instructions}"

        # Create child config with tighter limits
        child_config = AgentConfig(max_tool_rounds=policy.max_tool_rounds)

        # Create child agent — minimal: no memory, no middleware, no tracing, no sub_agents
        child = Agent(
          model=parent_agent.model,
          tools=child_tools,
          knowledge=child_knowledge,
          instructions=child_instructions,
          config=child_config,
        )

        log_debug(f"Sub-agent {thread_id[:8]} spawned for goal: {goal[:80]}")

        # Run the child agent
        run_output = await child.arun(goal)

        # Success path
        tcb.state = "completed"
        tcb.result = run_output.content
        tcb.run_output = run_output
        if hasattr(run_output, "metrics"):
          tcb.metrics = run_output.metrics

        # Emit completed event
        try:
          from definable.agent.run.agent import SubAgentCompletedEvent

          await parent_agent._event_bus.emit(
            SubAgentCompletedEvent(
              run_id=thread_id,
              thread_id=thread_id,
              result=run_output.content,
              metrics=tcb.metrics,
            )
          )
        except Exception:
          pass

        # Fire on_complete callback
        if policy.on_complete is not None:
          try:
            result = policy.on_complete(tcb)
            if asyncio.iscoroutine(result):
              await result
          except Exception as cb_err:
            log_warning(f"on_complete callback error: {cb_err}")

        log_debug(f"Sub-agent {thread_id[:8]} completed: {(run_output.content or '')[:80]}")
        return run_output.content or ""

      except Exception as exc:
        # Failure path — never crash the parent
        tcb.state = "failed"
        tcb.error = str(exc)
        error_msg = f"Sub-agent failed: {exc}"

        log_error(f"Sub-agent {thread_id[:8]} failed: {exc}")

        # Emit failed event
        try:
          from definable.agent.run.agent import SubAgentFailedEvent

          await parent_agent._event_bus.emit(
            SubAgentFailedEvent(
              run_id=thread_id,
              thread_id=thread_id,
              error=str(exc),
            )
          )
        except Exception:
          pass

        # Fire on_complete callback even on failure
        if policy.on_complete is not None:
          try:
            result = policy.on_complete(tcb)
            if asyncio.iscoroutine(result):
              await result
          except Exception as cb_err:
            log_warning(f"on_complete callback error: {cb_err}")

        return error_msg

  # Build the Function object with pre-defined schema
  fn = Function(
    name="spawn_agent",
    description=(
      "Spawn a sub-agent to handle a subtask autonomously. "
      "The sub-agent runs to completion and returns its answer. "
      "Multiple spawn_agent calls in one turn execute in parallel. "
      "Use this to delegate independent subtasks."
    ),
    parameters={
      "type": "object",
      "properties": {
        "goal": {
          "type": "string",
          "description": "The task for the sub-agent to accomplish.",
        },
        "instructions": {
          "type": "string",
          "description": "Optional system instructions for the sub-agent.",
        },
        "context": {
          "type": "string",
          "description": "Optional context to prepend to the sub-agent's prompt.",
        },
      },
      "required": ["goal"],
    },
    entrypoint=spawn_agent,
    skip_entrypoint_processing=True,
    add_instructions=False,
  )

  # Attach tcb_list for external inspection (e.g., tests)
  fn._tcb_list = tcb_list  # type: ignore[attr-defined]

  return fn
