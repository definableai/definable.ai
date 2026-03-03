"""Team — multi-agent coordination for Definable.

A Team composes multiple Agent instances under a leader that coordinates
their execution using one of four modes:

- **coordinate**: Leader selects members, crafts tasks, synthesizes responses.
- **route**: Leader routes to a single specialist; returns their response directly.
- **collaborate**: All members receive the same task in parallel; leader synthesizes.
- **tasks**: Leader decomposes goal into a shared TaskList; autonomous loop until done.

Example::

    from definable.agent import Agent
    from definable.agent.team import Team, TeamMode

    researcher = Agent(model="openai/gpt-4o", instructions="You are a research specialist.")
    writer = Agent(model="openai/gpt-4o", instructions="You are a technical writer.")

    team = Team(
        name="content-team",
        model="openai/gpt-4o",
        members=[researcher, writer],
        mode=TeamMode.coordinate,
        instructions="Produce well-researched technical content.",
    )

    result = await team.arun("Write an article about quantum computing.")
    print(result.content)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import (
  TYPE_CHECKING,
  Any,
  Callable,
  Dict,
  List,
  Optional,
  Type,
  Union,
)
from uuid import uuid4

from pydantic import BaseModel

from definable.agent.event_bus import EventBus
from definable.agent.team.events import (
  MemberCompletedEvent,
  MemberDelegatedEvent,
  MemberErrorEvent,
  MemberRoutedEvent,
  TaskIterationEvent,
  TeamRunCompletedEvent,
  TeamRunErrorEvent,
  TeamRunStartedEvent,
)
from definable.agent.team.mode import TeamMode
from definable.agent.team.task import TaskList, TaskStatus
from definable.utils.log import log_debug, log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.run.agent import RunOutput
  from definable.model.base import Model
  from definable.tool.function import Function


@dataclass
class Team:
  """Multi-agent coordination container.

  A Team is a leader agent that orchestrates member agents using
  one of four coordination modes. The leader gets auto-injected tools
  for delegation and task management.

  The leader's model does the thinking; members do the doing.
  """

  # ── Identity ──────────────────────────────────────────────
  name: str = ""
  instructions: Optional[str] = None
  description: Optional[str] = None

  # ── Model (leader) ────────────────────────────────────────
  model: "Union[str, Model, None]" = None

  # ── Members ───────────────────────────────────────────────
  members: List["Union[Agent, Team]"] = field(default_factory=list)

  # ── Mode ──────────────────────────────────────────────────
  mode: TeamMode = TeamMode.coordinate

  # ── Task-mode settings ────────────────────────────────────
  max_iterations: int = 10

  # ── Context sharing ───────────────────────────────────────
  share_member_interactions: bool = False

  # ── Leader tools (additive) ───────────────────────────────
  tools: Optional[List["Function"]] = None

  # ── Structured output ─────────────────────────────────────
  output_schema: Optional[Type[BaseModel]] = None

  # ── Debug ─────────────────────────────────────────────────
  debug: bool = False

  # ── Internal ──────────────────────────────────────────────
  _id: str = field(default_factory=lambda: str(uuid4()))
  _event_bus: EventBus = field(default_factory=EventBus)
  _member_map: Dict[str, Any] = field(default_factory=dict, repr=False)
  _leader: Optional["Agent"] = field(default=None, repr=False)
  _interactions: List[str] = field(default_factory=list, repr=False)

  def __post_init__(self) -> None:
    if isinstance(self.mode, str):
      try:
        self.mode = TeamMode(self.mode)
      except ValueError:
        valid = [m.value for m in TeamMode]
        raise ValueError(f"Invalid team mode '{self.mode}'. Valid modes: {valid}")
    if not self.name:
      self.name = f"team-{self._id[:8]}"
    self._build_member_map()

  # ── Properties ────────────────────────────────────────────

  @property
  def team_id(self) -> str:
    return self._id

  @property
  def member_names(self) -> List[str]:
    return list(self._member_map.keys())

  @property
  def events(self) -> EventBus:
    return self._event_bus

  # ── Public API ────────────────────────────────────────────

  def run(
    self,
    instruction: str,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> "RunOutput":
    """Synchronous team run."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    if loop and loop.is_running():
      import concurrent.futures

      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
          asyncio.run,
          self.arun(
            instruction,
            session_id=session_id,
            user_id=user_id,
            output_schema=output_schema,
          ),
        )
        return future.result()
    else:
      return asyncio.run(
        self.arun(
          instruction,
          session_id=session_id,
          user_id=user_id,
          output_schema=output_schema,
        )
      )

  async def arun(
    self,
    instruction: str,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
  ) -> "RunOutput":
    """Async team run — the primary execution path."""
    schema = output_schema or self.output_schema
    run_id = str(uuid4())

    await self._event_bus.emit(
      TeamRunStartedEvent(
        run_id=run_id,
        team_id=self._id,
        team_name=self.name,
        mode=self.mode.value,
        member_names=self.member_names,
      )
    )

    try:
      if self.mode == TeamMode.coordinate:
        result = await self._run_coordinate(instruction, run_id=run_id, session_id=session_id, user_id=user_id, output_schema=schema)
      elif self.mode == TeamMode.route:
        result = await self._run_route(instruction, run_id=run_id, session_id=session_id, user_id=user_id, output_schema=schema)
      elif self.mode == TeamMode.collaborate:
        result = await self._run_collaborate(instruction, run_id=run_id, session_id=session_id, user_id=user_id, output_schema=schema)
      elif self.mode == TeamMode.tasks:
        result = await self._run_tasks(instruction, run_id=run_id, session_id=session_id, user_id=user_id, output_schema=schema)
      else:
        raise ValueError(f"Unknown team mode: {self.mode}")

      await self._event_bus.emit(
        TeamRunCompletedEvent(
          run_id=run_id,
          team_id=self._id,
          team_name=self.name,
          content=result.content,
        )
      )
      return result

    except Exception as exc:
      await self._event_bus.emit(
        TeamRunErrorEvent(
          run_id=run_id,
          team_id=self._id,
          team_name=self.name,
          error=str(exc),
        )
      )
      raise

  # ── Mode implementations ──────────────────────────────────

  async def _run_coordinate(
    self,
    instruction: str,
    *,
    run_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    output_schema: Optional[Type[BaseModel]],
  ) -> "RunOutput":
    """Coordinate mode — leader picks members, crafts tasks, synthesizes."""
    from definable.agent.team._prompts import build_coordinate_prompt
    from definable.agent.team._tools import build_delegate_tool, build_member_info_tool

    leader = self._get_or_create_leader()
    delegate_tool = build_delegate_tool(self._member_map, self._make_run_member_fn(run_id))
    info_tool = build_member_info_tool(self._member_map)

    leader_tools = [delegate_tool, info_tool]
    if self.tools:
      leader_tools.extend(self.tools)

    # Temporarily inject tools and instructions (also update _tools_dict for dispatch)
    original_tools = leader.tools
    original_tools_dict = leader._tools_dict.copy()
    original_instructions = leader.instructions
    try:
      leader.tools = leader_tools
      leader._tools_dict = {t.name: t for t in leader_tools}
      leader.instructions = build_coordinate_prompt(
        self.name,
        self._member_map,
        team_instructions=self.instructions,
        member_interactions=self._interactions if self.share_member_interactions else None,
      )

      result = await leader.arun(
        instruction,
        session_id=session_id,
        user_id=user_id,
        output_schema=output_schema,
      )
      return result
    finally:
      leader.tools = original_tools
      leader._tools_dict = original_tools_dict
      leader.instructions = original_instructions

  async def _run_route(
    self,
    instruction: str,
    *,
    run_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    output_schema: Optional[Type[BaseModel]],
  ) -> "RunOutput":
    """Route mode — leader routes to a single specialist, returns directly."""
    from definable.agent.team._prompts import build_route_prompt
    from definable.agent.team._tools import build_delegate_tool

    leader = self._get_or_create_leader()

    # Track which member responded so we can return their output
    routed_result: Dict[str, Any] = {}

    async def route_run_member(member_name: str, task_input: str) -> str:
      result = await self._run_single_member(member_name, task_input, run_id)
      routed_result["output"] = result
      routed_result["member"] = member_name

      await self._event_bus.emit(
        MemberRoutedEvent(
          run_id=run_id,
          member_name=member_name,
          reason=f"Routed by leader for: {task_input[:100]}",
        )
      )
      return result.content or ""

    delegate_tool = build_delegate_tool(self._member_map, route_run_member)

    original_tools = leader.tools
    original_tools_dict = leader._tools_dict.copy()
    original_instructions = leader.instructions
    route_tools = [delegate_tool] + (self.tools or [])
    try:
      leader.tools = route_tools
      leader._tools_dict = {t.name: t for t in route_tools}
      leader.instructions = build_route_prompt(
        self.name,
        self._member_map,
        team_instructions=self.instructions,
      )

      # Let leader pick the route
      leader_result = await leader.arun(
        instruction,
        session_id=session_id,
        user_id=user_id,
      )

      # If a member was invoked, return their RunOutput directly
      if "output" in routed_result:
        member_output = routed_result["output"]
        # Apply output_schema if needed on member's raw content
        if output_schema and member_output.content:
          return await leader.arun(
            f"Format this response according to the required schema:\n\n{member_output.content}",
            output_schema=output_schema,
          )
        return member_output

      # Fallback: leader answered directly (shouldn't happen in route mode)
      return leader_result
    finally:
      leader.tools = original_tools
      leader._tools_dict = original_tools_dict
      leader.instructions = original_instructions

  async def _run_collaborate(
    self,
    instruction: str,
    *,
    run_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    output_schema: Optional[Type[BaseModel]],
  ) -> "RunOutput":
    """Collaborate mode — broadcast to all members, leader synthesizes."""
    from definable.agent.team._prompts import build_collaborate_prompt

    # Run all members in parallel
    member_results: Dict[str, str] = {}
    errors: Dict[str, str] = {}

    async def run_one(name: str, agent: "Agent") -> None:
      await self._event_bus.emit(
        MemberDelegatedEvent(
          run_id=run_id,
          member_name=name,
          task_input=instruction,
          mode="collaborate",
        )
      )
      try:
        result = await agent.arun(instruction)
        member_results[name] = result.content or ""
        await self._event_bus.emit(
          MemberCompletedEvent(
            run_id=run_id,
            member_name=name,
            content=result.content,
          )
        )
      except Exception as exc:
        errors[name] = str(exc)
        await self._event_bus.emit(
          MemberErrorEvent(
            run_id=run_id,
            member_name=name,
            error=str(exc),
          )
        )

    tasks = [run_one(name, agent) for name, agent in self._member_map.items()]
    await asyncio.gather(*tasks)

    # Build synthesis prompt with all results
    synthesis_parts = []
    for name, content in member_results.items():
      synthesis_parts.append(f"## Response from {name}\n{content}")
    if errors:
      for name, error in errors.items():
        synthesis_parts.append(f"## Error from {name}\n{error}")

    all_responses = "\n\n".join(synthesis_parts)

    leader = self._get_or_create_leader()
    original_instructions = leader.instructions
    original_tools = leader.tools
    original_tools_dict = leader._tools_dict.copy()
    synth_tools = self.tools or []
    try:
      leader.instructions = build_collaborate_prompt(
        self.name,
        self._member_map,
        team_instructions=self.instructions,
      )
      leader.tools = synth_tools
      leader._tools_dict = {t.name: t for t in synth_tools}

      synthesis_prompt = (
        f"The user asked: {instruction}\n\n"
        f"All team members have responded. Synthesize their responses into the best possible answer:\n\n"
        f"{all_responses}"
      )

      result = await leader.arun(
        synthesis_prompt,
        session_id=session_id,
        user_id=user_id,
        output_schema=output_schema,
      )
      return result
    finally:
      leader.instructions = original_instructions
      leader.tools = original_tools
      leader._tools_dict = original_tools_dict

  async def _run_tasks(
    self,
    instruction: str,
    *,
    run_id: str,
    session_id: Optional[str],
    user_id: Optional[str],
    output_schema: Optional[Type[BaseModel]],
  ) -> "RunOutput":
    """Tasks mode — autonomous task decomposition and execution loop."""
    from definable.agent.team._prompts import build_tasks_prompt
    from definable.agent.team._tools import build_task_tools

    task_list = TaskList()
    leader = self._get_or_create_leader()

    run_member_fn = self._make_run_member_fn(run_id)

    async def emit_fn(event: Any) -> None:
      event.run_id = run_id
      await self._event_bus.emit(event)

    # Build task management tools
    task_tools = build_task_tools(task_list, self._member_map, run_member_fn, emit_fn)
    all_leader_tools = task_tools + (self.tools or [])

    original_tools = leader.tools
    original_tools_dict = leader._tools_dict.copy()
    original_instructions = leader.instructions
    messages = None  # Carry conversation forward across iterations

    try:
      for iteration in range(self.max_iterations):
        await self._event_bus.emit(
          TaskIterationEvent(
            run_id=run_id,
            iteration=iteration,
            pending_count=sum(1 for t in task_list.tasks if t.status == TaskStatus.pending),
            completed_count=sum(1 for t in task_list.tasks if t.status == TaskStatus.completed),
            failed_count=sum(1 for t in task_list.tasks if t.status == TaskStatus.failed),
          )
        )

        leader.tools = all_leader_tools
        leader._tools_dict = {t.name: t for t in all_leader_tools}
        leader.instructions = build_tasks_prompt(
          self.name,
          self._member_map,
          task_list,
          team_instructions=self.instructions,
          iteration=iteration,
          max_iterations=self.max_iterations,
        )

        # First iteration: use original instruction. Subsequent: continue conversation.
        if iteration == 0:
          prompt = instruction
        else:
          prompt = (
            "Continue working on the goal. Check task status, execute ready tasks, "
            "or create new tasks as needed. If the goal is achieved, call mark_goal_complete."
          )

        result = await leader.arun(
          prompt,
          messages=messages,
          session_id=session_id,
          user_id=user_id,
        )
        messages = result.messages

        # Check termination conditions
        if task_list.goal_complete:
          log_info(f"Team '{self.name}' goal complete at iteration {iteration}")
          break

        if task_list.all_terminal():
          log_info(f"Team '{self.name}' all tasks terminal at iteration {iteration}")
          # Ask leader to synthesize
          leader.instructions = original_instructions
          result = await leader.arun(
            f"All tasks are complete. Summarize the results:\n\n{task_list.get_summary_string()}",
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            output_schema=output_schema,
          )
          break

      else:
        # Max iterations reached
        log_warning(f"Team '{self.name}' hit max iterations ({self.max_iterations})")

      return result

    finally:
      leader.tools = original_tools
      leader._tools_dict = original_tools_dict
      leader.instructions = original_instructions

  # ── Internal helpers ──────────────────────────────────────

  def _build_member_map(self) -> None:
    """Build name → Agent mapping from members list."""
    self._member_map = {}
    for i, member in enumerate(self.members):
      if isinstance(member, Team):
        # Nested team — wrap in a thin agent proxy
        name = member.name or f"team-{i}"
        self._member_map[name] = _TeamAsAgent(member)
      else:
        # Agent — use agent name or generate one
        member_name = getattr(member, "name", None) or getattr(member, "agent_name", f"agent-{i}")
        name = str(member_name)
        self._member_map[name] = member

  def _get_or_create_leader(self) -> "Agent":
    """Get or lazily create the leader agent."""
    if self._leader is not None:
      return self._leader

    from definable.agent.agent import Agent

    model = self.model
    if model is None:
      # Use the first member's model as fallback
      for member in self._member_map.values():
        if hasattr(member, "model") and member.model is not None:
          model = member.model
          break

    if model is None:
      raise ValueError("Team requires a model — set it on the Team or on at least one member.")

    self._leader = Agent(
      name=f"{self.name}-leader",
      model=model,
      debug=self.debug,
    )
    return self._leader

  def _make_run_member_fn(self, run_id: str) -> Callable[..., Any]:
    """Create an async function that runs a named member and tracks interactions."""

    async def run_member(member_name: str, task_input: str) -> str:
      result = await self._run_single_member(member_name, task_input, run_id)
      return result.content or ""

    return run_member

  async def _run_single_member(
    self,
    member_name: str,
    task_input: str,
    run_id: str,
  ) -> "RunOutput":
    """Run a single member agent and emit events."""
    if member_name not in self._member_map:
      raise ValueError(f"Member '{member_name}' not found. Available: {self.member_names}")

    member = self._member_map[member_name]

    await self._event_bus.emit(
      MemberDelegatedEvent(
        run_id=run_id,
        member_name=member_name,
        task_input=task_input,
        mode=self.mode.value,
      )
    )

    log_debug(f"Team '{self.name}' delegating to '{member_name}': {task_input[:100]}")

    try:
      result = await member.arun(task_input)

      # Track interaction for context sharing
      interaction = f"[{member_name}] Task: {task_input[:200]}\nResponse: {(result.content or '')[:500]}"
      self._interactions.append(interaction)

      await self._event_bus.emit(
        MemberCompletedEvent(
          run_id=run_id,
          member_name=member_name,
          content=result.content,
          metrics=result.metrics.to_dict() if result.metrics else None,
        )
      )

      log_debug(f"Member '{member_name}' completed: {(result.content or '')[:100]}")
      return result

    except Exception as exc:
      await self._event_bus.emit(
        MemberErrorEvent(
          run_id=run_id,
          member_name=member_name,
          error=str(exc),
        )
      )
      log_error(f"Member '{member_name}' failed: {exc}")
      raise


class _TeamAsAgent:
  """Thin wrapper so a nested Team can be used as a member.

  Implements the minimal Agent interface needed for delegation:
  - ``name`` property
  - ``instructions`` attribute
  - ``tools`` attribute
  - ``model`` attribute
  - ``arun()`` method
  """

  def __init__(self, team: Team) -> None:
    self._team = team
    self.name = team.name
    self.instructions = team.description or team.instructions or f"Team: {team.name}"
    self.tools: List[Any] = []
    self.model = team.model

  @property
  def agent_name(self) -> str:
    return self._team.name

  async def arun(self, instruction: str, **kwargs: Any) -> "RunOutput":
    return await self._team.arun(instruction, **kwargs)
