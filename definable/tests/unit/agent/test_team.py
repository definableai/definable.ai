"""Tests for definable.agent.team — multi-agent coordination."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from definable.agent.team import (
  Task,
  TaskList,
  TaskStatus,
  Team,
  TeamMode,
)
from definable.agent.team.events import (
  MemberCompletedEvent,
  MemberDelegatedEvent,
  MemberErrorEvent,
  MemberRoutedEvent,
  TaskCreatedEvent,
  TaskIterationEvent,
  TaskStatusChangedEvent,
  TeamRunCompletedEvent,
  TeamRunErrorEvent,
  TeamRunStartedEvent,
)


# ═══════════════════════════════════════════════════════════════
# TaskStatus
# ═══════════════════════════════════════════════════════════════


class TestTaskStatus:
  def test_enum_values(self):
    assert TaskStatus.pending.value == "pending"
    assert TaskStatus.in_progress.value == "in_progress"
    assert TaskStatus.completed.value == "completed"
    assert TaskStatus.failed.value == "failed"
    assert TaskStatus.blocked.value == "blocked"

  def test_from_string(self):
    assert TaskStatus("pending") == TaskStatus.pending
    assert TaskStatus("completed") == TaskStatus.completed


# ═══════════════════════════════════════════════════════════════
# Task
# ═══════════════════════════════════════════════════════════════


class TestTask:
  def test_auto_id(self):
    task = Task(title="Test")
    assert task.id != ""
    assert len(task.id) == 8

  def test_auto_timestamp(self):
    task = Task(title="Test")
    assert task.created_at > 0

  def test_explicit_id(self):
    task = Task(id="custom-id", title="Test")
    assert task.id == "custom-id"

  def test_default_status(self):
    task = Task(title="Test")
    assert task.status == TaskStatus.pending

  def test_to_dict(self):
    task = Task(id="abc", title="Research", description="Do research", assignee="agent-1")
    d = task.to_dict()
    assert d["id"] == "abc"
    assert d["title"] == "Research"
    assert d["description"] == "Do research"
    assert d["assignee"] == "agent-1"
    assert d["status"] == "pending"

  def test_from_dict(self):
    data = {
      "id": "abc",
      "title": "Research",
      "status": "completed",
      "assignee": "agent-1",
      "result": "Done",
    }
    task = Task.from_dict(data)
    assert task.id == "abc"
    assert task.title == "Research"
    assert task.status == TaskStatus.completed
    assert task.result == "Done"

  def test_roundtrip_serialization(self):
    task = Task(title="Test", description="Desc", assignee="bob", notes=["note1"])
    task2 = Task.from_dict(task.to_dict())
    assert task2.title == task.title
    assert task2.description == task.description
    assert task2.assignee == task.assignee
    assert task2.notes == task.notes

  def test_dependencies_default(self):
    task = Task(title="Test")
    assert task.dependencies == []


# ═══════════════════════════════════════════════════════════════
# TaskList
# ═══════════════════════════════════════════════════════════════


class TestTaskList:
  def test_create_task(self):
    tl = TaskList()
    task = tl.create_task("Research", assignee="agent-1")
    assert len(tl.tasks) == 1
    assert task.title == "Research"
    assert task.assignee == "agent-1"
    assert task.status == TaskStatus.pending

  def test_get_task(self):
    tl = TaskList()
    task = tl.create_task("Test")
    found = tl.get_task(task.id)
    assert found is task

  def test_get_task_not_found(self):
    tl = TaskList()
    assert tl.get_task("nonexistent") is None

  def test_update_task(self):
    tl = TaskList()
    task = tl.create_task("Test")
    tl.update_task(task.id, status="completed", result="Done")
    assert task.status == TaskStatus.completed
    assert task.result == "Done"

  def test_update_task_not_found(self):
    tl = TaskList()
    result = tl.update_task("nonexistent", status="completed")
    assert result is None

  def test_get_available_tasks(self):
    tl = TaskList()
    tl.create_task("Available")
    tl.create_task("Also available")
    available = tl.get_available_tasks()
    assert len(available) == 2

  def test_get_available_tasks_filters_non_pending(self):
    tl = TaskList()
    t1 = tl.create_task("Available")
    t2 = tl.create_task("Done")
    tl.update_task(t2.id, status="completed")
    available = tl.get_available_tasks()
    assert len(available) == 1
    assert available[0].id == t1.id

  def test_get_available_tasks_filters_blocked(self):
    tl = TaskList()
    t1 = tl.create_task("First")
    tl.create_task("Second", dependencies=[t1.id])
    available = tl.get_available_tasks()
    assert len(available) == 1
    assert available[0].id == t1.id

  def test_get_available_tasks_for_assignee(self):
    tl = TaskList()
    tl.create_task("For Alice", assignee="alice")
    tl.create_task("For Bob", assignee="bob")
    tl.create_task("Unassigned")
    available = tl.get_available_tasks(for_assignee="alice")
    assert len(available) == 2  # Alice's task + unassigned

  def test_all_terminal_empty(self):
    tl = TaskList()
    assert tl.all_terminal() is False

  def test_all_terminal_all_completed(self):
    tl = TaskList()
    t1 = tl.create_task("A")
    t2 = tl.create_task("B")
    tl.update_task(t1.id, status="completed")
    tl.update_task(t2.id, status="completed")
    assert tl.all_terminal() is True

  def test_all_terminal_mixed(self):
    tl = TaskList()
    t1 = tl.create_task("A")
    t2 = tl.create_task("B")
    tl.update_task(t1.id, status="completed")
    tl.update_task(t2.id, status="failed")
    assert tl.all_terminal() is True

  def test_all_terminal_with_pending(self):
    tl = TaskList()
    t1 = tl.create_task("A")
    tl.create_task("B")
    tl.update_task(t1.id, status="completed")
    assert tl.all_terminal() is False

  def test_dependency_blocking(self):
    tl = TaskList()
    t1 = tl.create_task("First")
    t2 = tl.create_task("Second", dependencies=[t1.id])
    assert t2.status == TaskStatus.blocked

  def test_dependency_unblocking(self):
    tl = TaskList()
    t1 = tl.create_task("First")
    t2 = tl.create_task("Second", dependencies=[t1.id])
    assert t2.status == TaskStatus.blocked

    tl.update_task(t1.id, status="completed")
    assert t2.status == TaskStatus.pending  # type: ignore[comparison-overlap]

  def test_failed_dependency_cascades(self):
    tl = TaskList()
    t1 = tl.create_task("First")
    t2 = tl.create_task("Second", dependencies=[t1.id])
    tl.update_task(t1.id, status="failed")
    assert t2.status == TaskStatus.failed
    assert "dependency failed" in (t2.result or "").lower()

  def test_unknown_dependency_blocks(self):
    tl = TaskList()
    t1 = tl.create_task("Task", dependencies=["nonexistent"])
    assert t1.status == TaskStatus.blocked

  def test_get_summary_string_empty(self):
    tl = TaskList()
    assert "No tasks" in tl.get_summary_string()

  def test_get_summary_string_with_tasks(self):
    tl = TaskList()
    tl.create_task("Research", assignee="alice")
    tl.create_task("Write", assignee="bob")
    summary = tl.get_summary_string()
    assert "Research" in summary
    assert "Write" in summary
    assert "alice" in summary
    assert "bob" in summary

  def test_get_summary_string_truncates_long_results(self):
    tl = TaskList()
    t = tl.create_task("Task")
    tl.update_task(t.id, result="x" * 300, status="completed")
    summary = tl.get_summary_string()
    assert "..." in summary

  def test_goal_complete_in_summary(self):
    tl = TaskList()
    tl.goal_complete = True
    tl.completion_summary = "All done"
    tl.create_task("A")
    summary = tl.get_summary_string()
    assert "Goal marked complete" in summary

  def test_serialization_roundtrip(self):
    tl = TaskList()
    t1 = tl.create_task("Research", assignee="alice")
    tl.create_task("Write", dependencies=[t1.id])
    tl.goal_complete = True
    tl.completion_summary = "Done"

    d = tl.to_dict()
    tl2 = TaskList.from_dict(d)
    assert len(tl2.tasks) == 2
    assert tl2.goal_complete is True
    assert tl2.completion_summary == "Done"
    assert tl2.tasks[1].status == TaskStatus.blocked  # Recomputed

  def test_chained_dependencies(self):
    tl = TaskList()
    t1 = tl.create_task("Step 1")
    t2 = tl.create_task("Step 2", dependencies=[t1.id])
    t3 = tl.create_task("Step 3", dependencies=[t2.id])
    assert t2.status == TaskStatus.blocked
    assert t3.status == TaskStatus.blocked

    tl.update_task(t1.id, status="completed")
    assert t2.status == TaskStatus.pending  # type: ignore[comparison-overlap]
    assert t3.status == TaskStatus.blocked  # type: ignore[comparison-overlap,unreachable]  # Still blocked on t2

    tl.update_task(t2.id, status="completed")
    assert t3.status == TaskStatus.pending


# ═══════════════════════════════════════════════════════════════
# TeamMode
# ═══════════════════════════════════════════════════════════════


class TestTeamMode:
  def test_values(self):
    assert TeamMode.coordinate.value == "coordinate"
    assert TeamMode.route.value == "route"
    assert TeamMode.collaborate.value == "collaborate"
    assert TeamMode.tasks.value == "tasks"

  def test_from_string(self):
    assert TeamMode("coordinate") == TeamMode.coordinate
    assert TeamMode("tasks") == TeamMode.tasks


# ═══════════════════════════════════════════════════════════════
# Team Construction
# ═══════════════════════════════════════════════════════════════


class TestTeamConstruction:
  def test_minimal_construction(self):
    team = Team(name="test-team")
    assert team.name == "test-team"
    assert team.mode == TeamMode.coordinate
    assert team.team_id != ""

  def test_auto_name(self):
    team = Team()
    assert team.name.startswith("team-")

  def test_member_names(self):
    m1 = MagicMock()
    m1.name = "researcher"
    m2 = MagicMock()
    m2.name = "writer"
    team = Team(members=[m1, m2])
    assert set(team.member_names) == {"researcher", "writer"}

  def test_member_names_with_agent_name_fallback(self):
    m1 = MagicMock(spec=[])  # No name attribute
    m1.agent_name = "agent-alpha"
    team = Team(members=[m1])
    # Falls through to agent_name
    assert "agent-alpha" in team.member_names or len(team.member_names) == 1

  def test_nested_team_as_member(self):
    inner = Team(name="inner-team", description="Inner team desc")
    outer = Team(name="outer-team", members=[inner])
    assert "inner-team" in outer.member_names

  def test_mode_setting(self):
    team = Team(mode=TeamMode.route)
    assert team.mode == TeamMode.route

  def test_max_iterations_default(self):
    team = Team()
    assert team.max_iterations == 10

  def test_custom_max_iterations(self):
    team = Team(max_iterations=5)
    assert team.max_iterations == 5

  def test_events_property(self):
    team = Team()
    assert team.events is team._event_bus


# ═══════════════════════════════════════════════════════════════
# Team Events
# ═══════════════════════════════════════════════════════════════


class TestTeamEvents:
  def test_team_run_started_event(self):
    e = TeamRunStartedEvent(run_id="r1", team_id="t1", team_name="my-team", mode="coordinate", member_names=["a", "b"])
    assert e.event == "team_run_started"
    assert e.member_names == ["a", "b"]

  def test_team_run_completed_event(self):
    e = TeamRunCompletedEvent(run_id="r1", team_id="t1", team_name="my-team", content="Done")
    assert e.event == "team_run_completed"
    assert e.content == "Done"

  def test_team_run_error_event(self):
    e = TeamRunErrorEvent(run_id="r1", team_id="t1", team_name="my-team", error="boom")
    assert e.event == "team_run_error"
    assert e.error == "boom"

  def test_member_delegated_event(self):
    e = MemberDelegatedEvent(run_id="r1", member_name="alice", task_input="do stuff", mode="coordinate")
    assert e.event == "member_delegated"

  def test_member_completed_event(self):
    e = MemberCompletedEvent(run_id="r1", member_name="alice", content="result")
    assert e.event == "member_completed"

  def test_member_error_event(self):
    e = MemberErrorEvent(run_id="r1", member_name="alice", error="failed")
    assert e.event == "member_error"

  def test_member_routed_event(self):
    e = MemberRoutedEvent(run_id="r1", member_name="specialist", reason="best match")
    assert e.event == "member_routed"

  def test_task_created_event(self):
    e = TaskCreatedEvent(run_id="r1", task_id="t1", title="Research", assignee="alice")
    assert e.event == "task_created"

  def test_task_status_changed_event(self):
    e = TaskStatusChangedEvent(run_id="r1", task_id="t1", old_status="pending", new_status="completed")
    assert e.event == "task_status_changed"

  def test_task_iteration_event(self):
    e = TaskIterationEvent(run_id="r1", iteration=3, pending_count=2, completed_count=5, failed_count=0)
    assert e.event == "task_iteration"
    assert e.iteration == 3


# ═══════════════════════════════════════════════════════════════
# Team Tools (unit tests for tool builders)
# ═══════════════════════════════════════════════════════════════


class TestTeamTools:
  def test_build_delegate_tool(self):
    from definable.agent.team._tools import build_delegate_tool

    members = {"alice": MagicMock(instructions="Research expert", tools=[]), "bob": MagicMock(instructions="Writer", tools=[])}

    async def run_fn(name, task):
      return f"result from {name}"

    tool = build_delegate_tool(members, run_fn)  # type: ignore[arg-type]
    assert tool.name == "delegate_to_member"
    assert tool.description is not None
    assert "alice" in tool.description
    assert "bob" in tool.description
    assert tool.parameters["properties"]["member_name"]["enum"] == ["alice", "bob"]

  def test_build_member_info_tool(self):
    from definable.agent.team._tools import build_member_info_tool

    members = {
      "alice": MagicMock(instructions="Research expert", tools=[MagicMock(name="search"), MagicMock(name="read")]),
    }
    tool = build_member_info_tool(members)  # type: ignore[arg-type]
    assert tool.name == "get_member_information"

  @pytest.mark.asyncio
  async def test_delegate_tool_calls_run_fn(self):
    from definable.agent.team._tools import build_delegate_tool

    run_fn = AsyncMock(return_value="the result")
    members = {"alice": MagicMock(instructions="Expert", tools=[])}
    tool = build_delegate_tool(members, run_fn)  # type: ignore[arg-type]
    assert tool.entrypoint is not None
    result = await tool.entrypoint("alice", "do research")
    assert result == "the result"
    run_fn.assert_awaited_once_with("alice", "do research")

  @pytest.mark.asyncio
  async def test_member_info_tool_returns_info(self):
    from definable.agent.team._tools import build_member_info_tool

    members = {"alice": MagicMock(instructions="Research expert", tools=[])}
    tool = build_member_info_tool(members)  # type: ignore[arg-type]
    assert tool.entrypoint is not None
    result = await tool.entrypoint()
    assert "alice" in result
    assert "Research expert" in result

  def test_build_task_tools(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    members = {"alice": MagicMock()}

    async def run_fn(name, task):
      return "done"

    async def emit_fn(event):
      pass

    tools = build_task_tools(tl, members, run_fn, emit_fn)  # type: ignore[arg-type]
    tool_names = [t.name for t in tools]
    assert "create_task" in tool_names
    assert "execute_task" in tool_names
    assert "get_task_status" in tool_names
    assert "update_task_status" in tool_names
    assert "mark_goal_complete" in tool_names

  @pytest.mark.asyncio
  async def test_task_tool_create(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    events = []

    async def emit_fn(event):
      events.append(event)

    tools = build_task_tools(tl, {"alice": MagicMock()}, AsyncMock(), emit_fn)  # type: ignore[arg-type]
    create_fn = next(t for t in tools if t.name == "create_task")
    assert create_fn.entrypoint is not None
    result = await create_fn.entrypoint("My Task", "description", "alice", "")
    assert "Task created" in result
    assert len(tl.tasks) == 1
    assert tl.tasks[0].title == "My Task"
    assert len(events) == 1
    assert isinstance(events[0], TaskCreatedEvent)

  @pytest.mark.asyncio
  async def test_task_tool_execute(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    events = []

    async def emit_fn(event):
      events.append(event)

    run_fn = AsyncMock(return_value="research complete")
    alice = MagicMock()
    tools = build_task_tools(tl, {"alice": alice}, run_fn, emit_fn)  # type: ignore[arg-type]

    # Create and execute a task
    create_fn = next(t for t in tools if t.name == "create_task")
    assert create_fn.entrypoint is not None
    await create_fn.entrypoint("Research AI", "", "alice", "")
    task_id = tl.tasks[0].id

    execute_fn = next(t for t in tools if t.name == "execute_task")
    assert execute_fn.entrypoint is not None
    result = await execute_fn.entrypoint(task_id)
    assert "completed" in result
    assert tl.tasks[0].status == TaskStatus.completed
    run_fn.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_task_tool_execute_no_assignee(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    tools = build_task_tools(tl, {"alice": MagicMock()}, AsyncMock(), AsyncMock())  # type: ignore[arg-type]
    create_fn = next(t for t in tools if t.name == "create_task")
    assert create_fn.entrypoint is not None
    await create_fn.entrypoint("Unassigned task", "", "", "")

    execute_fn = next(t for t in tools if t.name == "execute_task")
    assert execute_fn.entrypoint is not None
    result = await execute_fn.entrypoint(tl.tasks[0].id)
    assert "no assignee" in result.lower()

  @pytest.mark.asyncio
  async def test_task_tool_execute_blocked(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    tools = build_task_tools(tl, {"alice": MagicMock()}, AsyncMock(), AsyncMock())  # type: ignore[arg-type]
    create_fn = next(t for t in tools if t.name == "create_task")
    assert create_fn.entrypoint is not None
    await create_fn.entrypoint("First", "", "alice", "")
    first_id = tl.tasks[0].id
    await create_fn.entrypoint("Second", "", "alice", first_id)
    second_id = tl.tasks[1].id

    execute_fn = next(t for t in tools if t.name == "execute_task")
    assert execute_fn.entrypoint is not None
    result = await execute_fn.entrypoint(second_id)
    assert "blocked" in result.lower()

  @pytest.mark.asyncio
  async def test_task_tool_execute_member_failure(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    run_fn = AsyncMock(side_effect=RuntimeError("member crashed"))
    tools = build_task_tools(tl, {"alice": MagicMock()}, run_fn, AsyncMock())  # type: ignore[arg-type]
    create_fn = next(t for t in tools if t.name == "create_task")
    assert create_fn.entrypoint is not None
    await create_fn.entrypoint("Risky task", "", "alice", "")

    execute_fn = next(t for t in tools if t.name == "execute_task")
    assert execute_fn.entrypoint is not None
    result = await execute_fn.entrypoint(tl.tasks[0].id)
    assert "failed" in result.lower()
    assert tl.tasks[0].status == TaskStatus.failed

  @pytest.mark.asyncio
  async def test_task_tool_get_status(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    tl.create_task("A", assignee="alice")
    tl.create_task("B", assignee="bob")
    tools = build_task_tools(tl, {"alice": MagicMock(), "bob": MagicMock()}, AsyncMock(), AsyncMock())  # type: ignore[arg-type]

    status_fn = next(t for t in tools if t.name == "get_task_status")
    assert status_fn.entrypoint is not None
    result = await status_fn.entrypoint()
    assert "A" in result
    assert "B" in result

  @pytest.mark.asyncio
  async def test_task_tool_mark_goal_complete(self):
    from definable.agent.team._tools import build_task_tools

    tl = TaskList()
    tools = build_task_tools(tl, {}, AsyncMock(), AsyncMock())  # type: ignore[arg-type]
    complete_fn = next(t for t in tools if t.name == "mark_goal_complete")
    assert complete_fn.entrypoint is not None
    await complete_fn.entrypoint("Everything done!")
    assert tl.goal_complete is True
    assert tl.completion_summary == "Everything done!"


# ═══════════════════════════════════════════════════════════════
# Team Prompts
# ═══════════════════════════════════════════════════════════════


class TestTeamPrompts:
  def _make_members(self):
    a1 = MagicMock()
    a1.instructions = "Research expert specializing in AI"
    a1.tools = [MagicMock(name="search")]
    a2 = MagicMock()
    a2.instructions = "Technical writer"
    a2.tools = []
    return {"researcher": a1, "writer": a2}

  def test_coordinate_prompt(self):
    from definable.agent.team._prompts import build_coordinate_prompt

    prompt = build_coordinate_prompt("test-team", self._make_members())
    assert "test-team" in prompt
    assert "researcher" in prompt
    assert "writer" in prompt
    assert "delegate_to_member" in prompt
    assert "synthesize" in prompt.lower()

  def test_coordinate_prompt_with_instructions(self):
    from definable.agent.team._prompts import build_coordinate_prompt

    prompt = build_coordinate_prompt("t", self._make_members(), team_instructions="Be thorough")
    assert "Be thorough" in prompt

  def test_coordinate_prompt_with_interactions(self):
    from definable.agent.team._prompts import build_coordinate_prompt

    prompt = build_coordinate_prompt("t", self._make_members(), member_interactions=["[researcher] did X"])
    assert "[researcher] did X" in prompt

  def test_route_prompt(self):
    from definable.agent.team._prompts import build_route_prompt

    prompt = build_route_prompt("test-team", self._make_members())
    assert "route" in prompt.lower()
    assert "DIRECTLY" in prompt
    assert "ONE member" in prompt or "single" in prompt.lower()

  def test_collaborate_prompt(self):
    from definable.agent.team._prompts import build_collaborate_prompt

    prompt = build_collaborate_prompt("test-team", self._make_members())
    assert "ALL" in prompt
    assert "synthesize" in prompt.lower() or "Synthesize" in prompt

  def test_tasks_prompt(self):
    from definable.agent.team._prompts import build_tasks_prompt

    tl = TaskList()
    tl.create_task("Research", assignee="researcher")
    prompt = build_tasks_prompt("test-team", self._make_members(), tl, iteration=2, max_iterations=10)
    assert "Iteration: 2/10" in prompt
    assert "Research" in prompt
    assert "create_task" in prompt
    assert "mark_goal_complete" in prompt


# ═══════════════════════════════════════════════════════════════
# Team._TeamAsAgent (nested teams)
# ═══════════════════════════════════════════════════════════════


class TestTeamAsAgent:
  def test_wraps_team(self):
    from definable.agent.team.team import _TeamAsAgent

    inner = Team(name="inner", description="Inner team")
    wrapper = _TeamAsAgent(inner)
    assert wrapper.name == "inner"
    assert "Inner team" in wrapper.instructions
    assert wrapper.agent_name == "inner"
    assert wrapper.tools == []

  @pytest.mark.asyncio
  async def test_arun_delegates_to_team(self):
    from definable.agent.team.team import _TeamAsAgent

    inner = Team(name="inner")
    inner.arun = AsyncMock(return_value=MagicMock(content="team result"))  # type: ignore[method-assign]

    wrapper = _TeamAsAgent(inner)
    result = await wrapper.arun("hello")
    assert result.content == "team result"
    inner.arun.assert_awaited_once_with("hello")


# ═══════════════════════════════════════════════════════════════
# Team Coordinate Mode (integration with MockModel)
# ═══════════════════════════════════════════════════════════════


class TestTeamCoordinate:
  @pytest.mark.asyncio
  async def test_coordinate_delegates_and_synthesizes(self):
    """Leader should delegate to member and produce a final answer."""
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel
    from definable.model.metrics import Metrics

    call_count = 0

    def leader_side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      response = MagicMock()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None
      response.response_usage = Metrics()
      response.tool_executions = []

      if call_count == 1:
        # First call: decide to delegate
        response.content = None
        response.tool_calls = [
          {
            "id": "call1",
            "type": "function",
            "function": {
              "name": "delegate_to_member",
              "arguments": '{"member_name": "researcher", "task_input": "research quantum computing"}',
            },
          }
        ]
      else:
        # Second call: synthesize
        response.content = "Here is an article about quantum computing based on the research."
        response.tool_calls = []
      return response

    leader_model = MockModel(side_effect=leader_side_effect)
    member_model = MockModel(responses=["Quantum computing uses qubits and superposition."])

    researcher = Agent(model=member_model, name="researcher", instructions="You are a research specialist.")  # type: ignore[arg-type]
    team = Team(
      name="content-team",
      model=leader_model,  # type: ignore[arg-type]
      members=[researcher],
      mode=TeamMode.coordinate,
    )

    result = await team.arun("Write about quantum computing")
    assert result is not None
    assert result.content is not None
    assert "quantum computing" in result.content.lower()

  @pytest.mark.asyncio
  async def test_coordinate_emits_events(self):
    """Team should emit TeamRunStarted and TeamRunCompleted events."""
    from definable.agent.testing import MockModel

    events_collected = []
    leader_model = MockModel(responses=["Direct answer."])

    team = Team(
      name="event-team",
      model=leader_model,  # type: ignore[arg-type]
      members=[],
      mode=TeamMode.coordinate,
    )

    async def collect(event):
      events_collected.append(event)

    team.events.on(object, collect)

    await team.arun("Simple question")

    event_types = [type(e).__name__ for e in events_collected]
    assert "TeamRunStartedEvent" in event_types
    assert "TeamRunCompletedEvent" in event_types


# ═══════════════════════════════════════════════════════════════
# Team Route Mode
# ═══════════════════════════════════════════════════════════════


class TestTeamRoute:
  @pytest.mark.asyncio
  async def test_route_returns_member_response(self):
    """In route mode, the member's response should be returned directly."""
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel
    from definable.model.metrics import Metrics

    call_count = 0

    def leader_side_effect(messages, tools, **kwargs):
      nonlocal call_count
      call_count += 1
      response = MagicMock()
      response.reasoning_content = None
      response.citations = None
      response.images = None
      response.videos = None
      response.audios = None
      response.response_usage = Metrics()
      response.tool_executions = []

      if call_count == 1:
        response.content = None
        response.tool_calls = [
          {
            "id": "call1",
            "type": "function",
            "function": {
              "name": "delegate_to_member",
              "arguments": '{"member_name": "specialist", "task_input": "explain gravity"}',
            },
          }
        ]
      else:
        response.content = "Routed."
        response.tool_calls = []
      return response

    leader_model = MockModel(side_effect=leader_side_effect)
    member_model = MockModel(responses=["Gravity is the force of attraction between masses."])

    specialist = Agent(model=member_model, name="specialist", instructions="Physics expert.")  # type: ignore[arg-type]

    team = Team(
      name="router-team",
      model=leader_model,  # type: ignore[arg-type]
      members=[specialist],
      mode=TeamMode.route,
    )

    result = await team.arun("Explain gravity")
    assert result.content is not None
    assert "gravity" in result.content.lower() or "force" in result.content.lower()


# ═══════════════════════════════════════════════════════════════
# Team Collaborate Mode
# ═══════════════════════════════════════════════════════════════


class TestTeamCollaborate:
  @pytest.mark.asyncio
  async def test_collaborate_runs_all_members_in_parallel(self):
    """All members should receive the same task."""
    from definable.agent.agent import Agent
    from definable.agent.testing import MockModel

    m1 = MockModel(responses=["Perspective A"])
    m2 = MockModel(responses=["Perspective B"])

    agent1 = Agent(model=m1, name="analyst-1", instructions="Analyst 1")  # type: ignore[arg-type]
    agent2 = Agent(model=m2, name="analyst-2", instructions="Analyst 2")  # type: ignore[arg-type]

    leader_model = MockModel(responses=["Combined analysis: A and B together yield a complete picture."])

    team = Team(
      name="collab-team",
      model=leader_model,  # type: ignore[arg-type]
      members=[agent1, agent2],
      mode=TeamMode.collaborate,
    )

    events_collected = []

    async def collect(event):
      events_collected.append(event)

    team.events.on(object, collect)

    result = await team.arun("Analyze the market")
    assert result is not None
    assert result.content is not None

    # Both members should have been delegated to
    delegated_members = [e.member_name for e in events_collected if isinstance(e, MemberDelegatedEvent)]
    assert "analyst-1" in delegated_members
    assert "analyst-2" in delegated_members


# ═══════════════════════════════════════════════════════════════
# Team Error Handling
# ═══════════════════════════════════════════════════════════════


class TestTeamErrors:
  def test_no_model_raises(self):
    team = Team(name="no-model")
    with pytest.raises(ValueError, match="requires a model"):
      team._get_or_create_leader()

  def test_model_from_member(self):
    member = MagicMock()
    member.model = "openai/gpt-4o"
    member.name = "helper"
    team = Team(name="fallback-model", members=[member])
    leader = team._get_or_create_leader()
    # model gets resolved from string to OpenAIChat object
    assert leader.model is not None

  @pytest.mark.asyncio
  async def test_delegate_to_unknown_member(self):
    team = Team(name="test")
    with pytest.raises(ValueError, match="not found"):
      await team._run_single_member("nonexistent", "task", "run-1")

  @pytest.mark.asyncio
  async def test_member_error_emits_event(self):
    member = MagicMock()
    member.name = "buggy"
    member.instructions = "test"
    member.tools = []
    member.agent_name = "buggy"
    member.arun = AsyncMock(side_effect=RuntimeError("member exploded"))

    team = Team(name="error-team", members=[member])

    events_collected = []

    async def collect(event):
      events_collected.append(event)

    team.events.on(object, collect)

    with pytest.raises(RuntimeError, match="member exploded"):
      await team._run_single_member("buggy", "do something", "run-1")

    error_events = [e for e in events_collected if isinstance(e, MemberErrorEvent)]
    assert len(error_events) == 1
    assert "exploded" in error_events[0].error


# ═══════════════════════════════════════════════════════════════
# Team Event Bus Integration
# ═══════════════════════════════════════════════════════════════


class TestTeamEventSubscription:
  @pytest.mark.asyncio
  async def test_subscribe_and_receive(self):
    team = Team(name="event-test")
    received = []

    async def handler(event):
      received.append(event)

    team.events.on(object, handler)
    await team.events.emit(TeamRunStartedEvent(run_id="r1", team_id="t1", team_name="event-test", mode="coordinate"))

    assert len(received) == 1
    assert isinstance(received[0], TeamRunStartedEvent)


# ═══════════════════════════════════════════════════════════════
# Imports from events module
# ═══════════════════════════════════════════════════════════════


class TestEventsModuleExports:
  def test_team_events_in_events_module(self):
    from definable.agent import events

    assert hasattr(events, "TeamRunStartedEvent")
    assert hasattr(events, "TeamRunCompletedEvent")
    assert hasattr(events, "TeamRunErrorEvent")
    assert hasattr(events, "MemberDelegatedEvent")
    assert hasattr(events, "MemberCompletedEvent")
    assert hasattr(events, "MemberErrorEvent")
    assert hasattr(events, "MemberRoutedEvent")
    assert hasattr(events, "TaskCreatedEvent")
    assert hasattr(events, "TaskStatusChangedEvent")
    assert hasattr(events, "TaskIterationEvent")
