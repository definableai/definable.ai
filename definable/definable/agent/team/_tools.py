"""Auto-injected tools for team leader agents.

These tools are generated at runtime and allow the leader model to
interact with team members and (in tasks mode) manage the shared task list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

from definable.tool.function import Function

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.team.task import TaskList


# ---------------------------------------------------------------------------
# Coordinate / Route / Collaborate tools
# ---------------------------------------------------------------------------


def build_delegate_tool(
  members: Dict[str, "Agent"],
  run_member_fn: Callable[..., Any],
) -> Function:
  """Build a tool that lets the leader delegate work to a specific member.

  Args:
    members: Mapping of member name → Agent instance.
    run_member_fn: Async callable ``(member_name, task_input) -> str``.
  """
  member_descriptions = []
  for name, agent in members.items():
    desc = agent.instructions[:200] if agent.instructions else "No description"
    member_descriptions.append(f"- {name}: {desc}")
  member_list_str = "\n".join(member_descriptions)

  async def delegate_to_member(member_name: str, task_input: str) -> str:
    """Delegate a task to a specific team member and get their response.

    Choose the most appropriate member based on their expertise.
    Craft a clear, specific task description for them.
    """
    return await run_member_fn(member_name, task_input)

  fn = Function(
    name="delegate_to_member",
    description=(
      "Delegate a task to a specific team member. "
      "Choose the best member for the job and provide clear instructions.\n\n"
      f"Available members:\n{member_list_str}"
    ),
    parameters={
      "type": "object",
      "properties": {
        "member_name": {
          "type": "string",
          "description": "Name of the team member to delegate to.",
          "enum": list(members.keys()),
        },
        "task_input": {
          "type": "string",
          "description": "The specific task or question for this member.",
        },
      },
      "required": ["member_name", "task_input"],
    },
    entrypoint=delegate_to_member,
    skip_entrypoint_processing=True,
    add_instructions=False,
  )
  return fn


def build_member_info_tool(members: Dict[str, "Agent"]) -> Function:
  """Build a tool that returns information about all team members."""

  member_info: List[Dict[str, str]] = []
  for name, agent in members.items():
    info: Dict[str, str] = {"name": name}
    if agent.instructions:
      info["instructions"] = agent.instructions[:500]
    tool_names = [str(getattr(t, "name", "")) for t in agent.tools] if agent.tools else []
    tool_names = [n for n in tool_names if n]
    if tool_names:
      info["tools"] = ", ".join(tool_names[:20])
    member_info.append(info)

  async def get_member_information() -> str:
    """Get detailed information about all team members, including their capabilities and tools."""
    lines = []
    for info in member_info:
      lines.append(f"## {info['name']}")
      if "instructions" in info:
        lines.append(f"Instructions: {info['instructions']}")
      if "tools" in info:
        lines.append(f"Tools: {info['tools']}")
      lines.append("")
    return "\n".join(lines)

  return Function(
    name="get_member_information",
    description="Get detailed information about all team members, their roles, and available tools.",
    parameters={"type": "object", "properties": {}},
    entrypoint=get_member_information,
    skip_entrypoint_processing=True,
    add_instructions=False,
  )


# ---------------------------------------------------------------------------
# Task-mode tools (mode=tasks)
# ---------------------------------------------------------------------------


def build_task_tools(
  task_list: "TaskList",
  members: Dict[str, "Agent"],
  run_member_fn: Callable[..., Any],
  emit_fn: Callable[..., Any],
) -> List[Function]:
  """Build tools for autonomous task management (mode=tasks).

  Returns a list of Function objects the leader uses to manage and
  execute the shared task list.
  """
  from definable.agent.team.task import TaskStatus

  member_names = list(members.keys())

  async def create_task(
    title: str,
    description: str = "",
    assignee: str = "",
    dependencies: str = "",
  ) -> str:
    """Create a new task in the shared task list.

    Break down the overall goal into concrete, actionable tasks.
    Assign them to the most appropriate team member.
    """
    dep_list = [d.strip() for d in dependencies.split(",") if d.strip()] if dependencies else []
    task = task_list.create_task(
      title=title,
      description=description,
      assignee=assignee or None,
      dependencies=dep_list,
    )

    # Emit event
    from definable.agent.team.events import TaskCreatedEvent

    await emit_fn(
      TaskCreatedEvent(
        run_id="",
        task_id=task.id,
        title=task.title,
        assignee=task.assignee,
      )
    )

    return f"Task created: [{task.id}] {task.title} (status: {task.status.value})"

  async def execute_task(task_id: str) -> str:
    """Execute a task by delegating it to the assigned member.

    The assigned member will receive the task description and return their result.
    """
    task = task_list.get_task(task_id)
    if task is None:
      return f"Error: Task {task_id} not found."

    if task.assignee is None:
      return f"Error: Task {task_id} has no assignee. Assign it first."

    if task.assignee not in members:
      return f"Error: Member '{task.assignee}' not found. Available: {member_names}"

    if task.status in {TaskStatus.completed, TaskStatus.failed}:
      return f"Task {task_id} is already {task.status.value}. Result: {task.result or 'N/A'}"

    if task.status == TaskStatus.blocked:
      return f"Task {task_id} is blocked. Resolve dependencies first: {task.dependencies}"

    # Mark in-progress
    old_status = task.status.value
    task_list.update_task(task_id, status=TaskStatus.in_progress)

    from definable.agent.team.events import TaskStatusChangedEvent

    await emit_fn(
      TaskStatusChangedEvent(
        run_id="",
        task_id=task_id,
        old_status=old_status,
        new_status="in_progress",
      )
    )

    # Build context for the member
    task_prompt = f"Task: {task.title}"
    if task.description:
      task_prompt += f"\n\nDetails: {task.description}"
    if task.notes:
      task_prompt += "\n\nNotes:\n" + "\n".join(f"- {n}" for n in task.notes)

    try:
      result = await run_member_fn(task.assignee, task_prompt)
      task_list.update_task(task_id, status=TaskStatus.completed, result=result)

      await emit_fn(
        TaskStatusChangedEvent(
          run_id="",
          task_id=task_id,
          old_status="in_progress",
          new_status="completed",
        )
      )

      return f"Task [{task_id}] completed. Result:\n{result}"

    except Exception as exc:
      error_msg = str(exc)
      task_list.update_task(task_id, status=TaskStatus.failed, result=f"Error: {error_msg}")

      await emit_fn(
        TaskStatusChangedEvent(
          run_id="",
          task_id=task_id,
          old_status="in_progress",
          new_status="failed",
        )
      )

      return f"Task [{task_id}] failed: {error_msg}"

  async def get_task_status() -> str:
    """Get the current status of all tasks in the task list."""
    return task_list.get_summary_string()

  async def update_task_status(task_id: str, status: str, result: str = "", note: str = "") -> str:
    """Update a task's status, result, or add a note."""
    task = task_list.get_task(task_id)
    if task is None:
      return f"Error: Task {task_id} not found."

    old_status = task.status.value
    updates = {}
    if status:
      updates["status"] = status
    if result:
      updates["result"] = result
    if note:
      task.notes.append(note)

    task_list.update_task(task_id, **updates)

    if status and status != old_status:
      from definable.agent.team.events import TaskStatusChangedEvent

      await emit_fn(
        TaskStatusChangedEvent(
          run_id="",
          task_id=task_id,
          old_status=old_status,
          new_status=status,
        )
      )

    return f"Task [{task_id}] updated. Status: {task.status.value}"

  async def mark_goal_complete(summary: str) -> str:
    """Mark the overall goal as complete with a summary of what was accomplished."""
    task_list.goal_complete = True
    task_list.completion_summary = summary
    return f"Goal marked complete. Summary: {summary}"

  tools = [
    Function(
      name="create_task",
      description=(
        f"Create a new task in the shared task list. Break down goals into concrete tasks. Available members to assign to: {', '.join(member_names)}"
      ),
      parameters={
        "type": "object",
        "properties": {
          "title": {"type": "string", "description": "Short title for the task."},
          "description": {"type": "string", "description": "Detailed description of what needs to be done."},
          "assignee": {
            "type": "string",
            "description": "Team member to assign this task to.",
            "enum": member_names,
          },
          "dependencies": {
            "type": "string",
            "description": "Comma-separated task IDs this task depends on.",
          },
        },
        "required": ["title"],
      },
      entrypoint=create_task,
      skip_entrypoint_processing=True,
      add_instructions=False,
    ),
    Function(
      name="execute_task",
      description="Execute a task by delegating it to the assigned member agent.",
      parameters={
        "type": "object",
        "properties": {
          "task_id": {"type": "string", "description": "ID of the task to execute."},
        },
        "required": ["task_id"],
      },
      entrypoint=execute_task,
      skip_entrypoint_processing=True,
      add_instructions=False,
    ),
    Function(
      name="get_task_status",
      description="Get the current status summary of all tasks.",
      parameters={"type": "object", "properties": {}},
      entrypoint=get_task_status,
      skip_entrypoint_processing=True,
      add_instructions=False,
    ),
    Function(
      name="update_task_status",
      description="Update a task's status, add results, or add notes.",
      parameters={
        "type": "object",
        "properties": {
          "task_id": {"type": "string", "description": "ID of the task to update."},
          "status": {
            "type": "string",
            "description": "New status for the task.",
            "enum": ["pending", "in_progress", "completed", "failed"],
          },
          "result": {"type": "string", "description": "Result or output of the task."},
          "note": {"type": "string", "description": "A note to add to the task."},
        },
        "required": ["task_id"],
      },
      entrypoint=update_task_status,
      skip_entrypoint_processing=True,
      add_instructions=False,
    ),
    Function(
      name="mark_goal_complete",
      description="Mark the overall goal as complete when all tasks are done. Provide a summary.",
      parameters={
        "type": "object",
        "properties": {
          "summary": {"type": "string", "description": "Summary of what was accomplished."},
        },
        "required": ["summary"],
      },
      entrypoint=mark_goal_complete,
      skip_entrypoint_processing=True,
      add_instructions=False,
    ),
  ]

  return tools
