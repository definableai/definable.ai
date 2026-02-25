"""System prompt builders for each team mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.team.task import TaskList


def _member_roster(members: Dict[str, "Agent"]) -> str:
  """Build a member roster section for the system prompt."""
  lines = ["## Team Members"]
  for name, agent in members.items():
    desc = agent.instructions[:300] if agent.instructions else "General-purpose agent"
    tool_names = [str(getattr(t, "name", "")) for t in agent.tools] if agent.tools else []
    tool_names = [n for n in tool_names if n]
    tools_str = f" | Tools: {', '.join(tool_names[:10])}" if tool_names else ""
    lines.append(f"- **{name}**: {desc}{tools_str}")
  return "\n".join(lines)


def build_coordinate_prompt(
  team_name: str,
  members: Dict[str, "Agent"],
  team_instructions: Optional[str] = None,
  member_interactions: Optional[List[str]] = None,
) -> str:
  """Build system prompt for coordinate mode (supervisor pattern)."""
  roster = _member_roster(members)

  prompt = f"""You are the leader of team "{team_name}". Your role is to coordinate your team members to accomplish the user's request.

{roster}

## Your Responsibilities
1. Analyze the user's request and determine which team member(s) are best suited to handle it.
2. Use the `delegate_to_member` tool to assign specific tasks to members.
3. You may delegate to multiple members if the task requires different expertise.
4. After receiving responses from members, synthesize their results into a comprehensive final answer.
5. If a member's response is insufficient, you may delegate additional follow-up tasks.

## Guidelines
- Be strategic about delegation — choose the right member for each subtask.
- Provide clear, specific instructions when delegating.
- Synthesize member responses into a coherent final answer — do not just concatenate them."""

  if team_instructions:
    prompt += f"\n\n## Team Instructions\n{team_instructions}"

  if member_interactions:
    prompt += "\n\n## Prior Member Interactions\n" + "\n".join(member_interactions)

  return prompt


def build_route_prompt(
  team_name: str,
  members: Dict[str, "Agent"],
  team_instructions: Optional[str] = None,
) -> str:
  """Build system prompt for route mode (router pattern)."""
  roster = _member_roster(members)

  prompt = f"""You are the router for team "{team_name}". Your ONLY job is to route the user's request to the single most appropriate team member.

{roster}

## Your Responsibilities
1. Analyze the user's request.
2. Determine which single team member is the best specialist to handle it.
3. Use `delegate_to_member` to route the request to that member.
4. Return the member's response DIRECTLY — do NOT modify, summarize, or add to it.

## Guidelines
- Route to exactly ONE member per request.
- Choose based on expertise match.
- Pass the user's original request (or a clarified version) as the task input.
- The member's response IS the final response."""

  if team_instructions:
    prompt += f"\n\n## Team Instructions\n{team_instructions}"

  return prompt


def build_collaborate_prompt(
  team_name: str,
  members: Dict[str, "Agent"],
  team_instructions: Optional[str] = None,
) -> str:
  """Build system prompt for collaborate mode (broadcast pattern)."""
  roster = _member_roster(members)
  member_names = list(members.keys())

  prompt = f"""You are the coordinator for team "{team_name}". In this mode, ALL team members work on the same task simultaneously.

{roster}

## Your Responsibilities
1. The user's request will be sent to ALL members: {", ".join(member_names)}.
2. You will receive all of their responses.
3. Synthesize their collective output into the best possible final answer.
4. Resolve any contradictions between member responses.
5. Combine complementary insights.

## Guidelines
- Every member will receive the same task — this is intentional for gathering multiple perspectives.
- Weight responses based on each member's area of expertise.
- If members disagree, explain the different viewpoints and provide your best judgment."""

  if team_instructions:
    prompt += f"\n\n## Team Instructions\n{team_instructions}"

  return prompt


def build_tasks_prompt(
  team_name: str,
  members: Dict[str, "Agent"],
  task_list: "TaskList",
  team_instructions: Optional[str] = None,
  iteration: int = 0,
  max_iterations: int = 10,
) -> str:
  """Build system prompt for tasks mode (autonomous task decomposition)."""
  roster = _member_roster(members)
  task_summary = task_list.get_summary_string()

  prompt = f"""You are the leader of team "{team_name}" operating in autonomous task mode.

{roster}

## Your Responsibilities
1. Decompose the user's goal into concrete, actionable tasks using `create_task`.
2. Assign tasks to the most appropriate team members.
3. Execute tasks using `execute_task` — the assigned member will handle it.
4. Track progress using `get_task_status`.
5. When all tasks are complete, call `mark_goal_complete` with a summary.

## Current State
Iteration: {iteration}/{max_iterations}

{task_summary}

## Guidelines
- Break complex goals into small, focused tasks.
- Set dependencies between tasks that require sequential execution.
- Execute tasks that are ready (pending, no blocked dependencies).
- If a task fails, decide whether to retry, reassign, or create an alternative task.
- Call `mark_goal_complete` when the overall goal is achieved.
- You have {max_iterations - iteration} iterations remaining."""

  if team_instructions:
    prompt += f"\n\n## Team Instructions\n{team_instructions}"

  return prompt
