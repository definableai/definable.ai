"""Team execution modes."""

from enum import Enum


class TeamMode(str, Enum):
  """Execution mode for a Team.

  Controls how the team leader coordinates work with member agents.
  """

  coordinate = "coordinate"
  """Default supervisor pattern. Leader picks members, crafts tasks, synthesizes responses."""

  route = "route"
  """Router pattern. Leader routes to a single specialist and returns their response directly."""

  collaborate = "collaborate"
  """Broadcast pattern. All members receive the same task and execute in parallel."""

  tasks = "tasks"
  """Autonomous task-based execution. Leader decomposes goals into a shared task list,
  delegates tasks to members, and loops until all work is complete."""
