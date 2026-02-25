"""Team-specific events emitted during team execution."""

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from definable.agent.run.base import BaseRunOutputEvent


@dataclass
class BaseTeamEvent(BaseRunOutputEvent):
  """Base class for all team events — carries common identifiers."""

  created_at: int = field(default_factory=lambda: int(time()))
  event: str = ""
  run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Team lifecycle events
# ---------------------------------------------------------------------------


@dataclass
class TeamRunStartedEvent(BaseTeamEvent):
  """Emitted when a team run begins."""

  event: str = "team_run_started"
  team_id: str = ""
  team_name: str = ""
  mode: str = ""
  member_names: List[str] = field(default_factory=list)


@dataclass
class TeamRunCompletedEvent(BaseTeamEvent):
  """Emitted when a team run completes successfully."""

  event: str = "team_run_completed"
  team_id: str = ""
  team_name: str = ""
  content: Optional[str] = None


@dataclass
class TeamRunErrorEvent(BaseTeamEvent):
  """Emitted when a team run fails."""

  event: str = "team_run_error"
  team_id: str = ""
  team_name: str = ""
  error: str = ""


# ---------------------------------------------------------------------------
# Member delegation events
# ---------------------------------------------------------------------------


@dataclass
class MemberDelegatedEvent(BaseTeamEvent):
  """Emitted when the leader delegates a task to a member."""

  event: str = "member_delegated"
  member_name: str = ""
  task_input: str = ""
  mode: str = ""


@dataclass
class MemberCompletedEvent(BaseTeamEvent):
  """Emitted when a member finishes its delegated task."""

  event: str = "member_completed"
  member_name: str = ""
  content: Optional[str] = None
  metrics: Optional[Dict[str, Any]] = None


@dataclass
class MemberErrorEvent(BaseTeamEvent):
  """Emitted when a member fails its delegated task."""

  event: str = "member_error"
  member_name: str = ""
  error: str = ""


# ---------------------------------------------------------------------------
# Task-mode events (mode=tasks)
# ---------------------------------------------------------------------------


@dataclass
class TaskCreatedEvent(BaseTeamEvent):
  """Emitted when a new task is created in the shared task list."""

  event: str = "task_created"
  task_id: str = ""
  title: str = ""
  assignee: Optional[str] = None


@dataclass
class TaskStatusChangedEvent(BaseTeamEvent):
  """Emitted when a task status changes."""

  event: str = "task_status_changed"
  task_id: str = ""
  old_status: str = ""
  new_status: str = ""


@dataclass
class TaskIterationEvent(BaseTeamEvent):
  """Emitted at the start of each task-loop iteration."""

  event: str = "task_iteration"
  iteration: int = 0
  pending_count: int = 0
  completed_count: int = 0
  failed_count: int = 0


# ---------------------------------------------------------------------------
# Routing events
# ---------------------------------------------------------------------------


@dataclass
class MemberRoutedEvent(BaseTeamEvent):
  """Emitted when a request is routed to a specific member (mode=route)."""

  event: str = "member_routed"
  member_name: str = ""
  reason: str = ""
