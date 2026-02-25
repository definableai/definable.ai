"""Definable Team — multi-agent coordination.

Four execution modes for composing agents into teams:

- **coordinate** (default): Leader picks members, crafts tasks, synthesizes.
- **route**: Leader routes to a single specialist; returns their response directly.
- **collaborate**: All members work in parallel; leader synthesizes.
- **tasks**: Leader decomposes goals into a shared task list; autonomous loop.

Quick Start::

    from definable.agent import Agent
    from definable.agent.team import Team, TeamMode

    researcher = Agent(model="openai/gpt-4o", instructions="Research specialist.")
    writer = Agent(model="openai/gpt-4o", instructions="Technical writer.")

    team = Team(
        name="content-team",
        model="openai/gpt-4o",
        members=[researcher, writer],
        mode=TeamMode.coordinate,
    )
    result = await team.arun("Write about quantum computing.")
"""

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
from definable.agent.team.mode import TeamMode
from definable.agent.team.task import Task, TaskList, TaskStatus
from definable.agent.team.team import Team

__all__ = [
  # Core
  "Team",
  "TeamMode",
  # Task model
  "Task",
  "TaskList",
  "TaskStatus",
  # Events
  "TeamRunStartedEvent",
  "TeamRunCompletedEvent",
  "TeamRunErrorEvent",
  "MemberDelegatedEvent",
  "MemberCompletedEvent",
  "MemberErrorEvent",
  "MemberRoutedEvent",
  "TaskCreatedEvent",
  "TaskStatusChangedEvent",
  "TaskIterationEvent",
]
