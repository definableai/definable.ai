# Team -- Multi-Agent Coordination

The Team module composes multiple Agent instances under a leader that coordinates their execution. Four execution modes cover the common multi-agent patterns: supervised delegation, routing, parallel collaboration, and autonomous task decomposition.

```python
from definable.agent import Agent
from definable.agent.team import Team, TeamMode

researcher = Agent(model="openai/gpt-4o", instructions="Research specialist.")
writer = Agent(model="openai/gpt-4o", instructions="Technical writer.")

team = Team(
  name="content-team",
  model="openai/gpt-4o",
  members=[researcher, writer],
  mode=TeamMode.coordinate,
  instructions="Produce well-researched technical content.",
)
result = await team.arun("Write about quantum computing")
print(result.content)
```

---

## Architecture

A `Team` creates an internal leader agent powered by the team's `model`. The leader receives auto-injected tools for delegation and task management. Members are the workers -- they receive tasks from the leader and return results.

```
                         Team.arun(instruction)
                                  |
                                  v
                    +----------------------------+
                    |      Leader Agent           |
                    | (auto-created, team.model)  |
                    | +------------------------+  |
                    | | Auto-injected tools:   |  |
                    | | - delegate_to_member   |  |
                    | | - get_member_info      |  |
                    | | - create_task (tasks)   |  |
                    | | - execute_task (tasks)  |  |
                    | | - get_task_status       |  |
                    | | - mark_goal_complete    |  |
                    | +------------------------+  |
                    +----------------------------+
                         /        |        \
                        v         v         v
                   +---------+ +--------+ +--------+
                   | Member  | | Member | | Member |
                   | Agent A | | Agent B| | Team C |
                   +---------+ +--------+ +--------+
```

---

## The Four Modes

```
  coordinate                route               collaborate              tasks
  (supervisor)             (router)             (broadcast)          (autonomous)

  +--------+            +--------+            +--------+            +--------+
  | Leader |            | Leader |            | Leader |            | Leader |
  +--------+            +--------+            +--------+            +--------+
   /   |   \                |                  / | | \              /   |   \
  v    v    v               v                 v  v v  v            v    v    v
 [A]  [B]  [C]            [Best]           [A] [B] [C] [D]      [TaskList]
  \    |    /               |                 \  | |  /           / | | \
   v   v   v                v                  v v v v           v  v v  v
  +--------+            +--------+           +--------+        [A] [B] [A]
  | Leader |            | Direct |           | Leader |          \  |  /
  |synthesize|          | return |           |synthesize|         v v v
  +--------+            +--------+           +--------+        +--------+
      |                     |                    |             | Leader |
      v                     v                    v             |summarize|
   RunOutput             RunOutput            RunOutput        +--------+
                                                                   |
                                                                   v
                                                                RunOutput
```

### coordinate (default)

The leader analyzes the request, picks the best member(s), delegates tasks via `delegate_to_member`, and synthesizes all responses into a final answer. The leader may invoke multiple members sequentially.

### route

The leader routes the request to the single most appropriate specialist. The specialist's response is returned directly without modification. Best for classification/dispatch patterns.

### collaborate

All members receive the same input and execute in parallel (`asyncio.gather`). The leader then synthesizes their collective output, resolving contradictions and combining insights.

### tasks

The leader decomposes the goal into a `TaskList`, assigns tasks to members, and loops until all tasks complete or `max_iterations` is reached. Supports task dependencies, failure handling, and autonomous re-planning.

---

## Exports

```python
from definable.agent.team import (
  # Core
  Team,
  TeamMode,
  # Task model (used internally in tasks mode; available for inspection)
  Task,
  TaskList,
  TaskStatus,
  # Events
  TeamRunStartedEvent,
  TeamRunCompletedEvent,
  TeamRunErrorEvent,
  MemberDelegatedEvent,
  MemberCompletedEvent,
  MemberErrorEvent,
  MemberRoutedEvent,
  TaskCreatedEvent,
  TaskStatusChangedEvent,
  TaskIterationEvent,
)
```

---

## Team

**Constructor:**
```python
Team(
  name: str = "",
  instructions: str = None,          # Team-level instructions for the leader
  description: str = None,           # Used when this team is nested as a member
  model: str | Model = None,         # Leader's model (required unless a member has one)
  members: list[Agent | Team] = [],  # Worker agents or nested teams
  mode: TeamMode = TeamMode.coordinate,
  max_iterations: int = 10,          # Tasks mode only: max leader iterations
  share_member_interactions: bool = False,  # Pass member outputs to subsequent delegates
  tools: list[Function] = None,      # Additional tools for the leader
  output_schema: type[BaseModel] = None,   # Structured output for final response
  debug: bool = False,
)
```

**Properties:**
- `team_id` -- UUID of the team instance
- `member_names` -- List of member name strings
- `events` -- `EventBus` for subscribing to team events

**Methods:**
```python
# Async (preferred)
result = await team.arun(
  instruction="Write about quantum computing",
  session_id=None,       # Optional session ID
  user_id=None,          # Optional user ID
  output_schema=None,    # Override structured output schema
)
# Returns: RunOutput (same as Agent.arun())

# Sync (convenience -- uses asyncio.run or ThreadPoolExecutor)
result = team.run("Write about quantum computing")
```

---

## TeamMode

```python
from definable.agent.team import TeamMode

TeamMode.coordinate   # "coordinate" -- leader picks members, synthesizes
TeamMode.route        # "route"      -- routes to single specialist
TeamMode.collaborate  # "collaborate"-- all members parallel, leader synthesizes
TeamMode.tasks        # "tasks"      -- autonomous task list loop
```

---

## Task Model

The `Task` and `TaskList` classes power the `tasks` mode. They are available for inspection and testing, though in production they are managed internally by the leader's auto-injected tools.

### TaskStatus

```python
from definable.agent.team import TaskStatus

TaskStatus.pending      # "pending"     -- ready to be picked up
TaskStatus.in_progress  # "in_progress" -- currently being executed
TaskStatus.completed    # "completed"   -- finished successfully
TaskStatus.failed       # "failed"      -- encountered an error
TaskStatus.blocked      # "blocked"     -- waiting on dependencies
```

### Task

```python
from definable.agent.team import Task

task = Task(
  id="",                   # Auto-generated (8-char UUID)
  title="Research topic",
  description="Find sources about quantum computing",
  status=TaskStatus.pending,
  assignee=None,           # Member name or None
  parent_id=None,          # ID of parent task
  dependencies=[],         # List of task IDs this depends on
  result=None,             # Result string after completion
  notes=[],                # List of note strings
  created_at=0.0,          # Auto-set to time()
)

task.to_dict()             # Serialize to dict
Task.from_dict(data)       # Deserialize from dict
```

### TaskList

```python
from definable.agent.team import TaskList

tl = TaskList()

# Create tasks with dependencies
t1 = tl.create_task("Research", "Find sources", assignee="researcher")
t2 = tl.create_task("Write", "Draft article", assignee="writer", dependencies=[t1.id])

# Query
tl.get_task(t1.id)                   # -> Task | None
tl.get_available_tasks()              # -> [t1] (t2 is blocked by dependency)
tl.get_available_tasks(for_assignee="researcher")  # -> [t1]

# Update
tl.update_task(t1.id, status="completed", result="Found 5 sources")
tl.get_available_tasks()              # -> [t2] (dependency satisfied)

# Status
tl.all_terminal()                     # True when all tasks are completed or failed
tl.get_summary_string()               # Formatted task list for display

# Serialization
tl.to_dict()
TaskList.from_dict(data)
```

**Dependency behavior:**
- A task with unresolved dependencies is automatically marked `blocked`.
- When a dependency completes, the blocked task returns to `pending`.
- When a dependency fails, the blocked task is automatically marked `failed`.
- `all_terminal()` returns `True` only when every task is `completed` or `failed`.

---

## Usage Examples

### Coordinate Mode -- Research Pipeline

```python
from definable.agent import Agent
from definable.agent.team import Team, TeamMode

researcher = Agent(
  model="openai/gpt-4o",
  instructions="You are a research specialist. Find authoritative sources and extract key findings.",
)
writer = Agent(
  model="openai/gpt-4o",
  instructions="You are a technical writer. Create clear, well-structured content.",
)
editor = Agent(
  model="openai/gpt-4o",
  instructions="You are an editor. Review for accuracy, clarity, and tone.",
)

team = Team(
  name="content-team",
  model="openai/gpt-4o",
  members=[researcher, writer, editor],
  mode=TeamMode.coordinate,
  instructions="Produce polished technical content. Research first, then write, then edit.",
)

result = await team.arun("Write a guide on WebSocket security best practices")
```

### Route Mode -- Support Triage

```python
tech_support = Agent(model="openai/gpt-4o", instructions="Technical support specialist.")
billing = Agent(model="openai/gpt-4o", instructions="Billing and account specialist.")
general = Agent(model="openai/gpt-4o", instructions="General customer support.")

team = Team(
  name="support",
  model="openai/gpt-4o-mini",   # Cheaper model for routing decisions
  members=[tech_support, billing, general],
  mode=TeamMode.route,
)

# Leader picks the best specialist; their response is returned directly
result = await team.arun("I can't connect to the API after upgrading")
```

### Collaborate Mode -- Multi-Perspective Analysis

```python
optimist = Agent(model="openai/gpt-4o", instructions="Analyze from a bullish perspective.")
pessimist = Agent(model="openai/gpt-4o", instructions="Analyze from a bearish perspective.")
neutral = Agent(model="openai/gpt-4o", instructions="Provide balanced, data-driven analysis.")

team = Team(
  name="analysis-panel",
  model="openai/gpt-4o",
  members=[optimist, pessimist, neutral],
  mode=TeamMode.collaborate,
  instructions="Synthesize all perspectives into a balanced assessment.",
)

result = await team.arun("Analyze the impact of AI on software engineering jobs")
```

### Tasks Mode -- Autonomous Project

```python
from definable.agent.team import Team, TeamMode

researcher = Agent(model="openai/gpt-4o", instructions="Deep research specialist.")
analyst = Agent(model="openai/gpt-4o", instructions="Data analyst. Structures information.")
writer = Agent(model="openai/gpt-4o", instructions="Technical writer. Creates final documents.")

team = Team(
  name="report-team",
  model="openai/gpt-4o",
  members=[researcher, analyst, writer],
  mode=TeamMode.tasks,
  max_iterations=10,
  instructions="Produce a comprehensive research report.",
)

result = await team.arun("Create a competitive analysis of cloud providers")
# Leader creates tasks, delegates to members, tracks progress,
# and synthesizes when all tasks are complete.
```

### Nested Teams

Teams can be members of other teams. A nested team is wrapped in a thin proxy that implements the Agent interface.

```python
research_team = Team(
  name="research-team",
  model="openai/gpt-4o",
  members=[web_searcher, paper_reader],
  mode=TeamMode.coordinate,
  description="A specialized research team.",  # Used as member description
)

writing_team = Team(
  name="writing-team",
  model="openai/gpt-4o",
  members=[drafter, editor],
  mode=TeamMode.coordinate,
  description="A specialized writing team.",
)

# Top-level team coordinates the sub-teams
meta_team = Team(
  name="meta-team",
  model="openai/gpt-4o",
  members=[research_team, writing_team],
  mode=TeamMode.coordinate,
  instructions="Coordinate research and writing sub-teams.",
)
```

---

## Context Sharing

By default, each member delegation is independent. Enable `share_member_interactions` to pass previous member outputs as context to subsequent delegations:

```python
team = Team(
  name="context-aware",
  model="openai/gpt-4o",
  members=[researcher, writer],
  mode=TeamMode.coordinate,
  share_member_interactions=True,  # Writer sees researcher's output in leader context
)
```

When enabled, the leader's system prompt includes a "Prior Member Interactions" section showing truncated inputs and outputs from all previous delegations in the current run.

---

## Structured Output

Force the team's final response into a Pydantic model:

```python
from pydantic import BaseModel

class Report(BaseModel):
  title: str
  summary: str
  key_findings: list[str]

team = Team(
  name="report-team",
  model="openai/gpt-4o",
  members=[researcher, writer],
  mode=TeamMode.coordinate,
  output_schema=Report,
)

result = await team.arun("Analyze cloud provider pricing")
result.parsed  # Report instance
```

You can also override per-run:
```python
result = await team.arun("Analyze pricing", output_schema=Report)
```

---

## Events

Subscribe to team lifecycle events via the `events` property.

```python
from definable.agent.team import (
  Team,
  TeamRunStartedEvent,
  TeamRunCompletedEvent,
  MemberDelegatedEvent,
  MemberCompletedEvent,
  TaskCreatedEvent,
  TaskIterationEvent,
)

team = Team(name="my-team", model="openai/gpt-4o", members=[...])

@team.events.on(MemberDelegatedEvent)
async def on_delegated(event):
  print(f"Delegated to {event.member_name}: {event.task_input[:80]}")

@team.events.on(MemberCompletedEvent)
async def on_completed(event):
  print(f"{event.member_name} completed: {(event.content or '')[:80]}")

@team.events.on(TaskIterationEvent)
async def on_iteration(event):
  print(f"Tasks mode iteration {event.iteration}: "
        f"{event.completed_count} done, {event.pending_count} pending")

result = await team.arun("Do something complex")
```

### Event Types

| Event | Emitted When | Key Fields |
|-------|-------------|------------|
| `TeamRunStartedEvent` | Team run begins | `team_id`, `team_name`, `mode`, `member_names` |
| `TeamRunCompletedEvent` | Team run finishes | `team_id`, `team_name`, `content` |
| `TeamRunErrorEvent` | Team run fails | `team_id`, `team_name`, `error` |
| `MemberDelegatedEvent` | Leader delegates to member | `member_name`, `task_input`, `mode` |
| `MemberCompletedEvent` | Member finishes | `member_name`, `content`, `metrics` |
| `MemberErrorEvent` | Member fails | `member_name`, `error` |
| `MemberRoutedEvent` | Request routed (route mode) | `member_name`, `reason` |
| `TaskCreatedEvent` | Task created (tasks mode) | `task_id`, `title`, `assignee` |
| `TaskStatusChangedEvent` | Task status changes | `task_id`, `old_status`, `new_status` |
| `TaskIterationEvent` | Each tasks-mode iteration | `iteration`, `pending_count`, `completed_count`, `failed_count` |

All events carry a `run_id` and inherit from `BaseTeamEvent`.

---

## Auto-Injected Leader Tools

The leader agent receives different tools depending on the mode:

| Tool | Modes | Purpose |
|------|-------|---------|
| `delegate_to_member` | coordinate, route, tasks | Delegate a task to a named member |
| `get_member_information` | coordinate | Get member capabilities and tools |
| `create_task` | tasks | Add a task to the shared TaskList |
| `execute_task` | tasks | Run a task via its assigned member |
| `get_task_status` | tasks | Get formatted task list summary |
| `update_task_status` | tasks | Change status, add result/notes |
| `mark_goal_complete` | tasks | Signal that the overall goal is achieved |

These tools are injected temporarily during execution and removed after the run completes.

---

## Gotchas

| Pitfall | Details |
|---------|---------|
| `Team` requires a model | Set `model=` on the Team or ensure at least one member has a model. Otherwise raises `ValueError`. |
| `arun()` returns `RunOutput` | **Not** a `TeamOutput`. It returns the same `RunOutput` as `Agent.arun()`. Access `.content`, `.messages`, `.metrics`. |
| `run()` is sync convenience | Uses `asyncio.run` or `ThreadPoolExecutor`. Prefer `arun()` for production. |
| Member naming | Members are named by their `name` attribute (or `agent_name`). If not set, auto-named `agent-0`, `agent-1`, etc. Duplicates silently overwrite. |
| `max_iterations` only for tasks mode | Ignored in coordinate, route, and collaborate modes. |
| `share_member_interactions` | Only affects coordinate mode. Previous interactions are truncated (input: 200 chars, output: 500 chars). |
| Nested Team as member | Uses the team's `description` (or `instructions`) as the member description seen by the outer leader. Set `description=` for clarity. |
| Tasks mode termination | Stops when: `mark_goal_complete` is called, `all_terminal()` is True, or `max_iterations` is reached. |
| Leader tool injection | Tools are injected temporarily and restored after the run. Do not share leader agents across concurrent team runs. |
