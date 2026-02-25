# Scheduler

> Job lifecycle management, persistence, and a tick-based scheduling loop for Definable AI agents.

The Scheduler manages scheduled jobs -- recurring or one-shot tasks that fire on trigger conditions. Jobs have a full lifecycle (pending, active, paused, completed, failed, cancelled), are persisted to pluggable stores, and execute with bounded concurrency through a central tick loop.

## Quick Start

```python
import asyncio
from definable.agent.scheduler import Scheduler, ScheduledJob, JobStatus, InMemoryJobStore
from definable.agent.trigger import Interval

async def main():
  # Create a scheduler with defaults (InMemoryJobStore, 1s tick, 10 concurrent)
  scheduler = Scheduler()

  # Add a recurring job
  job = scheduler.add(Interval(seconds=30), name="health-check")
  print(job.status)      # JobStatus.ACTIVE (auto-activated on add)
  print(job.name)        # "health-check"
  print(scheduler.job_count)  # 1

  # Lifecycle control
  scheduler.pause(job.job_id)
  print(job.status)      # JobStatus.PAUSED
  scheduler.resume(job.job_id)
  print(job.status)      # JobStatus.ACTIVE

  # Persist all jobs to the store
  await scheduler.save_all()

asyncio.run(main())
```

## Architecture

```
Scheduler (tick loop)
  |
  +-- _jobs: Dict[str, ScheduledJob]     -- in-memory job registry
  +-- _store: JobStore                    -- persistence backend
  |     +-- InMemoryJobStore              -- dict-based, ephemeral
  |     +-- SQLiteJobStore                -- aiosqlite, persistent
  |
  +-- _semaphore: Semaphore               -- concurrency limiter
  +-- _active_tasks: Dict[str, Task]      -- currently running jobs
  |
  +-- Callbacks
        +-- on_job_started(job)
        +-- on_job_completed(job)
        +-- on_job_failed(job, error)
```

### Module Structure

```
agent/scheduler/
+-- __init__.py         # Public API: Scheduler, ScheduledJob, JobStatus, stores
+-- job.py              # ScheduledJob dataclass, JobStatus enum
+-- store.py            # JobStore protocol, InMemoryJobStore, SQLiteJobStore
+-- scheduler.py        # Scheduler class (tick loop, lifecycle, concurrency)
```

### How It Connects

```
Agent
  +-- triggers: List[BaseTrigger]        -- Cron, Interval, OneShot, Webhook
  +-- scheduler: Scheduler               -- auto-created when triggers are present
        +-- ScheduledJob(trigger)         -- wraps each trigger with lifecycle state
        +-- JobStore                      -- persists job state between restarts
        +-- TriggerExecutor              -- executes trigger handlers on fire
```

## API Reference

### JobStatus

Enum representing the lifecycle state of a scheduled job.

```python
from definable.agent.scheduler import JobStatus

JobStatus.PENDING     # "pending"   -- created but not yet started
JobStatus.ACTIVE      # "active"    -- running on schedule
JobStatus.PAUSED      # "paused"    -- temporarily suspended
JobStatus.COMPLETED   # "completed" -- finished (max_runs reached or OneShot fired)
JobStatus.FAILED      # "failed"    -- last run failed
JobStatus.CANCELLED   # "cancelled" -- manually cancelled
```

### ScheduledJob

A single scheduled job wrapping a trigger with lifecycle tracking.

```python
from definable.agent.scheduler import ScheduledJob, JobStatus
from definable.agent.trigger import Interval

job = ScheduledJob(
  trigger=Interval(seconds=60),  # Required -- the trigger that fires this job
  job_id="auto-generated-uuid",  # Auto-generated if omitted
  name="",                       # Human-readable name (defaults to trigger.name)
  status=JobStatus.PENDING,      # Initial status
  max_runs=None,                 # None = unlimited executions
  metadata={},                   # Arbitrary user metadata
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `trigger` | `BaseTrigger` | *required* | The trigger that fires this job |
| `job_id` | `str` | auto UUID | Unique identifier |
| `name` | `str` | `trigger.name` | Human-readable label |
| `status` | `JobStatus` | `PENDING` | Current lifecycle state |
| `max_runs` | `int \| None` | `None` | Max executions (None = unlimited) |
| `metadata` | `dict` | `{}` | Arbitrary user metadata |
| `created_at` | `float` | `time()` | Creation timestamp |
| `next_run_at` | `float` | computed | Next scheduled fire time |
| `last_run_at` | `float` | `0.0` | Last execution timestamp |
| `run_count` | `int` | `0` | Total executions |
| `failure_count` | `int` | `0` | Failed executions |
| `last_error` | `str \| None` | `None` | Most recent error message |

**Properties:**

| Property | Return | Description |
|----------|--------|-------------|
| `is_runnable` | `bool` | True if PENDING/ACTIVE and under `max_runs` |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `activate` | `job.activate()` | Set status to ACTIVE |
| `pause` | `job.pause()` | Set status to PAUSED |
| `resume` | `job.resume()` | Resume from PAUSED to ACTIVE |
| `cancel` | `job.cancel()` | Set status to CANCELLED |
| `record_run` | `job.record_run()` | Record a successful run (increments count, updates timing) |
| `record_failure` | `job.record_failure(error)` | Record a failed run (increments count + failure count) |
| `to_dict` | `job.to_dict() -> dict` | Serialize to plain dict for storage |

```python
from definable.agent.scheduler import ScheduledJob, JobStatus
from definable.agent.trigger import Interval

job = ScheduledJob(trigger=Interval(seconds=60), name="heartbeat")
print(job.status)   # JobStatus.PENDING
print(job.is_runnable)  # True

job.activate()
print(job.status)   # JobStatus.ACTIVE

job.record_run()
print(job.run_count)  # 1

job.pause()
print(job.status)   # JobStatus.PAUSED
print(job.is_runnable)  # False

job.resume()
print(job.status)   # JobStatus.ACTIVE

# Max runs auto-completes the job
limited = ScheduledJob(trigger=Interval(seconds=10), max_runs=1)
limited.activate()
limited.record_run()
print(limited.status)  # JobStatus.COMPLETED
```

### JobStore (Protocol)

Async protocol for job persistence backends.

```python
from definable.agent.scheduler import JobStore

class JobStore(Protocol):
  async def save(self, job: ScheduledJob) -> None: ...
  async def get(self, job_id: str) -> ScheduledJob | None: ...
  async def list_jobs(self, status: JobStatus | None = None) -> list[ScheduledJob]: ...
  async def delete(self, job_id: str) -> bool: ...
```

### InMemoryJobStore

Dict-based ephemeral store. Best for testing and short-lived processes.

```python
import asyncio
from definable.agent.scheduler import InMemoryJobStore, ScheduledJob, JobStatus
from definable.agent.trigger import Interval

async def main():
  store = InMemoryJobStore()

  job = ScheduledJob(trigger=Interval(seconds=30), name="check")
  await store.save(job)

  loaded = await store.get(job.job_id)
  print(loaded.name)  # "check"

  all_jobs = await store.list_jobs()
  print(len(all_jobs))  # 1

  count = await store.count()
  print(count)  # 1

  active_count = await store.count(status=JobStatus.ACTIVE)
  print(active_count)  # 0 (job is PENDING)

  deleted = await store.delete(job.job_id)
  print(deleted)  # True

asyncio.run(main())
```

### SQLiteJobStore

Persistent store backed by aiosqlite. Auto-creates the `scheduled_jobs` table on first use.

```python
import asyncio
from definable.agent.scheduler import SQLiteJobStore, ScheduledJob
from definable.agent.trigger import Interval

async def main():
  store = SQLiteJobStore(".definable/scheduler.db")  # default path
  await store.initialize()  # creates table if needed

  job = ScheduledJob(trigger=Interval(seconds=60), name="persist-me")
  await store.save(job)

  # List raw rows for reconstruction with triggers
  rows = await store.list_rows()
  print(rows[0]["trigger_name"])  # "interval(60s)"

  await store.close()

asyncio.run(main())
```

> **Requires:** `pip install aiosqlite`

> **Important:** SQLiteJobStore stores metadata but NOT trigger objects. Triggers must be re-attached on load via the row's `trigger_name` field.

### Scheduler

The central scheduling loop. Manages job registration, lifecycle control, and concurrent execution.

```python
from definable.agent.scheduler import Scheduler

scheduler = Scheduler(
  store=None,            # JobStore backend. None -> InMemoryJobStore
  tick_interval=1.0,     # Seconds between due-job checks (default 1.0)
  max_concurrent=10,     # Max simultaneous job executions (default 10)
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `scheduler.add(trigger, name="", max_runs=None, metadata=None) -> ScheduledJob` | Create and register a new job (auto-activated) |
| `add_job` | `scheduler.add_job(job) -> ScheduledJob` | Register an existing ScheduledJob |
| `get` | `scheduler.get(job_id) -> ScheduledJob \| None` | Get a job by ID |
| `list_jobs` | `scheduler.list_jobs(status=None) -> list[ScheduledJob]` | List jobs, optionally filtered by status |
| `pause` | `scheduler.pause(job_id) -> bool` | Pause a job |
| `resume` | `scheduler.resume(job_id) -> bool` | Resume a paused job |
| `cancel` | `scheduler.cancel(job_id) -> bool` | Cancel a job (also cancels running task) |
| `remove` | `scheduler.remove(job_id) -> bool` | Cancel and remove a job entirely |
| `start` | `await scheduler.start(executor)` | Start the tick loop (blocks until stopped) |
| `stop` | `scheduler.stop()` | Signal the scheduler to stop |
| `save_all` | `await scheduler.save_all()` | Persist all jobs to the store |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | Whether the tick loop is active |
| `job_count` | `int` | Number of registered jobs |

**Callbacks:**

| Callback | Signature | Description |
|----------|-----------|-------------|
| `on_job_started` | `(job: ScheduledJob) -> None` | Called when a job begins execution |
| `on_job_completed` | `(job: ScheduledJob) -> None` | Called after a successful run |
| `on_job_failed` | `(job: ScheduledJob, error: str) -> None` | Called after a failed run |

## Patterns

### Recurring Health Check

```python
from definable.agent.scheduler import Scheduler
from definable.agent.trigger import Interval

scheduler = Scheduler(max_concurrent=5)

job = scheduler.add(
  Interval(seconds=30),
  name="health-check",
  metadata={"endpoint": "/health"},
)

# Attach callbacks
scheduler.on_job_started = lambda j: print(f"Started: {j.name}")
scheduler.on_job_completed = lambda j: print(f"Completed: {j.name} (runs={j.run_count})")
scheduler.on_job_failed = lambda j, e: print(f"Failed: {j.name}: {e}")
```

### Job with Run Limit

```python
from definable.agent.scheduler import Scheduler
from definable.agent.trigger import Interval

scheduler = Scheduler()

# Run exactly 5 times, then auto-complete
job = scheduler.add(
  Interval(seconds=10),
  name="limited-task",
  max_runs=5,
)
```

### Persistent Scheduler with SQLite

```python
from definable.agent.scheduler import Scheduler, SQLiteJobStore
from definable.agent.trigger import Interval

store = SQLiteJobStore(".definable/scheduler.db")
scheduler = Scheduler(store=store)

job = scheduler.add(Interval(seconds=60), name="persist-check")

# Persist current state
await scheduler.save_all()
```

### Agent Integration

The Agent auto-creates a Scheduler when triggers are present:

```python
from definable.agent import Agent
from definable.agent.trigger import Interval

agent = Agent(
  model="openai/gpt-4o-mini",
  triggers=[Interval(seconds=300)],
)

# Access the scheduler
print(agent.scheduler.job_count)  # 1
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `scheduler.add()` auto-activates jobs | Jobs are set to ACTIVE immediately; no separate `.activate()` needed |
| `SQLiteJobStore` doesn't serialize triggers | Triggers must be re-attached when restoring from DB; use `trigger_name` from `list_rows()` |
| `SQLiteJobStore` requires `aiosqlite` | `pip install aiosqlite` |
| `start()` blocks until `stop()` | Run in a background task: `asyncio.create_task(scheduler.start(executor))` |
| `max_runs=None` means unlimited | Jobs run forever until explicitly paused/cancelled |
| `record_run()` auto-completes when `max_runs` reached | Status transitions to COMPLETED automatically |
| `resume()` only works from PAUSED | Calling resume on an ACTIVE or COMPLETED job is a no-op |

## Related Modules

- **[Trigger](../trigger/README.md)** -- BaseTrigger, Cron, Interval, OneShot, Webhook triggers
- **[Runtime](../runtime/README.md)** -- AgentRuntime integrates the Scheduler into the agent lifecycle
- **[Agent](../README.md)** -- Agent auto-creates a Scheduler when triggers are configured
