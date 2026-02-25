# triggers

Event triggers for webhooks, cron jobs, interval timers, one-shot delays, and programmatic events.

## Installation

Most triggers require no extra dependencies. Cron triggers require one optional package:

```bash
pip install 'definable[cron]'  # croniter — required for Cron only
```

## Quick Start

```python
from definable.agent import Agent
from definable.agent.trigger import Webhook, Cron, EventTrigger, Interval, OneShot
from definable.model.openai import OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), tools=[...])

# HTTP webhook
@agent.on(Webhook("/github", method="POST"))
async def handle_github(event):
  return f"Process this GitHub event: {event.body}"

# Cron — every hour (requires croniter)
@agent.on(Cron("0 * * * *"))
async def hourly_check(event):
  return "Run the hourly health check."

# Fixed interval — every 30 seconds
@agent.on(Interval(seconds=30))
async def poll(event):
  return "Poll the external service."

# One-shot — fire once after 60 seconds
@agent.on(OneShot(delay=60))
async def delayed_task(event):
  return "Run the deferred task now."

# Programmatic event
@agent.on(EventTrigger("user_signup"))
async def on_signup(event):
  return f"Welcome new user: {event.body}"

# Fire a programmatic event
agent.emit("user_signup", {"name": "Alice"})

# Start the runtime
agent.serve(port=8000)
```

## Module Structure

```
trigger/
├── __init__.py      # Public API (Cron, Interval, OneShot lazy-loaded)
├── base.py          # BaseTrigger ABC, TriggerEvent, TriggerResult
├── webhook.py       # Webhook
├── cron.py          # Cron (requires croniter)
├── event.py         # EventTrigger
├── executor.py      # TriggerExecutor — runs trigger handlers
├── interval.py      # Interval (no external deps)
└── oneshot.py       # OneShot (no external deps)
```

All public types are importable from the top-level package:

```python
from definable.agent.trigger import BaseTrigger, TriggerEvent, TriggerResult
from definable.agent.trigger import Webhook, Cron, EventTrigger, TriggerExecutor
from definable.agent.trigger import Interval, OneShot
```

`Cron`, `Interval`, and `OneShot` are lazy-loaded via `__getattr__` — they are not imported until first accessed, so missing optional dependencies do not cause import errors at module load time.

## API Reference

### TriggerEvent

```python
from definable.agent.trigger import TriggerEvent
```

Passed to every trigger handler as the single argument.

| Field | Type | Description |
|-------|------|-------------|
| `body` | `Optional[Dict]` | Parsed request body or event data |
| `headers` | `Optional[Dict]` | HTTP headers (webhooks only; `None` for non-HTTP triggers) |
| `source` | `str` | Human-readable trigger identifier (e.g. `"POST /github"`) |
| `timestamp` | `float` | Unix timestamp when the event was created |
| `raw` | `object` | Raw request object (framework-specific; `None` for timer triggers) |

### BaseTrigger

```python
from definable.agent.trigger import BaseTrigger
```

Abstract base class for all triggers.

| Member | Kind | Description |
|--------|------|-------------|
| `name` | abstract property | Human-readable identifier (e.g. `"interval(60s)"`) |
| `next_run(base_time)` | method | Returns next fire time as Unix timestamp; default returns `base_time` (fire immediately) |
| `handler` | attribute | Callable registered via `@agent.on(...)` |
| `agent` | attribute | The `Agent` instance this trigger is bound to |
| `auth` | attribute | Per-trigger auth override (`None` = inherit from agent) |

Time-based triggers (`Cron`, `Interval`, `OneShot`) override `next_run()`. Non-time triggers (`Webhook`, `EventTrigger`) use the default, which fires immediately on each invocation.

### Webhook

```python
from definable.agent.trigger import Webhook
```

HTTP webhook trigger. Registered as a route on the `AgentServer`.

```python
Webhook(
  path="/my-endpoint",   # URL path (leading / auto-prepended if absent)
  method="POST",         # HTTP method
  auth=None,             # None = inherit from agent, False = disable, AuthProvider = override
)
```

The trigger `name` is the method + path: `"POST /my-endpoint"`.

```python
wh = Webhook(path="/github", method="POST")
wh.name  # "POST /github"
```

### Cron

```python
from definable.agent.trigger import Cron  # requires: pip install 'definable[cron]'
```

Scheduled trigger using standard 5-field cron expressions. Requires `croniter`.

```python
Cron(
  schedule="*/5 * * * *",  # Every 5 minutes
  timezone="UTC",           # IANA timezone string
)
```

`next_run()` delegates to `croniter` and returns the next scheduled Unix timestamp after `base_time`. Importing `Cron` without `croniter` installed raises `ImportError`.

### Interval

```python
from definable.agent.trigger import Interval
```

Fires at a fixed cadence. No external dependencies required.

```python
Interval(
  seconds=60,  # Interval between executions in seconds (must be > 0)
)
```

```python
iv = Interval(seconds=60)
iv.name              # "interval(60s)"
iv.seconds           # 60
iv.next_run(1000.0)  # 1060.0

Interval(seconds=-1)  # raises ValueError
Interval(seconds=0)   # raises ValueError
```

`next_run(base_time)` returns `base_time + seconds`. The Scheduler calls this after each execution to compute the next fire time.

### OneShot

```python
from definable.agent.trigger import OneShot
```

Fires exactly once, then becomes inert. No external dependencies required.

Specify a relative delay or an absolute fire time — exactly one must be positive:

```python
# Fire in 60 seconds from now
trigger = OneShot(delay=60)

# Fire at a specific Unix timestamp
trigger = OneShot(fire_at=1700000000.0)

# Neither provided — raises ValueError
OneShot()            # ValueError: requires 'delay' > 0 or 'fire_at' > 0
OneShot(delay=0)     # ValueError
```

```python
os_t = OneShot(delay=60)
os_t.name       # "oneshot(at=<timestamp>)"
os_t.fire_at    # absolute Unix timestamp
os_t.fired      # False

os_t.mark_fired()
os_t.fired      # True
os_t.next_run(0)  # math.inf — Scheduler sees this as "never run again"
```

`next_run(base_time)` returns `fire_at` if not yet fired, or `math.inf` after `mark_fired()` is called.

### EventTrigger

```python
from definable.agent.trigger import EventTrigger
```

Programmatic trigger. Fired via `agent.emit(event_name, data)`.

```python
EventTrigger(event_name="user_signup")
# name → "event:user_signup"
```

Events are dispatched as fire-and-forget async tasks. Multiple handlers can listen to the same event name.

### TriggerExecutor

```python
from definable.agent.trigger import TriggerExecutor
```

Runs a trigger handler and processes its return value to drive the agent:

| Return value | Behavior |
|-------------|----------|
| `None` | No-op — handler ran but produced no agent input |
| `str` | Passed to `agent.arun(str)` |
| `dict` | Passed to `agent.arun(**dict)` |
| `awaitable` | Awaited; result processed recursively by the same rules |

## Scheduler Integration

`Interval` and `OneShot` integrate with `definable.agent.scheduler.Scheduler` for time-based job lifecycle management.

```python
from definable.agent.scheduler import Scheduler, ScheduledJob, InMemoryJobStore
from definable.agent.trigger import Interval, OneShot

scheduler = Scheduler(store=InMemoryJobStore())

# Recurring job — fires every 5 minutes
job = ScheduledJob(
  trigger=Interval(seconds=300),
  name="health-check",
  max_runs=100,          # stop after 100 executions
)
await scheduler.add_job(job)
await scheduler.start()

# One-shot job — fires once after a 10-second delay then stops
job = ScheduledJob(
  trigger=OneShot(delay=10),
  name="startup-task",
)
await scheduler.add_job(job)
```

The `Scheduler` tick loop:

1. Reads `job.next_run_at` to find due jobs (set from `trigger.next_run()` on init).
2. Fires due jobs up to the configured concurrency limit.
3. After execution, calls `job.record_run()`, which calls `trigger.next_run(last_run_at)` to advance the schedule.
4. For `OneShot`, `next_run()` returns `math.inf` after `mark_fired()`, causing the Scheduler to mark the job `COMPLETED`.

`Agent` exposes a `scheduler` property that auto-detects attached triggers:

```python
from definable.agent import Agent
from definable.agent.trigger import Interval

agent = Agent(
  model="openai/gpt-4o-mini",
  instructions="...",
)

@agent.on(Interval(seconds=60))
async def periodic(event):
  return "Run periodic task."

# agent.scheduler is auto-initialized when schedulable triggers are present
await agent.scheduler.start()
```

### JobStatus lifecycle

```
PENDING → ACTIVE → COMPLETED   (max_runs reached or OneShot fired)
                 ↓
               PAUSED ↔ ACTIVE
                 ↓
             CANCELLED
                 ↓
              FAILED
```

## Gotchas

| Trap | Reality |
|------|---------|
| `Cron` import without croniter | Raises `ImportError` at import time (lazy-loaded, not at module load) |
| `Interval(seconds=0)` | Raises `ValueError` — must be strictly positive |
| `OneShot()` with no args | Raises `ValueError` — `delay` or `fire_at` must be > 0 |
| `OneShot` after `mark_fired()` | `next_run()` returns `math.inf` — Scheduler stops scheduling it |
| `next_run()` on `Webhook`/`EventTrigger` | Returns `base_time` (fire immediately) — these are not time-scheduled |
| `headers` on non-HTTP triggers | Always `None` for `Cron`, `Interval`, `OneShot`, `EventTrigger` |
| `raw` on timer triggers | Always `None` — no underlying request object exists |
| `Interval` name format | `"interval(60s)"` uses the `float` value: `Interval(seconds=60.0).name == "interval(60.0s)"` |

## See Also

- `agent/` — `agent.on(trigger)` registers triggers, `agent.emit()` fires events
- `agent/scheduler/` — `Scheduler`, `ScheduledJob`, `JobStore`, `InMemoryJobStore`, `SQLiteJobStore`
- `agent/runtime/` — `AgentRuntime` manages webhook routes and the scheduler loop
- `agent/auth/` — Per-trigger auth overrides via `Webhook(auth=...)`
