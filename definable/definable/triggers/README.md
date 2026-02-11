# triggers

Event triggers for webhooks, cron jobs, and programmatic events.

## Installation

Cron triggers require an optional dependency:

```bash
pip install 'definable[cron]'  # croniter
```

## Quick Start

```python
from definable.agents import Agent
from definable.triggers import Webhook, Cron, EventTrigger

agent = Agent(model=model, tools=[...])

# Webhook trigger
@agent.on(Webhook("/github", method="POST"))
async def handle_github(event):
  return f"Process this GitHub event: {event.body}"

# Cron trigger (every hour)
@agent.on(Cron("0 * * * *"))
async def hourly_check(event):
  return "Run the hourly health check."

# Programmatic event trigger
@agent.on(EventTrigger("user_signup"))
async def on_signup(event):
  return f"Welcome new user: {event.body}"

# Fire an event
agent.emit("user_signup", {"name": "Alice"})

# Start the runtime
agent.serve(port=8000)
```

## Module Structure

```
triggers/
├── __init__.py      # Public API (Cron lazy-loaded)
├── base.py          # BaseTrigger ABC, TriggerEvent, TriggerResult
├── webhook.py       # Webhook trigger
├── cron.py          # Cron trigger (requires croniter)
├── event.py         # EventTrigger (programmatic)
└── executor.py      # TriggerExecutor — runs trigger handlers
```

## API Reference

### TriggerEvent

```python
from definable.triggers import TriggerEvent
```

The event object passed to trigger handlers.

| Field | Type | Description |
|-------|------|-------------|
| `body` | `Optional[Dict]` | Parsed request body or event data |
| `headers` | `Optional[Dict]` | HTTP headers (webhooks only) |
| `source` | `str` | Human-readable trigger identifier |
| `timestamp` | `float` | Unix timestamp |
| `raw` | `Any` | Raw request object |

### BaseTrigger

```python
from definable.triggers import BaseTrigger
```

Abstract base class for all triggers. Subclasses have a `handler` callback and an optional `auth` override.

### Webhook

```python
from definable.triggers import Webhook
```

HTTP webhook trigger. Registered as a route on the `AgentServer`.

```python
Webhook(
  path="/my-endpoint",   # URL path (auto-prepends /)
  method="POST",         # HTTP method
  auth=None,             # None=inherit, False=disable, AuthProvider=override
)
```

### Cron

```python
from definable.triggers import Cron
```

Scheduled trigger using cron expressions. Requires `croniter`.

```python
Cron(
  schedule="*/5 * * * *",  # Every 5 minutes
  timezone="UTC",          # IANA timezone
)
```

### EventTrigger

```python
from definable.triggers import EventTrigger
```

Programmatic event trigger. Fired via `agent.emit(event_name, data)`.

```python
EventTrigger(event_name="user_signup")
```

Events are dispatched as fire-and-forget tasks.

### TriggerExecutor

```python
from definable.triggers import TriggerExecutor
```

Executes trigger handlers and processes return values:

| Return Value | Behavior |
|-------------|----------|
| `None` | No-op |
| `str` | Passed to `agent.arun(str)` |
| `dict` | Passed to `agent.arun(**dict)` |
| `awaitable` | Awaited and result processed recursively |

## See Also

- `agents/` — `agent.on(trigger)` registers triggers, `agent.emit()` fires events
- `runtime/` — `AgentRuntime` manages webhook routes and cron scheduling
- `auth/` — Per-trigger auth overrides via `Webhook(auth=...)`
