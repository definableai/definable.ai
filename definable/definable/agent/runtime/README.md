# runtime

Agent server runtime — HTTP server, interface supervisor, and cron scheduler.

## Installation

```bash
pip install 'definable[serve]'     # FastAPI + uvicorn
pip install 'definable[runtime]'   # All runtime dependencies
```

## Quick Start

```python
from definable.agent import Agent

agent = Agent(model=model, tools=[...])

# Sync (blocking)
agent.serve(host="0.0.0.0", port=8000)

# Async
await agent.aserve(host="0.0.0.0", port=8000)
```

With interfaces and dev mode:

```python
from definable.agent.interface import TelegramInterface, TelegramConfig

telegram = TelegramInterface(config=TelegramConfig(bot_token="..."))

agent.serve(telegram, port=8000, dev=True)
```

## Module Structure

```
runtime/
├── __init__.py    # Exports AgentRuntime, AgentServer
├── runner.py      # AgentRuntime — orchestrates server, interfaces, cron
├── server.py      # AgentServer — FastAPI application
└── _dev.py        # Dev mode with hot-reload (watchfiles)
```

## API Reference

### AgentRuntime

```python
from definable.agent.runtime import AgentRuntime
```

The main orchestrator that runs the HTTP server, messaging interfaces, and cron triggers concurrently in a single event loop.

```python
runtime = AgentRuntime(
  agent,
  interfaces=[telegram, discord],
  host="0.0.0.0",
  port=8000,
  enable_server=True,    # None = auto-detect (True if webhooks registered)
  name="my-agent",
  dev=False,             # Enable hot-reload
)
await runtime.start()
```

Features:
- Concurrent execution of HTTP server, interfaces, and cron jobs
- Graceful SIGINT/SIGTERM shutdown
- Auto-detects webhook triggers and enables server if needed
- Interface supervision with auto-restart and exponential backoff

### AgentServer

```python
from definable.agent.runtime import AgentServer
```

FastAPI application for the agent HTTP API.

```python
server = AgentServer(agent, host="0.0.0.0", port=8000, dev=False)
app = server.create_app()  # Returns FastAPI instance
```

**Endpoints:**

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Health check |
| `/run` | POST | Invoke agent: `{"input": "...", "session_id": "...", "user_id": "..."}` |
| `/docs` | GET | OpenAPI docs (dev mode only) |
| Webhook paths | POST | Auto-registered from agent's webhook triggers |

Auth middleware is applied when `agent.auth` is set.

### Dev Mode

Dev mode uses `watchfiles` to watch for `.py` file changes and auto-restart the server process. Enabled via `dev=True` on `agent.serve()` or `AgentRuntime`.

## See Also

- `agents/` — `agent.serve()` and `agent.aserve()` create an `AgentRuntime` internally
- `auth/` — HTTP authentication middleware
- `triggers/` — Webhook and Cron triggers
- `interfaces/` — Messaging platform interfaces
