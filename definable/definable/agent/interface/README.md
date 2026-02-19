# interfaces

Connect agents to messaging platforms — Telegram, Discord, and Signal.

## Installation

Platform-specific dependencies:

```bash
pip install 'definable[discord]'   # Discord (discord.py)
pip install 'definable[telegram]'  # Telegram (httpx, included by default)
# Signal uses signal-cli-rest-api via Docker — no extra pip install needed
```

## Quick Start

```python
from definable.agent import Agent
from definable.agent.interface import TelegramInterface, TelegramConfig

agent = Agent(model=model, instructions="You are a helpful assistant.")

telegram = TelegramInterface(
  config=TelegramConfig(bot_token="YOUR_BOT_TOKEN"),
)

agent.serve(telegram)
```

## Module Structure

```
interfaces/
├── __init__.py      # Public API (platform impls lazy-loaded)
├── base.py          # BaseInterface ABC — message pipeline
├── config.py        # InterfaceConfig base dataclass
├── message.py       # InterfaceMessage, InterfaceResponse
├── session.py       # InterfaceSession, SessionManager
├── hooks.py         # InterfaceHook protocol, LoggingHook, AllowlistHook
├── identity.py      # IdentityResolver, SQLiteIdentityResolver, PlatformIdentity
├── errors.py        # InterfaceError hierarchy
├── serve.py         # serve() multi-interface supervisor
├── telegram/
│   ├── config.py    # TelegramConfig (polling/webhook modes)
│   └── interface.py # TelegramInterface
├── discord/
│   ├── config.py    # DiscordConfig
│   └── interface.py # DiscordInterface
└── signal/
    ├── config.py    # SignalConfig (optional Docker management)
    └── interface.py # SignalInterface
```

## API Reference

### BaseInterface

Abstract base class for all platform interfaces.

```python
from definable.agent.interface import BaseInterface

class MyInterface(BaseInterface):
  async def _start_receiver(self) -> None: ...
  async def _stop_receiver(self) -> None: ...
  async def _convert_inbound(self, raw_message) -> Optional[InterfaceMessage]: ...
  async def _send_response(self, original_msg, response, raw_message) -> None: ...
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `bind(agent)` | Bind an agent to this interface |
| `add_hook(hook)` | Add a message pipeline hook |
| `start()` / `stop()` | Lifecycle management |
| `serve_forever()` | Run until cancelled |
| `handle_platform_message(raw)` | Process an inbound message through the pipeline |

### Platform Implementations

| Class | Config | Transport | Key Features |
|-------|--------|-----------|--------------|
| `TelegramInterface` | `TelegramConfig` | httpx | Polling (dev) and webhook (prod) modes, media extraction |
| `DiscordInterface` | `DiscordConfig` | discord.py | Gateway connection, command prefix, auto message splitting |
| `SignalInterface` | `SignalConfig` | httpx | REST API polling, optional Docker container management |

### InterfaceMessage / InterfaceResponse

```python
from definable.agent.interface import InterfaceMessage, InterfaceResponse
```

`InterfaceMessage` — Normalized inbound message with platform metadata and media attachments (images, audio, videos, files).

`InterfaceResponse` — Outbound response with content and optional media.

### SessionManager

```python
from definable.agent.interface import SessionManager
```

Thread-safe session management with TTL-based expiry.

| Method | Description |
|--------|-------------|
| `get_or_create(platform, user_id, chat_id)` | Get or create a session |
| `get(platform, user_id, chat_id)` | Get existing session or None |
| `remove(platform, user_id, chat_id)` | Remove a session |
| `cleanup_expired()` | Remove expired sessions |

### Hooks

```python
from definable.agent.interface import InterfaceHook, LoggingHook, AllowlistHook
```

`InterfaceHook` — Protocol with optional lifecycle methods:

| Method | Description |
|--------|-------------|
| `on_message_received(message)` | Called on inbound message; return `False` to reject |
| `on_before_respond(message, session)` | Called before agent execution; can modify message |
| `on_after_respond(message, response, session)` | Called after agent execution; can modify response |
| `on_error(error, message)` | Called on pipeline errors |

Built-in hooks:
- `LoggingHook` — Logs messages and errors
- `AllowlistHook(allowed_user_ids)` — Restricts access to a set of user IDs

### Identity Resolution

```python
from definable.agent.interface import IdentityResolver, SQLiteIdentityResolver
```

- `IdentityResolver` — Protocol for cross-platform user identity mapping.
- `SQLiteIdentityResolver(db_path)` — SQLite-backed implementation with `resolve()`, `link()`, `unlink()`, `get_identities()`.

### Errors

```python
from definable.agent.interface import (
  InterfaceError,              # Base (500)
  InterfaceConnectionError,    # 503
  InterfaceAuthenticationError,# 401
  InterfaceRateLimitError,     # 429, has retry_after
  InterfaceMessageError,       # 400
)
```

## Usage with Agent

```python
from definable.agent import Agent
from definable.agent.interface import (
  TelegramInterface, TelegramConfig,
  DiscordInterface, DiscordConfig,
  LoggingHook, AllowlistHook,
)

agent = Agent(model=model)

telegram = TelegramInterface(
  config=TelegramConfig(bot_token="..."),
)
telegram.add_hook(LoggingHook())
telegram.add_hook(AllowlistHook(allowed_user_ids={"123456"}))

discord = DiscordInterface(
  config=DiscordConfig(bot_token="..."),
)

# Serve multiple interfaces
agent.serve(telegram, discord)
```

## See Also

- `agents/` — Agent class with `serve()` / `aserve()` methods
- `runtime/` — AgentRuntime orchestrator
- `auth/` — HTTP authentication (for webhook mode)
- `utils/supervisor.py` — Interface auto-restart with backoff
