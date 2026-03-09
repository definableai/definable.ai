# interfaces

Connect agents to messaging platforms, voice calls, and custom communication channels.

## Installation

Platform-specific dependencies:

```bash
pip install 'definable[discord]'   # Discord (discord.py)
pip install 'definable[slack]'     # Slack (slack-bolt, slack-sdk)
pip install 'definable[cli]'       # CLI TUI mode (Textual)
# Telegram, WebSocket, WhatsApp, Email — no extra extras required
```

## Quick Start

```python
from definable.agent import Agent
from definable.agent.interface import TelegramInterface

agent = Agent(model="openai/gpt-4o-mini", instructions="You are a helpful assistant.")

telegram = TelegramInterface(bot_token="YOUR_BOT_TOKEN")
agent.serve(telegram)
```

Using `async with` for full lifecycle control:

```python
from definable.agent.interface import TelegramInterface

interface = TelegramInterface(agent=agent, bot_token="YOUR_BOT_TOKEN")
async with interface:
  await interface.serve_forever()
```

## Module Structure

```
interface/
├── __init__.py      # Public API (platform impls lazy-loaded)
├── base.py          # BaseInterface ABC
├── config.py        # InterfaceConfig base
├── message.py       # InterfaceMessage, InterfaceResponse
├── session.py       # InterfaceSession, SessionManager
├── hooks.py         # InterfaceHook, LoggingHook, AllowlistHook
├── identity.py      # IdentityResolver, SQLiteIdentityResolver, PlatformIdentity
├── gateway.py       # InterfaceGateway, InterfaceStatus, lifecycle events
├── errors.py        # Error hierarchy
├── serve.py         # serve() supervisor
├── telegram/        # TelegramInterface
├── discord/         # DiscordInterface
├── desktop/         # DesktopInterface
├── cli/             # CLIInterface + TUI
├── call/            # CallInterface (Twilio ConversationRelay)
├── slack/           # SlackInterface
├── websocket/       # WebSocketInterface
├── whatsapp/        # WhatsAppInterface
└── email/           # EmailInterface
```

## Platform Implementations

| Class | Transport | Key Features |
|-------|-----------|---|
| `TelegramInterface` | httpx | Polling (dev) + webhook (prod), media extraction |
| `DiscordInterface` | discord.py | Gateway, command prefix, auto message splitting |
| `DesktopInterface` | websockets | Local WebSocket chat for macOS |
| `CLIInterface` | stdin/stdout | Rich REPL + event visualization, TUI mode with Textual |
| `CallInterface` | WebSocket/Twilio | Voice calls via ConversationRelay |
| `SlackInterface` | Slack SDK | Socket Mode + Events API |
| `WebSocketInterface` | websockets | Generic WebSocket server, JSON wire protocol |
| `WhatsAppInterface` | httpx | WhatsApp Business API via Twilio |
| `EmailInterface` | imaplib/smtplib | IMAP polling + SMTP sending, thread tracking |

All platform implementations are lazy-loaded — importing from `definable.agent.interface`
does not require their platform dependencies to be installed.

## API Reference

### BaseInterface

Abstract base class for all platform interfaces.

```python
from definable.agent.interface import BaseInterface, InterfaceMessage, InterfaceResponse


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
| `add_hook(hook)` | Add a message pipeline hook (returns self) |
| `start()` / `stop()` | Lifecycle management |
| `serve_forever()` | Run until cancelled |
| `handle_platform_message(raw)` | Process an inbound message through the pipeline |

### InterfaceMessage

Normalized inbound message from any platform.

```python
from definable.agent.interface import InterfaceMessage

msg = InterfaceMessage(
  platform="telegram",
  platform_user_id="u123",
  platform_chat_id="c456",
  platform_message_id="m789",
  text="Hello world",
)
```

**Required fields:** `platform`, `platform_user_id`, `platform_chat_id`, `platform_message_id`

**Optional fields:** `text`, `username`, `images`, `audio`, `videos`, `files`,
`reply_to_message_id`, `metadata`, `created_at`

Note: the text field is named `text`, not `content`.

### InterfaceResponse

Outbound response sent back to the platform.

```python
from definable.agent.interface import InterfaceResponse

resp = InterfaceResponse(content="Hello back!")
# Optional: images, videos, audio, files, metadata
```

### SessionManager

Thread-safe session management with TTL-based expiry.

```python
from definable.agent.interface import SessionManager

manager = SessionManager(session_ttl_seconds=3600)
```

| Method | Description |
|--------|-------------|
| `get_or_create(platform, user_id, chat_id)` | Get or create a session |
| `get(platform, user_id, chat_id)` | Get existing session or None |
| `remove(platform, user_id, chat_id)` | Remove a session |
| `cleanup_expired()` | Remove expired sessions, returns count removed |
| `active_session_count` | Property: number of non-expired sessions |

`InterfaceSession` carries `session_id`, `messages` (conversation history), `session_state`
(arbitrary dict), `last_run_output`, and timestamps. The `truncate_history(max_messages)` method
is tool-call-aware — it never splits an assistant message with tool calls from its tool results.

### Hooks

```python
from definable.agent.interface import InterfaceHook, LoggingHook, AllowlistHook

interface.add_hook(LoggingHook())
interface.add_hook(AllowlistHook(allowed_user_ids={"123456", "789012"}))
```

`InterfaceHook` is a `Protocol` — implement only the methods you need:

| Method | Signature | Description |
|--------|-----------|-------------|
| `on_message_received` | `(message) -> Optional[bool]` | Called on inbound message; return `False` to reject |
| `on_before_respond` | `(message, session) -> Optional[InterfaceMessage]` | Called before agent execution; return modified message |
| `on_after_respond` | `(message, response, session) -> Optional[InterfaceResponse]` | Called after agent execution; return modified response |
| `on_error` | `(error, message) -> None` | Called on pipeline errors |

Built-in hooks:
- `LoggingHook` — logs received messages and errors via `log_info` / `log_error`
- `AllowlistHook(allowed_user_ids={"u1"})` — silently drops messages from unlisted users

Custom hook example:

```python
class RateLimitHook:
  async def on_message_received(self, message):
    if self._is_over_limit(message.platform_user_id):
      return False  # veto the message
    return None  # pass through

  async def on_after_respond(self, message, response, session):
    response.content = response.content[:500]  # truncate long responses
    return response
```

### InterfaceGateway

Central coordinator for running multiple interfaces together. Provides shared hooks,
per-interface status tracking, lifecycle events, optional shared sessions, and
cross-platform identity linking.

```python
from definable.agent.interface import (
  InterfaceGateway,
  TelegramInterface,
  DiscordInterface,
  SlackInterface,
  LoggingHook,
  SQLiteIdentityResolver,
)

resolver = SQLiteIdentityResolver("./identity.db")

gateway = InterfaceGateway(
  agent,
  shared_sessions=True,
  identity_resolver=resolver,
  enable_identity_linking=True,  # enables /link command flow
)

gateway.add(TelegramInterface(bot_token="..."))
gateway.add(DiscordInterface(bot_token="..."))
gateway.add(SlackInterface(bot_token="...", app_token="..."))
gateway.add_hook(LoggingHook())

gateway.serve()  # sync; use await gateway.aserve() in async context
```

**Gateway constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent` | required | Agent instance |
| `shared_sessions` | `False` | Share one SessionManager across all interfaces |
| `identity_resolver` | `None` | Resolver for cross-platform user identity |
| `session_ttl_seconds` | `3600` | TTL for the shared SessionManager |
| `hooks` | `None` | Initial gateway-level hook list |
| `enable_identity_linking` | `False` | Enable self-service `/link` command flow |
| `link_command` | `"/link"` | Command prefix for identity linking |
| `link_code_ttl` | `300` | Link code validity in seconds |

**Gateway methods:**

| Method | Description |
|--------|-------------|
| `add(interface)` | Register an interface, bind it to the agent (returns self) |
| `remove(interface)` | Deregister an interface (returns True if found) |
| `add_hook(hook)` | Add a gateway-level hook applying to all interfaces (returns self) |
| `remove_hook(hook)` | Remove a gateway-level hook (returns True if found) |
| `status(interface)` | Get `InterfaceStatus` for a specific interface |
| `statuses` | Property: `dict[platform_name, InterfaceStatus]` |
| `interfaces` | Property: list of registered interfaces |
| `is_healthy` | Property: True if all interfaces are running or pending |
| `serve(name=None)` | Start all interfaces (sync wrapper) |
| `aserve(name=None)` | Start all interfaces (async, blocks until cancelled) |

### InterfaceStatus

Lifecycle state of an interface managed by the gateway.

```python
from definable.agent.interface import InterfaceStatus

# Values:
InterfaceStatus.pending  # registered, not yet started
InterfaceStatus.starting  # start() called, not yet running
InterfaceStatus.running  # receiving messages
InterfaceStatus.restarting  # crashed, waiting for backoff before restart
InterfaceStatus.stopped  # stopped cleanly
InterfaceStatus.error  # crashed
```

The gateway auto-restarts crashed interfaces with exponential backoff (1s → 60s max).
An interface that ran stably for 60+ seconds resets its backoff on the next crash.

**Gateway lifecycle events** (emitted through the agent's EventBus):

| Event class | Emitted when |
|-------------|---|
| `InterfaceStartedEvent` | Interface starts successfully |
| `InterfaceStoppedEvent` | Interface stops cleanly |
| `InterfaceRestartedEvent` | Crashed interface is restarted (includes `restart_count`, `backoff_seconds`) |
| `InterfaceErrorEvent` | Interface crashes (includes `error_message`) |

### Identity Resolution

Maps `(platform, platform_user_id)` to a canonical user ID so that one user's memory
is unified across platforms. Sessions remain platform-scoped; identity linking is opt-in.

```python
from definable.agent.interface import (
  IdentityResolver,
  SQLiteIdentityResolver,
  PlatformIdentity,
)

resolver = SQLiteIdentityResolver("./identity.db")
await resolver.initialize()

# Link a user across platforms
await resolver.link("telegram", "tg_user_123", canonical_user_id="user_abc")
await resolver.link("slack", "U0123ABC", canonical_user_id="user_abc")

# Resolve to canonical ID
canonical = await resolver.resolve("telegram", "tg_user_123")  # "user_abc"

# Get all linked platform identities for a user
identities = await resolver.get_identities("user_abc")  # List[PlatformIdentity]

# Unlink one platform
removed = await resolver.unlink("telegram", "tg_user_123")  # True

await resolver.close()
# Or use as async context manager:
async with SQLiteIdentityResolver("./identity.db") as resolver:
  canonical = await resolver.resolve("telegram", "tg_user_123")
```

`PlatformIdentity` fields: `platform`, `platform_user_id`, `canonical_user_id`,
`username` (optional), `linked_at` (unix timestamp).

`IdentityResolver` is a `Protocol` — implement it for custom storage backends.

**Self-service `/link` flow** (enabled via `InterfaceGateway(enable_identity_linking=True)`):
- User sends `/link` on platform A → receives a 6-character code (valid for 5 minutes)
- User sends `/link CODE` on platform B → accounts linked automatically

### Errors

```python
from definable.agent.interface import (
  InterfaceError,  # Base — HTTP 500
  InterfaceConnectionError,  # HTTP 503 — platform connection failed
  InterfaceAuthenticationError,  # HTTP 401 — invalid credentials
  InterfaceRateLimitError,  # HTTP 429 — has retry_after attribute
  InterfaceMessageError,  # HTTP 400 — message send/receive failed
)
```

All errors carry a `platform` attribute for the originating platform name.

## Platform-Specific Details

### TelegramInterface

```python
from definable.agent.interface import TelegramInterface

interface = TelegramInterface(
  bot_token="YOUR_BOT_TOKEN",
  # Optional:
  # webhook_url="https://your.domain/telegram/webhook"  # prod webhook mode
  # polling_timeout=30
)
```

Runs in polling mode by default (suitable for development). Pass `webhook_url` to
enable webhook mode for production.

### DiscordInterface

```python
from definable.agent.interface import DiscordInterface

interface = DiscordInterface(
  bot_token="YOUR_DISCORD_BOT_TOKEN",
)
```

Requires `pip install 'definable[discord]'`. Uses discord.py Gateway for real-time events.
Long responses are split automatically to respect Discord's 2000-character limit.

### SlackInterface

```python
from definable.agent.interface import SlackInterface

interface = SlackInterface(
  bot_token="xoxb-...",
  app_token="xapp-...",  # required for Socket Mode
)
```

Requires `pip install 'definable[slack]'`. Supports Socket Mode (no public URL needed)
and Events API (webhook-based). Slack markdown is auto-converted to mrkdwn format.

### CLIInterface

```python
from definable.agent.interface import CLIInterface

# REPL mode (default)
interface = CLIInterface()

# TUI mode — requires pip install 'definable[cli]'
interface = CLIInterface(mode="tui")
```

Provides a Rich-based REPL with event visualization. TUI mode uses Textual for a
full-screen terminal UI with streaming output, token/cost metrics, and slash commands.

### CallInterface

```python
from definable.agent.interface import CallInterface

# Managed pipeline (Twilio ConversationRelay — simplest, ~500ms latency)
interface = CallInterface(
  provider="twilio",
  account_sid="AC...",
  auth_token="...",
  phone_number="+15551234567",
  pipeline="managed",
  welcome_message="Hello! How can I help you today?",
)

# Cascading pipeline (raw audio, pluggable STT/TTS, ~800-1200ms latency)
from definable.agent.interface.call.stt.deepgram import DeepgramSTT
from definable.agent.interface.call.tts.cartesia import CartesiaTTS

interface = CallInterface(
  provider="twilio",
  account_sid="AC...",
  auth_token="...",
  phone_number="+15551234567",
  pipeline="cascading",
  stt=DeepgramSTT(api_key="..."),
  tts=CartesiaTTS(api_key="..."),
)
```

Pipeline modes:

| Mode | Latency | Notes |
|------|---------|-------|
| `managed` | ~500ms | Telephony provider handles STT/TTS (Twilio only) |
| `cascading` | ~800-1200ms | Raw audio → STT → Agent → TTS; requires `stt=` and `tts=` |
| `realtime` | ~200-300ms | Speech-to-speech proxy; requires `realtime=` provider |

Note: Plivo does not support `pipeline="managed"`. Use `cascading` or `realtime` with Plivo.

HTTP routes are mounted on the `AgentRuntime`'s FastAPI server automatically.

### WebSocketInterface

```python
from definable.agent.interface import WebSocketInterface

interface = WebSocketInterface(
  path="/ws",
  heartbeat_interval=30,
  max_connections=100,
)
```

Wire protocol (JSON):

```
Client → Server: {"type": "message", "text": "Hello", "session_id": "...", "user_id": "..."}
Server → Client: {"type": "response", "content": "Hi!", "session_id": "...", "run_id": "..."}
                 {"type": "error", "message": "..."}
                 {"type": "heartbeat"}
```

Requires `AgentRuntime` with `enable_server=True` to mount the WebSocket endpoint.

### WhatsAppInterface

```python
from definable.agent.interface import WhatsAppInterface

interface = WhatsAppInterface(
  account_sid="AC...",
  auth_token="...",
  from_number="whatsapp:+14155238886",
  webhook_path="/whatsapp/webhook",
)
```

Webhook-based delivery via Twilio's WhatsApp API. Configure the Twilio webhook URL to
point to your server's webhook endpoint. Requires `AgentRuntime` with `enable_server=True`.

### EmailInterface

```python
from definable.agent.interface import EmailInterface

interface = EmailInterface(
  imap_host="imap.gmail.com",
  imap_port=993,
  smtp_host="smtp.gmail.com",
  smtp_port=587,
  email_address="agent@example.com",
  email_password="app-specific-password",
  poll_interval=30.0,  # seconds between IMAP checks
)
```

Polls IMAP for new messages and replies via SMTP. Maintains conversation threads using
`In-Reply-To` and `References` headers. Uses blocking IMAP/SMTP I/O in executor threads.

## Agent Integration

### Bind and serve

```python
from definable.agent import Agent
from definable.agent.interface import TelegramInterface, DiscordInterface, LoggingHook, AllowlistHook

agent = Agent(model="openai/gpt-4o-mini", instructions="You are a helpful assistant.")

telegram = TelegramInterface(bot_token="...")
telegram.add_hook(LoggingHook())
telegram.add_hook(AllowlistHook(allowed_user_ids={"123456"}))

discord = DiscordInterface(bot_token="...")

# Serve multiple interfaces — runs until KeyboardInterrupt
agent.serve(telegram, discord)
```

### Bind separately from serving

```python
telegram = TelegramInterface(bot_token="...")
telegram.bind(agent)
telegram.add_hook(LoggingHook())

async with telegram:
  await telegram.serve_forever()
```

### Pass interface at Agent construction

```python
telegram = TelegramInterface(bot_token="...")
agent = Agent(
  model="openai/gpt-4o-mini",
  interfaces=[telegram],
)
agent.serve()
```

## Multi-Interface serve()

`serve()` runs multiple interfaces concurrently with automatic restart on failure.

```python
from definable.agent.interface import serve, TelegramInterface, DiscordInterface

await serve(
  TelegramInterface(agent=agent, bot_token="..."),
  DiscordInterface(agent=agent, bot_token="..."),
  name="my-bot",
)
```

For advanced coordination (shared sessions, gateway hooks, status tracking), use
`InterfaceGateway` instead of `serve()`.

## Gotchas

| Issue | Fix |
|-------|-----|
| `InterfaceMessage.content` AttributeError | The field is `text`, not `content`. Use `msg.text`. |
| `InterfaceResponse.text` AttributeError | The field is `content`, not `text`. Use `resp.content`. |
| `CLIInterface(mode="tui")` ImportError | Install Textual: `pip install 'definable[cli]'` |
| `DiscordInterface` ImportError | Install discord.py: `pip install 'definable[discord]'` |
| `SlackInterface` ImportError | Install Slack SDK: `pip install 'definable[slack]'` |
| `CallInterface(provider="plivo", pipeline="managed")` ValueError | Plivo has no ConversationRelay. Use `pipeline="cascading"` or `"realtime"`. |
| `CallInterface(pipeline="cascading")` ValueError | Must pass both `stt=` and `tts=` providers. |
| `CallInterface(pipeline="realtime")` ValueError | Must pass `realtime=` provider. |
| `InterfaceGateway.add()` called after `aserve()` started | Interfaces must be added before `serve()` / `aserve()`. |
| `InterfaceGateway` with no interfaces | `aserve()` raises `ValueError`. Call `add()` first. |
| `SQLiteIdentityResolver` without `initialize()` | Call `await resolver.initialize()` before any resolve/link calls (auto-called on first use). |
| Session history not persisting across messages | Sessions are per-interface by default. Use `InterfaceGateway(shared_sessions=True)` across interfaces, or `Agent(memory=True)` for persistent memory. |

## See Also

- `agent/` — Agent class with `serve()` / `aserve()` methods
- `agent/runtime/` — AgentRuntime orchestrator
- `agent/auth/` — HTTP authentication (APIKeyAuth, JWTAuth, AllowlistAuth)
- `agent/security/` — SecurityConfig with RateLimitHook for interface-level throttling
- `utils/supervisor.py` — supervise_interfaces() with exponential backoff
