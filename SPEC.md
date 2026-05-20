# SPEC: Interface → Channel rename + Agent-owned multi-channel lifecycle

**Status**: draft, awaiting confirmation
**Author**: hash
**Date**: 2026-05-11
**Branch**: `feat/harness-v2` (extension)

---

## 1. Objective

Rename `Interface` → `Channel`. Move multi-channel lifecycle ownership from user code into `Agent`. Single canonical attach path. Hard rename — no compat alias.

### Why

Current DX:

```python
agent = Agent(name="bot", model="gpt-4o")
tg = TelegramInterface(agent, bot_token="...")
async with tg:
    await tg.serve()
```

Each interface owns its own `serve()` loop. Multi-channel = manual `asyncio.gather(tg.serve(), wa.serve(), ws.serve())`. User wrangles event loop, lifecycle, signal handling.

New DX:

```python
agent = Agent(
    name="bot",
    model="gpt-4o",
    channels=[
        TelegramChannel(bot_token=os.environ["TELEGRAM_BOT_TOKEN"]),
        DiscordChannel(token=os.environ["DISCORD_TOKEN"]),
        WhatsappChannel(provider="baileys"),
    ],
)
agent.serve()    # blocks, drives all channels with supervision
```

Or for single-turn programmatic use (no channels needed):

```python
result = agent.run("hello world")
```

### Target users

Definable framework users building agents that need:
- Programmatic invocation (`agent.run`/`agent.arun`)
- Long-running multi-channel deployment (Telegram + WhatsApp + WebSocket + ...)
- Local testing (`CLIChannel`)

### Non-goals (v1)

- Hot-attach channels after `serve()` has started — defer to v2.
- Cross-channel identity linking — defer to v2.
- Per-channel kill switches / circuit breakers — defer to v2.

---

## 2. Architecture decisions

### 2.1 Verbs — clean split, no overloading

| Method | Sync/Async | Purpose | Returns |
|--------|-----------|---------|---------|
| `agent.run(input, *, session_id=None, user_id=None)` | sync | single turn | `RunResult` |
| `agent.arun(input, *, session_id=None, user_id=None)` | async | single turn | `RunResult` |
| `agent.serve()` | sync | block driving channels | `None` |
| `agent.aserve()` | async | block driving channels | `None` |

`run` = execute. `serve` = host. Each verb has one signature, one meaning.

### 2.2 Channel boundary — agent unaffected

- `Channel.bind(agent: Agent) -> None` — sync, stash ref only. No I/O.
- Channels call `agent.arun(prompt, session_id=...)` and read `RunResult.content`.
- Channels MUST NOT mutate `agent.memory`, `agent.tools`, `agent.events`, or any other internal state. Enforced by convention + code review.
- Memory mutations during a turn happen via the agent's own tools firing — that's the agent acting on itself, not the channel acting on the agent.

### 2.3 Identity — `session_id` + `user_id`, per-call

Two distinct identifiers. Both per-call kwargs on `arun()`/`run()`. Both removed from `Agent.__init__`.

| Identifier | Meaning | Scope | Example |
|-----------|---------|-------|---------|
| `session_id` | Conversation thread | Channel-namespaced | `"telegram:chat:12345"` |
| `user_id` | Person identity (sender) | Cross-channel-portable (raw, NOT namespaced) | `"anandesh"`, `"+919876543210"`, `"12345"` |

Why two:
- **1-on-1 chat**: `session_id == user_id` semantically (one thread, one user). But still pass both — the meaning is different.
- **Group chat**: one `session_id` (the group), many `user_id`s (the senders). Memory/permissions/analytics need user-awareness inside a shared session.
- **Cross-channel linking (v2)**: same person on Telegram + WhatsApp = same `user_id` if app links them. `session_id` stays per-channel.

`session_id` namespaced (channel-prefixed) to prevent numeric-ID collision across channels. `user_id` NOT namespaced — kept raw so cross-channel identity linking is possible later without re-keying.

```python
class Channel(ABC):
    name: ClassVar[str]   # "telegram", "discord", "whatsapp"

    def make_session_id(self, native_id: str | int) -> str:
        return f"{self.name}:{native_id}"

    @abstractmethod
    def _native_session_id(self, raw_message: Any) -> str | int:
        """Per-thread identifier (chat_id, conversation_id, etc.)."""

    @abstractmethod
    def _native_user_id(self, raw_message: Any) -> str | int | None:
        """Per-sender identifier. None for anonymous/system messages."""
```

Format examples:

| Channel | `session_id` | `user_id` |
|---------|-------------|-----------|
| Telegram 1-on-1 | `"telegram:chat:12345"` | `"12345"` (from.id) |
| Telegram group | `"telegram:chat:-100789"` | `"12345"` (sender's from.id) |
| WhatsApp DM | `"whatsapp:chat:+919876543210"` | `"+919876543210"` |
| WhatsApp group | `"whatsapp:chat:120363045@g.us"` | `"+919876543210"` |
| Discord | `"discord:guild_123:channel_456"` | `"<author_id>"` |
| WebSocket | `"websocket:<connection_id>"` | `"<auth_principal>"` or `None` |
| CLI | `"cli:local"` | `os.environ.get("USER", "local")` |

`None` semantics:
- `session_id=None` → anonymous singleton (CLI script, one-shot). Memory dir: `{agent_name}/_anonymous/`.
- `user_id=None` → unknown sender. Tools / context that need it must handle `None`.

`user_id` propagation:
- Channel `handle()` injects both per-call.
- `core.run()` accepts `user_id` and threads it through.
- Memory operations may use `user_id` for per-user notes within a shared session (v2 feature; v1 only stores the value alongside, doesn't index).
- Available to tools via context — **deferred**: no context injection layer exists in current harness. v1 just propagates; tools read via closure if needed. v2 adds explicit context.

### 2.4 Memory — per-session sharding

`FileMemory` shards by `session_id` at the directory level:

```
.definable/memory/{agent_name}/{session_id}/
```

N channels × N users = N directories. No contention. No locks. Concurrency for free.

Cross-session shared knowledge (if introduced later) gets its own concurrency strategy (likely SQLite WAL), separate from per-session conversational memory.

### 2.5 Channel lifecycle — supervised TaskGroup

```python
async def aserve(self) -> None:
    if not self._channels:
        raise ConfigurationError(
            "agent.serve() requires at least one channel. "
            "Options: (1) add a Channel to channels=[...], "
            "(2) use agent.run(input) for single-turn invocation, "
            "(3) use CLIChannel() for local testing."
        )
    await self.aopen()   # opens toolkits/MCP
    try:
        async with asyncio.TaskGroup() as tg:
            for ch in self._channels:
                tg.create_task(self._supervise(ch))
    finally:
        await asyncio.gather(
            *(ch.aclose() for ch in self._channels),
            return_exceptions=True,
        )
        await self.aclose()
```

### 2.6 Channel error isolation — per-channel supervisor

```python
async def _supervise(self, ch: Channel) -> None:
    backoff = ExponentialBackoff(min=1.0, max=60.0)
    while not self._stopping:
        try:
            await ch.astart()    # blocks for channel lifetime
            return               # clean exit, don't restart
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_error(f"[channel:{ch.name}] crashed: {e}")
            self.events.emit(ChannelErrored(channel=ch.name, error=str(e)))
            await asyncio.sleep(backoff.next())
```

One channel crashing logs + emits `ChannelErrored` event + restarts with exponential backoff capped at 60s. Other channels unaffected. `CancelledError` propagates for graceful shutdown.

### 2.7 SIGINT / shutdown

`KeyboardInterrupt` propagates through `asyncio.run()` → `TaskGroup` cancellation → channel `aclose()` in `finally`. Standard pattern.

```python
def serve(self) -> None:
    try:
        asyncio.run(self.aserve())
    except KeyboardInterrupt:
        pass  # cleanup already happened in aserve's finally
```

### 2.8 `add_channel()` — no hot-attach in v1

```python
def add_channel(self, ch: Channel) -> None:
    if self._serving:
        raise RuntimeError(
            "add_channel() cannot be called after serve() has started. "
            "Add all channels before calling serve()."
        )
    ch.bind(self)
    self._channels.append(ch)
```

YAGNI. Re-evaluate when a user requests it.

### 2.9 `None` content — silent skip with debug log

`handle()` (shown in 3.2) skips send when `result.content is None` and logs at DEBUG. Tool-only turns produce no reply by design.

### 2.10 Explicit `session_id` / `user_id` precedence

Explicit kwargs win. No validation, no normalization, no coercion. Caller owns key consistency.

| `session_id` | `user_id` | Behaviour |
|---|---|---|
| `None` | `None` | Memory dir = `_anonymous/`. No user attribution. All anonymous calls **share** that memory. |
| `None` | `"anandesh"` | Anonymous session, attributed turn. Memory dir = `_anonymous/`. `user_id` stored as metadata only (v1). |
| `"my-debug"` | `None` | Custom session, no user attribution. Memory dir = `my-debug/`. |
| `"my-debug"` | `"anandesh"` | Both explicit. As-is. No prefix added. Memory dir = `my-debug/`, turn attributed to anandesh. |
| `"telegram:chat:12345"` | `"12345"` | Allowed. Caller intentionally writes into Telegram's keyspace — useful for **replay/testing/seeding state** that a live channel will pick up. No collision guard. |

Rules:
- **No coercion** — caller's string used verbatim. No lowercasing, no whitespace strip, no prefix injection.
- **No registry of channel namespaces** — agent doesn't know about `telegram:` prefix. Only `Channel.make_session_id` does.
- **Empty string** — treated as `None` (filesystem can't take empty dir name).
- **Filesystem-unsafe chars** — caller's problem in v1. Document the convention. v2 adds `_safe_id()` filter.
- **Concurrent same-`session_id` calls** — N parallel `arun()` calls with same key hit same memory dir. No lock in v1. Channels avoid this naturally (one-task-per-chat). Programmatic callers responsible.

Programmatic + live channels coexist cleanly because keys differ:

```python
agent.serve()  # Telegram fills "telegram:chat:12345" automatically

# In a separate thread / process / pre-serve invocation:
result = await agent.arun("debug query", session_id="ops:debug", user_id="hash")
```

Both write to same `FileMemory` root, different subdirs. No bleed.

**Power-user case** — caller passes exactly the same key a live channel uses → **intentional write into that user's memory**. Used for replay tooling, fixture seeding, admin ops ("remember user X is VIP"). No guard. Documented as advanced usage in `arun()` docstring.

---

## 3. Commands / API surface

### 3.1 Agent (new shape)

```python
class Agent:
    def __init__(
        self,
        *,
        name: str,
        model: Model | str,
        instructions: str | None = None,
        tools: list[Function] | None = None,
        toolkits: list[Toolkit] | None = None,
        mcp: list[MCPToolkit] | None = None,
        skills: list[Skill] | None = None,
        memory: FileMemory | bool = False,
        max_turns: int = 50,
        observability: Any | bool = False,
        channels: list[Channel] | None = None,    # NEW
    ) -> None: ...

    # Single-turn invocation
    def run(self, input: str, *,
            session_id: str | None = None,
            user_id: str | None = None,
            output_schema: Any | None = None) -> RunResult: ...
    async def arun(self, input: str, *,
                   session_id: str | None = None,
                   user_id: str | None = None,
                   stream: bool = False,
                   output_schema: Any | None = None) -> RunResult | AsyncIterator[Event]: ...

    # Multi-channel hosting
    def serve(self) -> None: ...
    async def aserve(self) -> None: ...

    # Channel management
    def add_channel(self, ch: Channel) -> None: ...    # NEW

    # Lifecycle (unchanged)
    async def aopen(self) -> None: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> Agent: ...
    async def __aexit__(self, *args: Any) -> None: ...
```

Removed from `__init__`: `session_id` (moved to per-call).

### 3.2 Channel (new ABC, replaces Interface)

```python
class Channel(ABC):
    name: ClassVar[str]   # subclasses override: "telegram", "discord", etc.

    def __init__(self) -> None:
        self.agent: Agent | None = None

    def bind(self, agent: Agent) -> None:
        """Sync. Stash agent ref. No I/O."""
        self.agent = agent

    def make_session_id(self, native_id: str | int) -> str:
        return f"{self.name}:{native_id}"

    # Subclass contract
    @abstractmethod
    async def aopen(self) -> None: ...
    @abstractmethod
    async def aclose(self) -> None: ...
    @abstractmethod
    async def astart(self) -> None:
        """Block for channel lifetime. Drive the poll/socket/listener loop here."""
    @abstractmethod
    async def _convert(self, raw_message: Any) -> str: ...
    @abstractmethod
    async def _send(self, raw_message: Any, reply: str) -> None: ...
    @abstractmethod
    def _native_session_id(self, raw_message: Any) -> str | int:
        """Pull the per-thread identifier (chat_id, conversation_id, etc.)."""
    @abstractmethod
    def _native_user_id(self, raw_message: Any) -> str | int | None:
        """Pull the per-sender identifier. None for anonymous/system messages."""

    # Base-provided
    async def handle(self, raw_message: Any) -> None:
        """convert → arun(input, session_id, user_id) → send."""
        prompt = await self._convert(raw_message)
        if not prompt:
            return
        session_id = self.make_session_id(self._native_session_id(raw_message))
        native_user = self._native_user_id(raw_message)
        user_id = str(native_user) if native_user is not None else None
        result = await self.agent.arun(prompt, session_id=session_id, user_id=user_id)
        if result.content:
            await self._send(raw_message, result.content)
        else:
            log_debug(f"[channel:{self.name}] turn produced no user-facing content for {session_id}")
```

Removed: `serve()` method. Removed: `stop()`. Renamed flow: `Channel.astart()` replaces `Interface.serve()` semantically.

### 3.3 CLIChannel (new, first-class)

```python
class CLIChannel(Channel):
    name: ClassVar[str] = "cli"

    async def aopen(self) -> None: ...
    async def aclose(self) -> None: ...
    async def astart(self) -> None:
        """Read stdin loop. Print agent replies to stdout."""
    async def _convert(self, raw_message: str) -> str: ...
    async def _send(self, raw_message: str, reply: str) -> None: ...
    def _native_session_id(self, raw_message: str) -> str:
        return "local"
    def _native_user_id(self, raw_message: str) -> str | None:
        return os.environ.get("USER") or "local"
```

Local testing in two lines:

```python
Agent(name="bot", model="gpt-4o", channels=[CLIChannel()]).serve()
```

### 3.4 New event types

```python
@dataclass(frozen=True)
class ChannelErrored(Event):
    channel: str
    error: str
```

### 3.5 New exception

```python
class ConfigurationError(DefinableError):
    """Raised when agent is misconfigured (e.g. serve() with zero channels)."""
```

---

## 4. Project structure changes

### 4.1 Filesystem rename

```
definable/agent/interface/  →  definable/agent/channel/
definable/agent/interface/base.py  →  definable/agent/channel/base.py
definable/agent/interface/telegram/  →  definable/agent/channel/telegram/
definable/agent/interface/discord/  →  definable/agent/channel/discord/
definable/agent/interface/slack/  →  definable/agent/channel/slack/
definable/agent/interface/whatsapp/  →  definable/agent/channel/whatsapp/
definable/agent/interface/email/  →  definable/agent/channel/email/
definable/agent/interface/desktop/  →  definable/agent/channel/desktop/
definable/agent/interface/websocket/  →  definable/agent/channel/websocket/
```

New file:

```
definable/agent/channel/cli/interface.py    # CLIChannel
```

### 4.2 Class renames

| Old | New |
|-----|-----|
| `Interface` | `Channel` |
| `TelegramInterface` | `TelegramChannel` |
| `DiscordInterface` | `DiscordChannel` |
| `SlackInterface` | `SlackChannel` |
| `WhatsAppInterface` | `WhatsappChannel` |
| `EmailInterface` | `EmailChannel` |
| `DesktopInterface` | `DesktopChannel` |
| `WebSocketInterface` | `WebsocketChannel` |

### 4.3 Test rename

```
definable/tests/agent/interface/  →  definable/tests/agent/channel/
smoke/wave7_interfaces.py  →  smoke/wave7_channels.py
```

### 4.4 Docs / wiki / memory sweep

Update references in:
- `README.md`
- `MEMORY.md` (project root + ~/.claude/projects/...)
- `.claude/brain/memory/INDEX.md` + topic files mentioning interfaces
- `.claude/brain/wiki/` (architecture.md, api-surface.md, etc.)
- `docs/` MDX pages
- Code comments / docstrings throughout `definable/`

### 4.5 No new abstractions

- Only new class: `Channel` (replaces `Interface`).
- Only new module: `agent/channel/cli/`.
- Only new event: `ChannelErrored`.
- Only new exception: `ConfigurationError`.
- Backoff helper inline (no new util module) unless one already exists.

---

## 5. Code style

Follow existing project conventions (see `ruff.toml`, `framework-patterns` skill):
- 2-space indent
- 150 char line length
- Async-first
- `from __future__ import annotations`
- No comments unless explaining non-obvious WHY
- Type hints throughout

---

## 6. Testing strategy

### 6.1 Unit tests (must pass)

- All 894 existing unit tests green after rename.
- New tests for `Agent.serve` / `aserve`:
  - Empty channels → `ConfigurationError`
  - Single channel, normal completion path
  - Multi-channel, all start + clean shutdown on cancel
  - One channel raises → other channels keep running + `ChannelErrored` emitted + backoff
- New tests for `Channel.make_session_id`:
  - Format correctness per channel name
  - Numeric + string native IDs
- New tests for `_native_user_id`:
  - 1-on-1 chat → user_id == sender (not chat_id)
  - Group chat → user_id == sender, session_id == group
  - `None` user_id propagates cleanly through arun
- New tests for `user_id` propagation:
  - `arun(input, session_id="s", user_id="u")` reaches `core.run()` with both
  - Missing `user_id` (None) does not raise
  - Memory stores `user_id` alongside session entries (v1: as metadata, not indexed)
- New tests for explicit-ID precedence (matrix from §2.10):
  - `(None, None)` → memory dir `_anonymous/`
  - `(None, "u")` → `_anonymous/`, attributed
  - `("s", None)` → `s/`, unattributed
  - `("s", "u")` → `s/`, attributed
  - Empty string `session_id` / `user_id` → coerced to `None`
  - Caller passing `"telegram:chat:12345"` writes into same dir as a Telegram-driven turn (collision-by-design proof)
  - No mutation of caller's input (verbatim propagation)
- New tests for `add_channel`:
  - Pre-serve append works
  - Post-serve raises `RuntimeError`
- New tests for `Agent.run` (sync wrapper):
  - Returns `RunResult`
  - Propagates exceptions correctly
- New tests for `CLIChannel`:
  - Stdin input → agent.arun → stdout reply
  - Empty input handled
- `FileMemory` sharding tests:
  - Two `session_id`s → two distinct directories
  - Concurrent writes to different sessions don't collide

### 6.2 Smoke tests (must pass)

- `smoke/wave7_channels.py` — full multi-channel composite (CLI + WebSocket + mock telegram-style).
- Power-user composite (existing) updated for new API, must stay green.

### 6.3 Quality gates

All four must pass before commit:
- `ruff check definable/`
- `ruff format --check definable/`
- `mypy definable/definable/ definable/tests/`
- `pytest definable/tests/`

### 6.4 Manual verification

- Single-channel example (Telegram) ran against live bot at least once.
- CLI channel REPL session ran end-to-end.

---

## 7. Boundaries

### Always
- Hard rename — no `Interface` alias, no `TelegramInterface` shim.
- Channel-namespaced session IDs everywhere (channel-side `make_session_id`).
- Raw (non-namespaced) `user_id` — keep portable for cross-channel linking.
- Both `session_id` and `user_id` per-call kwargs, never constructor.
- Per-session memory directories.
- `TaskGroup` + supervisor for channel lifecycle.
- `ConfigurationError` on empty `serve()`.
- Exponential backoff capped at 60s on channel crash.
- Explicit kwargs win — no validation/coercion of caller-supplied IDs.
- Empty-string `session_id` / `user_id` → treat as `None`.

### Ask first
- Adding any new public class beyond `Channel`, `CLIChannel`.
- Adding any new event type beyond `ChannelErrored`.
- Changing single-turn `arun()` semantics beyond adding `session_id` kwarg.
- Touching memory store internals beyond directory sharding.

### Never
- Hot-attach channels at runtime (defer to v2).
- Cross-channel identity linking (defer to v2).
- Channel mutating agent state directly (boundary violation).
- Re-introducing `Interface.serve()` on Channel.
- Backward-compat alias for `Interface` (hard rename means hard rename).
- Locks in `FileMemory` (sharding obviates need).
- Namespacing `user_id` (kills v2 cross-channel linking).
- Channel computing `user_id` from `chat_id` in group chats (must use sender id).
- Normalizing / coercing / rejecting caller-supplied `session_id` or `user_id`.
- Sanitizing IDs for filesystem safety (v1 — caller's responsibility; v2 adds `_safe_id`).

---

## 8. Migration plan (build phases)

1. **Skeleton** — create `agent/channel/base.py` with new `Channel` ABC.
2. **Agent surface** — add `channels` kwarg, `add_channel`, `run`, `serve`, `aserve`. Remove `session_id` from `__init__`. Add `session_id` + `user_id` kwargs to `run` / `arun`. Thread `user_id` through `core.run()`.
3. **Memory sharding** — `FileMemory(root)` becomes session-aware; per-session subdirs.
4. **Supervise / backoff** — `_supervise()` + `ChannelErrored` event.
5. **Port channels** — Telegram → Discord → Slack → WhatsApp → Email → Desktop → WebSocket. One PR per channel? Single PR for all? TBD per `agent-skills:plan`.
6. **CLIChannel** — new module.
7. **Rename `interface/` → `channel/`** — filesystem move + import rewrites.
8. **Test + smoke updates** — rename test dirs, update assertions.
9. **Docs sweep** — README, MEMORY.md, wiki, MDX.
10. **Quality gates** — all four green.
11. **Review** — `agent-skills:review`.
12. **Ship** — `agent-skills:ship`, commit, PR.

---

## 9. Open questions

None at draft stage — all 8 issues from design discussion resolved. Ready for `agent-skills:plan` once user confirms this spec.
