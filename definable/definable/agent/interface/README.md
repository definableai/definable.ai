# Interfaces — channel adapters

A platform interface connects an `Agent` to a messaging channel. Each adapter
subclasses the `Interface` ABC in `base.py` and supplies the four
platform-specific hooks: `aopen`, `aclose`, `_convert` (platform message ->
prompt text), `_send` (text -> platform delivery). Everything else flows
through the new harness's `agent.events` bus and `agent.arun()`.

## The base contract

```python
from definable import Agent
from definable.agent.interface import Interface

class MyChannelInterface(Interface):
  def __init__(self, agent: Agent, **kwargs):
    super().__init__(agent)
    # store platform-specific config

  async def aopen(self) -> None: ...        # connect / login / start receiver
  async def aclose(self) -> None: ...       # clean teardown
  async def _convert(self, raw) -> str: ... # extract prompt from inbound
  async def _send(self, raw, reply): ...    # deliver outbound
```

The base provides:

- `handle(raw_message)` — default flow: convert -> agent.arun -> send.
- `serve()` — block until cancelled (override for poll-loop platforms).
- `__aenter__` / `__aexit__` — calls `aopen` / `aclose`.

Override `handle` if a platform needs richer behaviour (streaming reply chunks,
multi-message responses, typing indicators, etc).

## Status

| Adapter   | Status     | Lines (was → now) | Notes |
|-----------|------------|-------------------|-------|
| websocket | ✅ ported  | 250 → 95          | FastAPI WS server, JSON wire |
| email     | ✅ ported  | 391 → 211         | IMAP poll + SMTP send |
| discord   | ✅ ported  | 404 → 130         | discord.py gateway |
| telegram  | ✅ ported  | 2,333 → 145       | Bot API long-poll (httpx) |
| slack     | ✅ ported  | 1,902 → 130       | slack-bolt Socket Mode |
| whatsapp  | ✅ ported  | 2,162 → 2,162     | Pluggable: **Baileys** (personal, QR-scan) + **Twilio** |
| desktop   | ✅ ported  | 1,276 → 105       | Localhost WebSocket |
| call      | ❌ removed | 3,965 → 0         | Voice = different shape (audio streaming, STT/TTS). Spin off as `definable.voice` package when needed. |
| cli       | ❌ removed | 5,707 → 0         | Use TUI library directly |

All ported adapters import clean and pass mypy. Live smoke requires platform
credentials per adapter (telegram bot token, slack workspace, twilio account,
etc).

## Features stripped during port

Each port focuses on the core text-in/text-out flow. Bell-and-whistle features
present in the original were removed to keep ports honest:

- **discord** — media attachments
- **telegram** — typing circuit breaker, sticker cache, sliding-window rate
  limiter, formatting helpers, agent-controlled inline keyboards, callback
  handlers, slash commands
- **slack** — HTTP webhook mode (Socket Mode only), slash commands, Block Kit
  actions, modal submissions, shortcuts, reaction events
- **whatsapp** — *retained* the Baileys sidecar, pluggable provider abstraction,
  policy module, formatting + normalize helpers. Personal-WhatsApp-via-QR-scan
  is the headline use case
- **desktop** — macOS Vapor 4 sidecar bridge client (camera, screen, OCR,
  shell). That belongs as a Toolkit, not embedded in the interface.

To bring any of these back: subclass the adapter and override `handle` or
subscribe to `agent.events.on(EventType)` from the harness's bus.

## Voice / call

Real-time voice doesn't fit the request/reply Interface contract. The
original `call/` had 24 files and ~4,000 lines of bidirectional audio
streaming, STT, TTS, telephony providers (Twilio + Plivo), and three
pipeline modes (Managed / Cascading / Realtime). When voice agents are
needed, build a separate `definable.voice` package with its own contract
(audio in / audio out, barge-in, VAD, etc) — same way `definable.flow` was
spun off for the workflow-manifest use case.

## CLI

5,707 lines of TUI / commands / completer / renderers — too heavy for the
framework. Use a TUI library (Textual, prompt-toolkit, Rich) directly in user
code if you want a terminal REPL.

## Multi-interface concurrency

The old `utils/supervisor.py` (with auto-restart + exponential backoff) is
gone. Run multiple interfaces concurrently with stdlib instead::

    async def run_all():
      tg = TelegramInterface(agent, bot_token=...)
      sl = SlackInterface(agent, bot_token=..., app_token=...)
      async with tg, sl:
        await asyncio.gather(tg.serve(), sl.serve())
