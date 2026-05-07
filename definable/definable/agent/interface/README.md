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

| Adapter   | Status | Notes |
|-----------|--------|-------|
| websocket | ✅ ported | template — ~95 lines, see `websocket/interface.py` |
| telegram  | ⏳ stub   | platform glue preserved, port to new `Interface` base |
| whatsapp  | ⏳ stub   | Twilio + Baileys providers, port to new base |
| discord   | ⏳ stub   |  |
| slack     | ⏳ stub   |  |
| email     | ⏳ stub   | IMAP receive + SMTP send |
| desktop   | ⏳ stub   |  |
| call      | ⏳ stub   | Twilio + ConversationRelay |
| cli       | ❌ removed | use TUI library directly if needed |

Stubs keep their platform-specific code (Telegram bot polling, Baileys Node
sidecar, Discord intents, Slack websockets, IMAP loop, Twilio webhook handler,
etc) but reference the deleted `BaseInterface` / `InterfaceHook` /
`SessionManager` / `IdentityResolver` / `auth` modules. Each port replaces:

- `BaseInterface` -> `Interface` (this file's base)
- `InterfaceHook` system -> subscribe to `agent.events.on(EventType)`
- `SessionManager` -> per-conversation state lives in `FileMemory` + the
  platform's own session/thread model
- `IdentityResolver` -> resolve in `_convert` if you need user id
- `auth` -> guard inside `aopen` or `_convert`; reject early
- multi-interface concurrency -> `asyncio.gather(iface1.serve(), iface2.serve())`
  (the old `utils/supervisor.py` is gone — orchestrate in user code)

When porting an adapter, mark it ✅ here. mypy.ini excludes the unported
adapter directories; remove the entry once a directory is clean.

## Why not port them all today

Each adapter is ~250-3000 lines of platform-specific glue. Wholesale rewrite
risks shallow ports. Incremental port-on-demand is the right move — most users
(per CLAUDE.md note on Anandesh's E-Garuda + Clinic projects) wire Baileys
directly anyway and don't depend on the framework's adapters.
