"""Interface — minimal base class for connecting agents to messaging platforms.

The new shape:

  class MyInterface(Interface):
    async def aopen(self) -> None: ...        # connect to platform
    async def aclose(self) -> None: ...       # disconnect
    async def _convert(self, raw) -> str: ... # platform message -> prompt text
    async def _send(self, raw, reply: str): ...   # deliver agent reply

  iface = MyInterface(agent)
  async with iface:
    await iface.serve()                       # listen forever

Whatever lifecycle the platform needs (long-poll loop, websocket, webhook
server) the subclass owns. The base only knows how to take a converted
prompt, dispatch through `agent.arun`, and pass the reply back to `_send`.
Observability is via `agent.events.subscribe(...)` — the interface does
not own a hook system.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from definable.agent.agent import Agent


class Interface(ABC):
  """Abstract base for platform interfaces (Telegram, WhatsApp, WebSocket, ...).

  Subclasses implement four hooks: aopen, aclose, _convert, _send. The
  base orchestrates: aopen on enter, dispatch each inbound message
  through `agent.arun()`, deliver via `_send`, aclose on exit.
  """

  def __init__(self, agent: Agent) -> None:
    self.agent = agent
    self._serving = False

  # ---- platform-specific hooks -------------------------------------------

  @abstractmethod
  async def aopen(self) -> None:
    """Connect to the platform (start websocket, bind webhook, login, ...)."""

  @abstractmethod
  async def aclose(self) -> None:
    """Disconnect cleanly."""

  @abstractmethod
  async def _convert(self, raw_message: Any) -> str:
    """Translate a raw platform message into a prompt string for the agent."""

  @abstractmethod
  async def _send(self, raw_message: Any, reply: str) -> None:
    """Deliver the agent's reply back via the platform."""

  # ---- orchestration ------------------------------------------------------

  async def handle(self, raw_message: Any) -> None:
    """Default flow: convert raw -> arun -> send. Subclasses can override
    for richer behaviour (streaming reply, multi-message responses, etc)."""
    prompt = await self._convert(raw_message)
    if not prompt:
      return
    result = await self.agent.arun(prompt)
    # arun returns AsyncIterator[Event] when stream=True (we always use
    # non-streaming here so the call returns RunResult).
    content = getattr(result, "content", None)
    if content:
      await self._send(raw_message, content)

  async def serve(self) -> None:
    """Hold the interface open until cancelled. Subclasses with a built-in
    blocking receive loop (e.g. polling) override this to drive that loop;
    interfaces that listen on callbacks rely on this base impl."""
    self._serving = True
    try:
      while self._serving:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
      pass

  def stop(self) -> None:
    self._serving = False

  # ---- lifecycle ---------------------------------------------------------

  async def __aenter__(self) -> Interface:
    await self.aopen()
    return self

  async def __aexit__(self, *args: Any) -> None:
    with contextlib.suppress(Exception):
      await self.aclose()
