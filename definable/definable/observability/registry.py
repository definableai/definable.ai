"""Process-wide registry of agents that opted into observability.

The dashboard reads from this registry to populate the sidebar; the
playground endpoint looks up the live :class:`Agent` instance here
when it has a user request to forward.

Registration is keyed on ``agent.name``. Re-registering replaces the
prior reference (and warns once) — same-process duplicate names are
rare but valid (e.g. notebook reloads).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from definable.agent.agent import Agent

log = logging.getLogger(__name__)


class AgentRegistry:
  """Lightweight thread-safe registry of agent name → Agent.

  Synchronous on purpose — registration happens at Agent construct time
  which is not necessarily inside an event loop. The lock is a plain
  :class:`threading.Lock` since membership is the only contended state.
  """

  _instance: "AgentRegistry | None" = None
  _bootstrap_lock = threading.Lock()

  def __init__(self) -> None:
    self._agents: dict[str, Agent] = {}
    self._lock = threading.Lock()

  @classmethod
  def get(cls) -> AgentRegistry:
    """Return (or lazily create) the process-wide singleton."""
    if cls._instance is None:
      with cls._bootstrap_lock:
        if cls._instance is None:
          cls._instance = cls()
    return cls._instance

  def register(self, agent: Agent) -> None:
    with self._lock:
      if agent.name in self._agents:
        log.warning("Agent %r re-registered — replacing previous reference", agent.name)
      self._agents[agent.name] = agent

  def unregister(self, name: str) -> None:
    with self._lock:
      self._agents.pop(name, None)

  def list(self) -> list[Agent]:
    with self._lock:
      return list(self._agents.values())

  def lookup(self, name: str) -> Agent | None:
    with self._lock:
      return self._agents.get(name)

  def clear(self) -> None:
    """Drop every entry. Test-only — production should let registrations live."""
    with self._lock:
      self._agents.clear()
