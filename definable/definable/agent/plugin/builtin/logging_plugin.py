"""LoggingPlugin — structured logging for agent runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, FrozenSet, Optional

from definable.agent.plugin.base import Plugin
from definable.utils.log import log_debug, log_info

if TYPE_CHECKING:
  from definable.agent.agent import Agent
  from definable.agent.pipeline.state import LoopState


class LoggingPlugin(Plugin):
  """Logs pipeline phase transitions and run lifecycle events.

  Registers before/after hooks on all phases (wildcard) to log
  timing and state transitions. Useful for debugging and auditing.

  Args:
    verbose: If True, log full state details (default: summary only).
    log_fn: Custom log function (default: ``log_info``).

  Example::

    agent = Agent(
      model="gpt-4o",
      plugins=[LoggingPlugin(verbose=True)],
    )
  """

  def __init__(self, *, verbose: bool = False, log_fn: Optional[Any] = None) -> None:
    self._verbose = verbose
    self._log = log_fn or log_info

  @property
  def name(self) -> str:
    return "logging"

  @property
  def description(self) -> str:
    return "Structured logging for pipeline phases and run lifecycle."

  @property
  def modifies(self) -> FrozenSet[str]:
    return frozenset({"*"})

  async def on_load(self, agent: "Agent") -> None:
    agent.pipeline.hook("before:*", self._before_phase, priority=100)
    agent.pipeline.hook("after:*", self._after_phase, priority=-100)

  async def _before_phase(self, state: "LoopState") -> "LoopState":
    self._log(f"[logging] Entering phase: {state.phase}")
    if self._verbose:
      log_debug(f"[logging]   messages={len(state.all_messages)} tools={len(state.tools)}")
    return state

  async def _after_phase(self, state: "LoopState") -> "LoopState":
    self._log(f"[logging] Completed phase: {state.phase}")
    if self._verbose and state.content:
      preview = str(state.content)[:100]
      log_debug(f"[logging]   output_preview={preview!r}")
    return state
