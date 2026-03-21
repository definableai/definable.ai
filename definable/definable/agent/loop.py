"""Agent loop data types and cancel token.

Pure data types used by the Agent's core loop. No business logic,
no dependency on Agent.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from definable.model.response import ToolExecution
from definable.agent.run.requirement import RunRequirement


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class Cancelled(Exception):
  """Raised when a CancelToken is triggered during a run."""


@dataclass
class CancelToken:
  """Cooperative cancellation token for agent runs.

  Call ``cancel()`` from any thread/task to request cancellation.
  The loop calls ``check()`` at each iteration boundary.
  """

  _cancelled: bool = False
  _event: asyncio.Event = field(default_factory=asyncio.Event)

  def cancel(self) -> None:
    """Request cancellation."""
    self._cancelled = True
    self._event.set()

  def check(self) -> None:
    """Raise :class:`Cancelled` if cancellation was requested."""
    if self._cancelled:
      raise Cancelled("Run cancelled via CancelToken")

  @property
  def is_cancelled(self) -> bool:
    return self._cancelled

  # Backwards compat alias used by old code
  def raise_if_cancelled(self) -> None:
    self.check()


# ---------------------------------------------------------------------------
# Tool dispatch results
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
  """Result of a single tool execution within the loop."""

  tool_call_id: Optional[str] = None
  tool_name: str = ""
  result: Optional[str] = None
  error: Optional[str] = None
  should_stop: bool = False
  is_paused: bool = False
  requirement: Optional[RunRequirement] = None
  tool_execution: Optional[ToolExecution] = None
  events: list = field(default_factory=list)


@dataclass
class ToolBatchResult:
  """Result of executing a batch of tool calls (parallel + sequential)."""

  results: List[ToolResult]
  events: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Old names → new names (AgentCancelled was the old exception name)
AgentCancelled = Cancelled
CancellationToken = CancelToken
