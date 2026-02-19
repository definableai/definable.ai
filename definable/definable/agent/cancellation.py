"""Cooperative cancellation for agent runs."""

from dataclasses import dataclass


class AgentCancelled(Exception):
  """Raised when an agent run is cancelled via CancellationToken."""

  pass


@dataclass
class CancellationToken:
  """Cooperative cancellation token for agent runs.

  Create a token, pass it to ``agent.arun(cancellation_token=token)``,
  and call ``token.cancel()`` from any coroutine or thread to stop the run.

  The loop checks ``raise_if_cancelled()`` at safe points (before each
  model call and before each tool execution) and raises ``AgentCancelled``.
  """

  _cancelled: bool = False

  def cancel(self) -> None:
    """Request cancellation. Thread-safe (single bool write)."""
    self._cancelled = True

  @property
  def is_cancelled(self) -> bool:
    return self._cancelled

  def raise_if_cancelled(self) -> None:
    """Raise ``AgentCancelled`` if cancellation was requested."""
    if self._cancelled:
      raise AgentCancelled("Run was cancelled")
