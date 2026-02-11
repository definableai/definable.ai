"""Interface supervisor — run multiple interfaces concurrently with auto-restart."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from definable.interfaces.base import BaseInterface
from definable.utils.log import log_error, log_info, log_warning

if TYPE_CHECKING:
  from definable.interfaces.identity import IdentityResolver

_STABILITY_THRESHOLD = 60.0  # seconds before backoff resets
_MAX_BACKOFF = 60.0  # maximum backoff delay in seconds


async def supervise_interfaces(
  *interfaces: BaseInterface,
  name: Optional[str] = None,
  identity_resolver: Optional["IdentityResolver"] = None,
) -> None:
  """Run multiple interfaces concurrently with automatic restart on failure.

  Acts as a supervisor: each interface runs in its own asyncio task.
  If an interface crashes (raises an exception), it is restarted with
  exponential backoff. If an interface stops cleanly (no exception),
  it is not restarted. When all interfaces have stopped cleanly,
  the function returns.

  On cancellation (e.g. Ctrl+C), all running tasks are cancelled and
  cleaned up, then the function returns normally.

  Args:
    *interfaces: One or more BaseInterface instances to run.
    name: Optional prefix for log messages (defaults to "serve").
    identity_resolver: Optional shared resolver for cross-platform user identity.
        Propagated to any interface that doesn't already have one.

  Raises:
    ValueError: If no interfaces are provided.
  """
  if not interfaces:
    raise ValueError("supervise_interfaces() requires at least one interface")

  # Propagate shared resolver to interfaces that don't have one
  if identity_resolver is not None:
    for iface in interfaces:
      if iface._identity_resolver is None:
        iface._identity_resolver = identity_resolver

  prefix = f"[{name}]" if name else "[serve]"

  # Backoff state: interface id -> (backoff_delay, last_start_time)
  backoff: Dict[int, Tuple[float, float]] = {}
  # Map task -> interface for the supervisor loop
  task_to_iface: Dict[asyncio.Task, BaseInterface] = {}

  def _start_task(iface: BaseInterface) -> asyncio.Task:
    """Create a task for an interface and track it."""
    task = asyncio.create_task(iface.serve_forever())
    task_to_iface[task] = iface
    iface_id = id(iface)
    _, _ = backoff.setdefault(iface_id, (0.0, 0.0))
    backoff[iface_id] = (backoff[iface_id][0], time.monotonic())
    return task

  # Start all interfaces
  for iface in interfaces:
    _start_task(iface)
    log_info(f"{prefix} Started {iface.config.platform or type(iface).__name__}")

  try:
    while task_to_iface:
      done, _ = await asyncio.wait(
        task_to_iface.keys(),
        return_when=asyncio.FIRST_COMPLETED,
      )

      for task in done:
        iface = task_to_iface.pop(task)
        iface_id = id(iface)
        iface_name = iface.config.platform or type(iface).__name__
        exc = task.exception() if not task.cancelled() else None

        if exc is not None:
          # Interface crashed — restart with backoff
          current_backoff, last_start = backoff.get(iface_id, (0.0, 0.0))
          elapsed = time.monotonic() - last_start

          if elapsed >= _STABILITY_THRESHOLD:
            # Was stable before failing — reset backoff
            next_backoff = 1.0
          else:
            next_backoff = min((current_backoff * 2) or 1.0, _MAX_BACKOFF)

          backoff[iface_id] = (next_backoff, 0.0)

          log_error(f"{prefix} {iface_name} crashed: {exc}")
          log_warning(f"{prefix} Restarting {iface_name} in {next_backoff:.1f}s")

          await asyncio.sleep(next_backoff)
          _start_task(iface)
          log_info(f"{prefix} Restarted {iface_name}")
        else:
          # Clean stop — don't restart
          log_info(f"{prefix} {iface_name} stopped cleanly")

  except asyncio.CancelledError:
    log_info(f"{prefix} Shutting down all interfaces")
    for task in task_to_iface:
      task.cancel()
    await asyncio.gather(*task_to_iface, return_exceptions=True)
    log_info(f"{prefix} All interfaces stopped")
